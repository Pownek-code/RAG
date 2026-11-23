"""Disk caches that speed up cold starts and repeated queries.

A cache entry is only ever reused for exactly the inputs it was computed
from: its key includes a version of everything the value depends on, so
an entry made from an older index, model or code version is simply never
looked up again. Deleting a cache directory is always safe.
"""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def fingerprint_files(paths: list[Path]) -> str:
    """Return a hash of the content of files, so it changes with them.

    A missing file is hashed as missing, so creating or deleting one also
    changes the fingerprint.
    """
    digest = hashlib.sha256()

    for path in paths:
        digest.update(str(path).encode("utf-8") + b"\0")

        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")

        digest.update(b"\0")

    return digest.hexdigest()


def write_atomically(path: Path, data: bytes) -> None:
    """Write a file so that readers see the old file or the whole new one.

    The data goes to a temporary file in the same directory, which is then
    renamed over ``path``; a rename within one directory is atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as file:
        file.write(data)
        temporary_path = Path(file.name)

    os.replace(temporary_path, path)


class IndexCache:
    """Keeps arrays derived from an index in their ready-to-use form.

    It holds one version at a time: saving another version replaces it.
    The version file is removed first and written last, so arrays from an
    interrupted save are never read.
    """

    VERSION_FILE_NAME = "version.txt"

    def __init__(self, directory: Path) -> None:
        """Set the directory that holds the cached arrays."""
        self._directory = directory

    def load(self, version: str) -> dict[str, NDArray[np.generic]] | None:
        """Return the cached arrays of ``version``, or ``None`` if absent.

        The arrays are memory-mapped: the operating system reads them from
        disk only when they are used.
        """
        try:
            cached_version = (
                self._directory / self.VERSION_FILE_NAME
            ).read_text(encoding="utf-8")

            if cached_version != version:
                return None

            return {
                path.stem: np.load(path, mmap_mode="r")
                for path in sorted(self._directory.glob("*.npy"))
            }
        except (OSError, ValueError):
            return None

    def save(
        self,
        version: str,
        arrays: Mapping[str, NDArray[np.generic]],
    ) -> None:
        """Store the arrays of ``version`` in place of any older version.

        A cache that cannot be written is only a missed speed-up, so the
        error is logged and ignored.
        """
        version_path = self._directory / self.VERSION_FILE_NAME

        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            version_path.unlink(missing_ok=True)

            for name, array in arrays.items():
                np.save(self._directory / f"{name}.npy", array)

            write_atomically(version_path, version.encode("utf-8"))
        except OSError as error:
            logger.warning("Could not write the index cache: %s", error)


class QueryCache:
    """Stores query results as small JSON files named by their key.

    ``CACHE_VERSION`` is part of every key. Bump it when code that shapes
    a cached result changes (the prompt, the fusion constants...), since
    such a change is invisible in the inputs.
    """

    CACHE_VERSION = 1

    def __init__(self, directory: Path) -> None:
        """Set the directory that holds the cached results."""
        self._directory = directory

    def get(self, key: Mapping[str, object]) -> str | None:
        """Return the cached value for ``key``, or ``None`` on a miss."""
        try:
            return self._path(key).read_text(encoding="utf-8")
        except OSError:
            return None

    def put(self, key: Mapping[str, object], value: str) -> None:
        """Store ``value`` under ``key``.

        A cache that cannot be written is only a missed speed-up, so the
        error is logged and ignored.
        """
        try:
            write_atomically(self._path(key), value.encode("utf-8"))
        except OSError as error:
            logger.warning("Could not write the query cache: %s", error)

    def _path(self, key: Mapping[str, object]) -> Path:
        """Return the file of ``key``: the hash of the key and the version."""
        text = json.dumps(
            {"cache_version": self.CACHE_VERSION, **key},
            sort_keys=True,
            ensure_ascii=False,
        )

        return (
            self._directory
            / f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}.json"
        )
