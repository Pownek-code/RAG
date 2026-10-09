"""Retrieval quality measured with recall@k."""

from src.models import (
    AnsweredDataset,
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


class RetrievalEvaluator:
    """Computes recall@k the way the subject defines it.

    A reference source counts as found when a retrieved source is in the
    same file and overlaps it with an IoU of at least ``MINIMUM_IOU``.
    """

    MINIMUM_IOU = 0.05

    def evaluate(
        self,
        dataset: AnsweredDataset,
        student_results: StudentSearchResults,
    ) -> float:
        """Average recall over all questions of a dataset.

        Questions without search results count as a recall of zero.

        Args:
            dataset: Questions with their reference sources.
            student_results: Retrieved sources; only the first ``k`` count.

        Returns:
            The mean recall between 0 and 1.
        """
        results_by_question_id = {
            result.question_id: result
            for result in student_results.search_results
        }

        total_recall = 0.0

        for question in dataset.rag_questions:
            search_result = results_by_question_id.get(
                question.question_id
            )

            question_recall = (
                self._calculate_question_recall(
                    question=question,
                    search_result=search_result,
                    k=student_results.k,
                )
            )

            total_recall += question_recall

        return total_recall / len(
            dataset.rag_questions
        )

    def _calculate_question_recall(
        self,
        question: AnsweredQuestion,
        search_result: MinimalSearchResults | None,
        k: int,
    ) -> float:
        """Return the share of reference sources found in the top k."""
        if search_result is None:
            return 0.0

        retrieved_sources = (
            search_result.retrieved_sources[:k]
        )
        found_sources = 0

        for expected_source in question.sources:
            for retrieved_source in retrieved_sources:
                if self._sources_match(
                    expected_source=expected_source,
                    retrieved_source=retrieved_source,
                ):
                    found_sources += 1
                    break

        return found_sources / len(
            question.sources
        )

    def _sources_match(
        self,
        expected_source: MinimalSource,
        retrieved_source: MinimalSource,
    ) -> bool:
        """Tell whether two sources are in the same file and overlap enough."""
        if (
            expected_source.file_path
            != retrieved_source.file_path
        ):
            return False

        iou = self.calculate_iou(
            expected_source=expected_source,
            retrieved_source=retrieved_source,
        )

        return iou >= self.MINIMUM_IOU

    @staticmethod
    def calculate_iou(
        expected_source: MinimalSource,
        retrieved_source: MinimalSource,
    ) -> float:
        """Intersection over union of two character ranges.

        Args:
            expected_source: The reference source.
            retrieved_source: The retrieved source.

        Returns:
            A value between 0 and 1; 0 when the ranges do not overlap.
            The file paths are not compared here.
        """
        expected_length = (
            expected_source.last_character_index
            - expected_source.first_character_index
        )
        retrieved_length = (
            retrieved_source.last_character_index
            - retrieved_source.first_character_index
        )

        intersection_start = max(
            expected_source.first_character_index,
            retrieved_source.first_character_index,
        )
        intersection_end = min(
            expected_source.last_character_index,
            retrieved_source.last_character_index,
        )

        intersection_length = max(
            0,
            intersection_end - intersection_start,
        )

        if intersection_length == 0:
            return 0.0

        union_length = (
            expected_length
            + retrieved_length
            - intersection_length
        )

        return intersection_length / union_length
