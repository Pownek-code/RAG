import fire


class CLI:
    def status(self) -> str:
        return "RAG project foundation is ready"


if __name__ == "__main__":
    fire.Fire(CLI)
