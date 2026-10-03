from src.models import (
    AnsweredDataset,
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


class RetrievalEvaluator:
    MINIMUM_IOU = 0.05

    def evaluate(
        self,
        dataset: AnsweredDataset,
        student_results: StudentSearchResults,
    ) -> float:
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