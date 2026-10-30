import unittest

from llm.counselor import present_answer
from question_plan import build_question_plan


class CounselorTests(unittest.TestCase):
    def test_verified_facts_and_exceptions_are_kept_verbatim(self):
        plan = build_question_plan("연세대 정시 제출서류")
        facts = "국내고 동의자: 별도 제출 없음. 미동의자: 사본 1부."
        answer = present_answer(facts, plan, "문서탐색", True)
        self.assertIn(facts, answer)
        self.assertIn("본인의 출신 학교", answer)
        self.assertNotIn("?", answer)

    def test_counseling_asks_for_relevant_context_without_prediction(self):
        plan = build_question_plan("건국대 학생부종합 물리 안 들었는데 불이익이 클까요?")
        answer = present_answer(plan.advice, plan, "문서탐색", True)
        self.assertIn("판단할 수 없습니다", answer)
        self.assertIn("어떤 과목을 이수", answer)
        self.assertEqual(answer.count("?"), 1)

    def test_clarification_stays_last_and_unchanged(self):
        plan = build_question_plan("어느 대학이 더 좋아?")
        text = "어느 모집단위를 말씀하신 건가요?"
        self.assertEqual(present_answer(text, plan, "재질문", True), text)

    def test_missing_evidence_does_not_sound_verified(self):
        plan = build_question_plan("건국대 수시 모집인원")
        text = "해당 내용을 확인하지 못했습니다."
        self.assertEqual(present_answer(text, plan, "문서탐색"), text)

    def test_external_results_request_keeps_limit_and_asks_for_scores(self):
        plan = build_question_plan("건대 경영 정시 백분위 몇 정도?")
        text = "공식 입시결과가 필요합니다."
        answer = present_answer(text, plan, "답변 생성")
        self.assertIn(text, answer)
        self.assertIn("현재 성적", answer)


if __name__ == "__main__":
    unittest.main()
