# test_cases/flow_simulations.py

import json
from app.routes import run_pipeline  # run_pipeline 불러오기

# 테스트 JSON 로드 함수
def load_test_cases(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# 단일 테스트 실행
def run_test_case(question: str, first: bool):
    print(f"\n[질문] ({'첫 질문' if first else '후속 질문'})")
    print(f"> {question}\n")

    result = run_pipeline(question, first)
    print("[응답 요약]")
    for i, ans in enumerate(result["Answers"]):
        print(f"  - {i+1}) {ans['query']['UNI']} {ans['query']['TYPE']} {ans['query']['KEYWORD']}")
        print(f"     => {ans['answer']}\n")


# 전체 테스트 실행
def run_all_tests():
    first_cases = load_test_cases('./test_cases/first_question_tests.json')
    followup_cases = load_test_cases('./test_cases/followup_question_tests.json')

    print("======== 첫 질문 테스트 ========")
    for case in first_cases:
        run_test_case(case["question"], first=True)

    print("======== 후속 질문 테스트 ========")
    for case in followup_cases:
        run_test_case(case["question"], first=False)


if __name__ == "__main__":
    run_all_tests()
