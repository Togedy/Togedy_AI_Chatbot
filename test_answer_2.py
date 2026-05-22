# test_answer_2.py
# -*- coding: utf-8 -*-

import json
import argparse
from app.main import server_main


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="test2.txt")
    args = parser.parse_args()

    cases = load_cases(args.file)

    for idx, case in enumerate(cases, 1):
        print("\n" + "#" * 100)
        print(f"# 테스트 케이스 {idx}")
        print("#" * 100)

        first_payload = {
            "question_1": case.get("question_1", ""),
            "question_2": "",
            "first": True,
            "NER_Keyword": {},
        }

        print("\n[1] first 요청")
        print(json.dumps(first_payload, ensure_ascii=False, indent=2))

        first_response = server_main(first_payload)

        print("\n[1] first 응답")
        print(json.dumps({
            "first": first_response["first"],
            "NER_Keyword": first_response["NER_Keyword"],
            "answer": first_response["answer"],
        }, ensure_ascii=False, indent=2))

        second_payload = {
            "question_1": case.get("question_1", ""),
            "question_2": case.get("question_2", ""),
            "first": False,
            "NER_Keyword": first_response["NER_Keyword"],
        }

        print("\n[2] second 요청")
        print(json.dumps(second_payload, ensure_ascii=False, indent=2))

        second_response = server_main(second_payload)

        print("\n[2] second 응답")
        print(json.dumps({
            "first": second_response["first"],
            "NER_Keyword": second_response["NER_Keyword"],
            "answer": second_response["answer"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()