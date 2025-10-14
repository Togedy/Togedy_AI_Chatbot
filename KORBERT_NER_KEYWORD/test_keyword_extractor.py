# -*- coding: utf-8 -*-
"""
test_keyword_extractor.py
- 모델만 vs 모델+규칙(보강) 결과 비교
사용:
  python test_keyword_extractor.py
  python test_keyword_extractor.py --q "KU일반학생 전형 방법"
"""
import argparse

try:
    from KORBERT_NER_KEYWORD.keyword_extractor import KeywordExtractor
except Exception:
    try:
        from .keyword_extractor import KeywordExtractor  # type: ignore
    except Exception:
        from keyword_extractor import KeywordExtractor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", dest="queries", nargs="*", default=[
        "건대 정시 전형 일정 알려줘",
        "건국대 정시 모집인원",
        "KU일반학생 전형 방법",
        "영어등급 환산표 보여줘",
        "가군 모집단위 안내",
        "수능 최저 기준은?",
        "입결 궁금해",
        "컴퓨터공학부 모집인원 알려줘",
        "경제학부 전형 일정",
        "컴퓨터공학부와 경제학부 모두 모집일정 궁금해",
        "컴퓨터공학부 모집인원 알려줘",
        "경제학부 전형 일정",
        "컴퓨터공학부와 경제학부 모두 모집일정 궁금해",
        "모집일정과 전공명을 함께 말했어",
        "건대랑 서연고 수시 모집 일정 알려줘",
        "건국대랑 연세대랑 붙으면 누가 이겨?",
        "서성한에서 입결 누가 더 높아?",
        "연세대 출신 연예인 누구 있어?",
        "자율전공 입결이 어느 정도야?"
    ])
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    ke = KeywordExtractor(use_model=True)

    for q in args.queries:
        model_only = ke.extract_model_only(q, topn=args.topn)
        hybrid     = ke.extract(q, topn=args.topn, allow_composite=True)
        added      = [k for k in hybrid if k not in set(model_only)]

        print("-"*78)
        print(f"Q: {q}")
        print(f"[Model only]         {model_only}")
        print(f"[Model + rules]      {hybrid}")
        print(f"[Added by rules]     {added}")

if __name__ == "__main__":
    main()
