# -*- coding: utf-8 -*-
"""
main_keyword.py
- NER 우선 + 규칙 보강으로 문장에서 키워드 추출
- PDF 사용 안 함
사용:
  python main_keyword.py --q "건대 정시 전형 일정 알려줘" --q "영어등급 환산표"
  python main_keyword.py --interactive
"""
import argparse

# 안전 임포트
try:
    from KORBERT_NER_KEYWORD.keyword_extractor import KeywordExtractor
except Exception:
    try:
        from .keyword_extractor import KeywordExtractor  # type: ignore
    except Exception:
        from keyword_extractor import KeywordExtractor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", dest="queries", nargs="*", default=[], help="질문 문장(여러 개 가능)")
    ap.add_argument("--topn", type=int, default=10, help="표시할 상위 키워드 수")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--no-composite", action="store_true", help="합성 금지(문장 원형만)")
    args = ap.parse_args()

    ke = KeywordExtractor(use_model=True)

    if args.interactive:
        print("[대화식] 문장을 입력하세요. (빈 줄 종료)")
        while True:
            try:
                s = input("> ").strip()
            except KeyboardInterrupt:
                break
            if not s: break
            print("->", ke.extract(s, topn=args.topn, allow_composite=not args.no_composite))
        return

    queries = args.queries or [
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
    ]
    for q in queries:
        print(f"\nQ: {q}")
        print("->", ke.extract(q, topn=args.topn, allow_composite=not args.no_composite))

if __name__ == "__main__":
    main()
