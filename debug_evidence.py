# debug_evidence.py
# 사용법:
#   python debug_evidence.py --jungsi ./university/seoul/jungsi_text.txt --answer "정시 모집: 99명" --keyword "컴퓨터공학부" --pages 13 74 93
#
# 의도:
# 1) 답변(LLM 출력)에서 숫자(예: 99)를 뽑고
# 2) 컨텍스트(페이지 텍스트) 안에 그 숫자가 실제 존재하는지
# 3) 키워드(컴퓨터공학부) 주변 window 안에 존재하는지
# 4) 키워드가 있는 라인(또는 근처 라인)을 그대로 출력해서 "근거"를 확인

import argparse
import re
from typing import List, Tuple

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_pages(raw: str) -> List[str]:
    # 너희 프로젝트에서 쓰는 split marker가 "==== Page N ====" 형태라 가정
    # (현재 jungsi_text.txt가 그 형태로 되어 있음)
    parts = re.split(r"==== Page\s+\d+\s+====", raw)
    # split 결과: [프리앰블, page1, page2, ...] 구조가 될 수 있음
    # 프리앰블 제거
    if len(parts) > 1:
        return parts[1:]
    # 혹시 marker가 없으면 전체를 1페이지로
    return [raw]

def extract_numbers_from_answer(answer: str) -> List[str]:
    # 1,194 같은 콤마 숫자도 포함
    nums = re.findall(r"\b\d{1,3}(?:,\d{3})*\b", answer)
    return nums

def normalize_num(s: str) -> str:
    return s.replace(",", "")

def find_keyword_context(page_text: str, keyword: str, window: int = 400) -> List[str]:
    idxs = [m.start() for m in re.finditer(re.escape(keyword), page_text)]
    contexts = []
    for i in idxs:
        start = max(0, i - window)
        end = min(len(page_text), i + window)
        contexts.append(page_text[start:end])
    return contexts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jungsi", required=True, help="jungsi_text.txt 경로")
    ap.add_argument("--answer", required=True, help="LLM이 생성한 답변 문자열")
    ap.add_argument("--keyword", default="컴퓨터공학부", help="검증할 키워드")
    ap.add_argument("--pages", nargs="*", type=int, default=[], help="검사할 split page 번호들(1-based)")
    ap.add_argument("--window", type=int, default=500, help="키워드 주변 컨텍스트 window")
    args = ap.parse_args()

    raw = load_text(args.jungsi)
    pages = split_pages(raw)

    nums = extract_numbers_from_answer(args.answer)
    nums_norm = [normalize_num(n) for n in nums]

    print("[INFO] extracted numbers from answer:", nums)
    print("[INFO] normalized:", nums_norm)
    print("[INFO] total split pages:", len(pages))
    print()

    # 검사 페이지 선택: 지정 없으면 전체
    check_pages = args.pages if args.pages else list(range(1, len(pages) + 1))

    # 숫자 존재 여부, 키워드 주변 존재 여부를 각각 체크
    for pno in check_pages:
        if pno < 1 or pno > len(pages):
            continue
        text = pages[pno - 1]

        # 1) 페이지에 키워드가 있는지
        has_kw = args.keyword in text

        # 2) 페이지 전체에 답변 숫자가 있는지
        hits_any = []
        for n_raw, n_norm in zip(nums, nums_norm):
            if n_raw in text or n_norm in text:
                hits_any.append(n_raw)

        # 3) 키워드 주변 window 안에 답변 숫자가 있는지
        nearby_hits = []
        if has_kw:
            contexts = find_keyword_context(text, args.keyword, window=args.window)
            for ctx in contexts:
                for n_raw, n_norm in zip(nums, nums_norm):
                    if n_raw in ctx or n_norm in ctx:
                        nearby_hits.append(n_raw)

        if has_kw or hits_any:
            print("=" * 100)
            print(f"[PAGE {pno}] has_keyword={has_kw} page_hits={hits_any} nearby_hits={sorted(set(nearby_hits))}")
            if has_kw:
                contexts = find_keyword_context(text, args.keyword, window=args.window)
                for i, ctx in enumerate(contexts, 1):
                    print("-" * 100)
                    print(f"[CONTEXT #{i}] around keyword='{args.keyword}' (window={args.window})")
                    print(ctx.replace("\n", "\\n"))
            print("=" * 100)
            print()

if __name__ == "__main__":
    main()
