# search_and_export.py
# -*- coding: utf-8 -*-
import os, re, sys, csv, time, argparse
from typing import List, Tuple, Dict, Any

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

# === extract_all.py의 파이프라인 구성요소 재사용 ===
from extract_all import (
    UniExtractor, TypeExtractor, KeywordExtractorBridge,
    load_env, gemini_sort, final_bucket
)

# === mapping_loader의 표준화/슬러그 사용 (별칭 없이 공식 한글명 기준) ===
from utils.mapping_loader import (
    normalize_uni, normalize_type, uni_to_slug, type_to_slug
)

# ---------------------------
# 텍스트 페이지 분할 & 스코어링
# ---------------------------
PAGE_SEPS = [
    r"\f",
    r"^={3,}\s*page\s*\d+\s*=*$",
    r"^-{3,}\s*page\s*\d+\s*-*$",
    r"^\s*\[?page\s*\d+\]?\s*$",
]
SEP_REGEX = re.compile("|".join(f"(?:{p})" for p in PAGE_SEPS),
                       re.IGNORECASE | re.MULTILINE)

def split_text_into_pages(raw: str, fallback_chars: int = 1200) -> List[str]:
    pages = [p.strip() for p in SEP_REGEX.split(raw) if p.strip()]
    if len(pages) >= 2:
        return pages
    txt = raw.strip()
    if not txt:
        return []
    return [txt[i:i+fallback_chars].strip() for i in range(0, len(txt), fallback_chars)]

def score_pages(pages: List[str], keywords: List[str], k: int = 5) -> List[Tuple[int, float]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not pages:
        return []
    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    X = vect.fit_transform(pages)

    kw_clean = [kw.strip() for kw in keywords if kw and kw.strip()]
    if not kw_clean:
        return []

    # 결합 쿼리(루스 OR) + 개별 키워드 평균 혼합
    query_joined = " ".join(kw_clean)
    s_join = cosine_similarity(vect.transform([query_joined]), X)[0]
    indiv = [cosine_similarity(vect.transform([kw]), X)[0] for kw in kw_clean]
    indiv_avg = (sum(indiv) / len(indiv)) if indiv else s_join

    final = 0.6 * s_join + 0.4 * indiv_avg
    ranked = list(enumerate(final.tolist()))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:k]

# ---------------------------
# 폴더/경로 해석 (별칭 미사용)
# ---------------------------
def get_uni_slug(uni_name: str) -> str:
    """
    - 공식 한글 대학명을 normalize_uni로 정규화 후 uni_to_slug로 변환.
    - 매핑 실패 시 공백 제거 원문을 폴백(프로젝트에 한글 폴더가 있는 경우 대비).
    """
    if not uni_name:
        return ""
    canon = normalize_uni(uni_name)  # 없으면 입력 그대로 반환
    slug = uni_to_slug(canon)        # 없으면 ""
    return slug or canon.replace(" ", "")

def get_type_folder(type_text: str) -> str:
    """
    - normalize_type으로 '정시'/'수시' 표준화 후 type_to_slug('jungsi'/'susi').
    """
    if not type_text:
        return ""
    canon = normalize_type(type_text)  # '', '정시', '수시'
    return type_to_slug(canon) or ""

def resolve_text_path(uni_slug: str, type_folder: str) -> str:
    if not (uni_slug and type_folder):
        return ""
    return os.path.join(THIS, "university", uni_slug, f"{type_folder}_text.txt")

# ---------------------------
# 단일 질의 처리
# ---------------------------
def search_top_pages_for_query(
    text: str,
    uni_ex: UniExtractor,
    type_ex: TypeExtractor,
    kw_ex: KeywordExtractorBridge,
    api_key: str,
    gemini_model: str,
    top_pages: int = 5
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    반환:
      rows: CSV 행들
      stats: {'pairs': int, 'docs_found': int, 'pages_scored': int}
      ner_dump: {'uni': list, 'type': list|str, 'keywords': list, 'decision': str}
    """
    # 1) NER 추출 (extract_all.py와 동일한 컴포넌트)
    ner_uni = uni_ex.extract_uni(text)
    ner_type = type_ex.extract_type(text)
    ner_kw   = kw_ex.extract_keywords(text)

    # 2) LLM 재정렬 (대학/전형/키워드 매칭쌍 생성)
    pairs = gemini_sort(api_key, gemini_model, ner_uni, ner_type, ner_kw)

    # 3) 최종 분류 (문서탐색/재질문/답변 생성 등)
    decision = final_bucket(ner_uni, ner_kw)

    stats = {"pairs": len(pairs), "docs_found": 0, "pages_scored": 0}
    rows: List[Dict[str, Any]] = []

    # 3-1) 문서탐색이 아니면 검색 스킵
    if decision != "문서탐색":
        rows.append({
            "input_query": text,
            "ner_uni": "|".join(ner_uni) if ner_uni else "",
            "ner_type": "|".join(ner_type) if isinstance(ner_type, list) else (ner_type or ""),
            "ner_keywords": "|".join(ner_kw) if ner_kw else "",
            "decision": decision,
            "matched_uni": "", "matched_type": "", "matched_keywords": "",
            "doc_path": "", "page_index": -1, "score": 0.0,
            "snippet": f"(최종 분류: {decision} — 문서탐색 아님)"
        })
        ner_dump = {"uni": ner_uni, "type": ner_type, "keywords": ner_kw, "decision": decision}
        return rows, stats, ner_dump

    # 4) 문서탐색인 경우에만 검색 수행
    for p in pairs:
        u = p.get("UNI")
        t = p.get("TYPE", "")
        klist = p.get("KEYWORD", [])
        if not isinstance(klist, list):
            klist = [klist] if klist else []

        uni_slug = get_uni_slug(u) if u else ""
        type_folder = get_type_folder(t)
        doc_path = resolve_text_path(uni_slug, type_folder)

        if not doc_path or not os.path.exists(doc_path):
            rows.append({
                "input_query": text,
                "ner_uni": "|".join(ner_uni) if ner_uni else "",
                "ner_type": "|".join(ner_type) if isinstance(ner_type, list) else (ner_type or ""),
                "ner_keywords": "|".join(ner_kw) if ner_kw else "",
                "decision": decision,
                "matched_uni": u or "", "matched_type": t or "", "matched_keywords": "|".join(klist),
                "doc_path": doc_path or "(경로 생성 실패)", "page_index": -1, "score": 0.0,
                "snippet": "(문서 없음 — university/<uni_slug>/<type>_text.txt 확인)"
            })
            continue

        stats["docs_found"] += 1
        with open(doc_path, "r", encoding="utf-8") as f:
            raw = f.read()
        pages = split_text_into_pages(raw)
        stats["pages_scored"] += len(pages)
        ranking = score_pages(pages, klist, k=top_pages)

        if not ranking:
            rows.append({
                "input_query": text,
                "ner_uni": "|".join(ner_uni) if ner_uni else "",
                "ner_type": "|".join(ner_type) if isinstance(ner_type, list) else (ner_type or ""),
                "ner_keywords": "|".join(ner_kw) if ner_kw else "",
                "decision": decision,
                "matched_uni": u or "", "matched_type": t or "", "matched_keywords": "|".join(klist),
                "doc_path": doc_path, "page_index": -1, "score": 0.0,
                "snippet": "(적합 페이지 없음)"
            })
            continue

        for (idx, sc) in ranking:
            rows.append({
                "input_query": text,
                "ner_uni": "|".join(ner_uni) if ner_uni else "",
                "ner_type": "|".join(ner_type) if isinstance(ner_type, list) else (ner_type or ""),
                "ner_keywords": "|".join(ner_kw) if ner_kw else "",
                "decision": decision,
                "matched_uni": u or "", "matched_type": t or "", "matched_keywords": "|".join(klist),
                "doc_path": doc_path, "page_index": idx + 1, "score": round(float(sc), 6),
                "snippet": pages[idx][:300].replace("\n", " ").strip()
            })

    ner_dump = {"uni": ner_uni, "type": ner_type, "keywords": ner_kw, "decision": decision}
    return rows, stats, ner_dump

# ---------------------------
# CSV & I/O
# ---------------------------
def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = [
        "input_query",
        "ner_uni", "ner_type", "ner_keywords",
        "decision",
        "matched_uni", "matched_type", "matched_keywords",
        "doc_path", "page_index", "score", "snippet"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

def read_questions(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"입력 파일 없음: {path}")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            out.append(line)
    return out

def fmt_sec(s: float) -> str:
    return f"{s*1000:.1f} ms" if s < 1.0 else f"{s:.3f} s"

# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input", default="test.txt", help="질문 목록 파일 경로")
    ap.add_argument("-o","--output", default="hits.csv", help="CSV 저장 경로")
    ap.add_argument("--topn", type=int, default=10, help="키워드 추출 상위 N")
    ap.add_argument("--pages", type=int, default=5, help="문서당 상위 페이지 개수")
    args = ap.parse_args()

    api_key, gemini_model = load_env()
    uni_ex = UniExtractor(max_len=128)
    type_ex = TypeExtractor()
    kw_ex = KeywordExtractorBridge(topn=args.topn)

    queries = read_questions(args.input)
    all_rows: List[Dict[str, Any]] = []
    per_durations: List[float] = []
    total_start = time.perf_counter()

    for qi, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        rows, stats, ner = search_top_pages_for_query(
            q, uni_ex, type_ex, kw_ex, api_key, gemini_model, top_pages=args.pages
        )
        all_rows.extend(rows)
        dt = time.perf_counter() - t0
        per_durations.append(dt)

        # ---- 콘솔 출력: (입력) → (NER) → (최종분류) → (요약) → (Top 미리보기)
        print(f"\n[{qi:03d}] 입력 문장: {q}")
        print(f"     NER 추출 → UNI:{ner.get('uni') or []}  TYPE:{ner.get('type') or []}  KEYWORD:{ner.get('keywords') or []}")
        print(f"     최종 분류: {ner.get('decision')}")
        print(f"     매칭쌍: {stats['pairs']}개, 문서 발견: {stats['docs_found']}개, 스코어링 대상 페이지: {stats['pages_scored']}장")

        if ner.get("decision") != "문서탐색":
            print("     (문서탐색이 아니므로 검색 스킵)")
        else:
            preview = [r for r in rows if r["page_index"] != -1][:3]
            for i, r in enumerate(preview, 1):
                print(f"       - Top{i}: {os.path.basename(r['doc_path'])} | p.{r['page_index']} | score={r['score']:.4f} | kw={r['matched_keywords']}")

        print(f"     처리 시간: {fmt_sec(dt)}")

    # CSV 저장
    csv_t0 = time.perf_counter()
    write_csv(args.output, all_rows)
    csv_dt = time.perf_counter() - csv_t0

    total_dt = time.perf_counter() - total_start
    n = len(per_durations)
    avg_dt = sum(per_durations)/n if n else 0.0
    mn = min(per_durations) if per_durations else 0.0
    mx = max(per_durations) if per_durations else 0.0

    print("\n=== 처리 시간 요약 ===")
    print(f"총 질의 수: {n}")
    print(f"총 소요 시간: {fmt_sec(total_dt)} (CSV 저장: {fmt_sec(csv_dt)})")
    print(f"평균/최솟값/최댓값: {fmt_sec(avg_dt)} / {fmt_sec(mn)} / {fmt_sec(mx)}")
    print(f"CSV 저장 위치: {os.path.abspath(args.output)}")
    print(f"총 결과 행 수: {len(all_rows)}")

if __name__ == "__main__":
    main()
