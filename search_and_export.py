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
# (UNI, TYPE) 페어 생성 보정
# ---------------------------

def _find_all_positions(text: str, token: str) -> List[int]:
    """text 내에서 token이 등장하는 시작 인덱스 목록을 반환"""
    if not text or not token:
        return []
    return [m.start() for m in re.finditer(re.escape(token), text)]


def build_pairs_smart(
    question: str,
    ner_uni: Any,
    ner_type: Any,
    ner_kw: List[str],
) -> List[Dict[str, Any]]:
    """
    NER 결과(uni/type)가 여러 개인 경우, 불필요한 카테시안 곱을 만들지 않기 위해
    "문장 내 근접도"로 (UNI, TYPE) 페어를 우선 매칭한다.

    우선순위
    1) uni 2개 이상 & type 2개 이상: 질문 원문에서 type 토큰을 가장 가까운 uni에 매칭
       예) "건국대 수시랑 연세대 정시" -> (건국대, 수시), (연세대, 정시)
    2) uni 2개 이상 & type 1개: 모든 uni에 동일 type 부여
       예) uni=[연세대, 고려대], type=[정시] -> (연세대, 정시), (고려대, 정시)
    3) uni 1개 & type 2개 이상: 해당 uni에 모든 type 부여
       예) uni=[서울대], type=[정시, 수시] -> (서울대, 정시), (서울대, 수시)

    반환이 빈 리스트면(근접도 매칭 실패/정보 부족) 호출부에서 gemini_sort 또는
    카테시안 곱으로 폴백하면 된다.
    """
    uni_list = ner_uni if isinstance(ner_uni, list) else ([ner_uni] if ner_uni else [])
    type_list = ner_type if isinstance(ner_type, list) else ([ner_type] if ner_type else [])

    if not uni_list or not type_list:
        return []

    # (2) uni 다수 + type 단일
    if len(uni_list) > 1 and len(type_list) == 1:
        t = type_list[0]
        return [{"UNI": u, "TYPE": t, "KEYWORD": ner_kw} for u in uni_list]

    # (3) uni 단일 + type 다수
    if len(uni_list) == 1 and len(type_list) > 1:
        u = uni_list[0]
        return [{"UNI": u, "TYPE": t, "KEYWORD": ner_kw} for t in type_list]

    # (1) uni 다수 + type 다수 -> 근접도 매칭
    if len(uni_list) > 1 and len(type_list) > 1:
        uni_pos: List[Tuple[str, int]] = []
        type_pos: List[Tuple[str, int]] = []

        for u in uni_list:
            for p in _find_all_positions(question, u):
                uni_pos.append((u, p))

        for t in type_list:
            for p in _find_all_positions(question, t):
                type_pos.append((t, p))

        if not uni_pos or not type_pos:
            return []

        pairs_set = set()
        for t, tp in type_pos:
            best_u, best_d = None, None
            for u, up in uni_pos:
                d = abs(tp - up)
                if best_d is None or d < best_d:
                    best_u, best_d = u, d
            if best_u is not None:
                pairs_set.add((best_u, t))

        return [{"UNI": u, "TYPE": t, "KEYWORD": ner_kw} for (u, t) in sorted(pairs_set)]

    # uni/type 모두 단일
    return [{"UNI": uni_list[0], "TYPE": type_list[0], "KEYWORD": ner_kw}]


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
    return [txt[i:i + fallback_chars].strip() for i in range(0, len(txt), fallback_chars)]


# ---------------------------
# 추가: 정규화 / 키워드 포함 보너스 / 키워드 주변 스니펫
# ---------------------------

_NORM_REMOVE = re.compile(r"[\s\.\,\(\)\[\]\{\}\-_/\\:;\"'`~!@#$%^&*+=|<>?·•ㆍ…]+")

def normalize_for_match(s: str) -> str:
    """띄어쓰기/특수문자/중점 등을 제거한 비교용 정규화 문자열"""
    if not s:
        return ""
    s = s.lower()
    s = _NORM_REMOVE.sub("", s)
    return s


def keyword_bonus_score(page_text: str, keywords: List[str]) -> float:
    """
    페이지에 키워드가 '정확히 포함'될수록 점수를 올려줌.
    - 표/목록 페이지에서 TF-IDF가 흔들려도 '키워드 포함'을 강하게 반영하기 위한 보정.
    """
    if not page_text or not keywords:
        return 0.0

    p_norm = normalize_for_match(page_text)
    bonus = 0.0

    # 키워드별로 포함/등장 횟수 반영 (과도한 가중 방지 위해 cap)
    for kw in keywords:
        kw = (kw or "").strip()
        if not kw:
            continue
        k_norm = normalize_for_match(kw)
        if not k_norm:
            continue

        if k_norm in p_norm:
            bonus += 0.030  # 포함 자체 보너스

            # 등장횟수 보너스(최대 3회까지만)
            cnt = p_norm.count(k_norm)
            cnt = min(cnt, 3)
            bonus += 0.008 * cnt

    # 너무 커지지 않게 상한
    return min(bonus, 0.08)


def extract_snippet_around_keywords(page_text: str, keywords: List[str], window: int = 600) -> str:
    """
    기존 pages[idx][:300] 대신,
    키워드가 실제로 등장하는 위치 주변을 스니펫으로 잘라서 반환.
    - 키워드가 없으면 앞부분 일부를 반환.
    """
    if not page_text:
        return ""

    # 원문에서의 위치를 찾기 위해 원문 기반 탐색도 같이 수행
    best_pos = None
    for kw in keywords:
        kw = (kw or "").strip()
        if not kw:
            continue

        # 원문에서 그대로 검색
        pos = page_text.find(kw)
        if pos != -1:
            if best_pos is None or pos < best_pos:
                best_pos = pos
            continue

        # 정규화 기반 매칭: 위치 정확도는 떨어지지만 "없음" 방지는 됨
        p_norm = normalize_for_match(page_text)
        k_norm = normalize_for_match(kw)
        if k_norm and k_norm in p_norm:
            # 정규화 매칭은 원문 위치로 역매핑이 어렵다 → 앞부분에서 조금 더 길게 주는 방식
            best_pos = 0 if best_pos is None else min(best_pos, 0)

    if best_pos is None:
        # 키워드가 정말로 안 잡히면 앞부분을 조금 더 길게
        snippet = page_text[:window]
    else:
        half = window // 2
        start = max(best_pos - half, 0)
        end = min(best_pos + half, len(page_text))
        snippet = page_text[start:end]

    snippet = snippet.replace("\n", " ").strip()
    return snippet


def score_pages(pages: List[str], keywords: List[str], k: int = 5) -> List[Tuple[int, float]]:
    """
    TF-IDF + 키워드 포함 보너스로 재랭킹
    - 표/목록 페이지에서 '키워드가 있는데도' Top에 안 뜨는 문제를 완화
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not pages:
        return []

    kw_clean = [kw.strip() for kw in keywords if kw and kw.strip()]
    if not kw_clean:
        return []

    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    X = vect.fit_transform(pages)

    query_joined = " ".join(kw_clean)
    s_join = cosine_similarity(vect.transform([query_joined]), X)[0]
    indiv = [cosine_similarity(vect.transform([kw]), X)[0] for kw in kw_clean]
    indiv_avg = (sum(indiv) / len(indiv)) if indiv else s_join

    base = 0.6 * s_join + 0.4 * indiv_avg

    # 키워드 포함 보너스 반영
    final_scores = []
    for i, b in enumerate(base.tolist()):
        bonus = keyword_bonus_score(pages[i], kw_clean)
        final_scores.append((i, float(b) + bonus))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores[:k]


# ---------------------------
# 폴더/경로 해석
# ---------------------------

def get_uni_slug(uni_name: str) -> str:
    if not uni_name:
        return ""
    canon = normalize_uni(uni_name)
    slug = uni_to_slug(canon)
    return slug or canon.replace(" ", "")


def get_type_folder(type_text: str) -> str:
    if not type_text:
        return ""
    canon = normalize_type(type_text)
    return type_to_slug(canon) or ""


def resolve_text_path(uni_slug: str, type_folder: str) -> str:
    if not (uni_slug and type_folder):
        return ""
    return os.path.join(THIS, "university", uni_slug, f"{type_folder}_text.txt")


# ---------------------------
# 단일 질의 처리 (페어별 Top3 탐색 포함)
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

    # 1) NER 추출
    ner_uni = uni_ex.extract_uni(text)
    ner_type = type_ex.extract_type(text)
    ner_kw = kw_ex.extract_keywords(text)

    # 2) 최종 분류 결정 (업데이트된 시그니처: UNI + TYPE + KEYWORD)
    decision = final_bucket(ner_uni, ner_type, ner_kw)

    rows: List[Dict[str, Any]] = []
    stats = {"pairs": 0, "docs_found": 0, "pages_scored": 0}

    # ▶ 문서탐색이 아니면 여기서 바로 반환 (generate_answers.py에서 GPT만 사용)
    if decision != "문서탐색":
        rows.append({
            "input_query": text,
            "ner_uni": "|".join(ner_uni) if ner_uni else "",
            "ner_type": "|".join(ner_type) if isinstance(ner_type, list) else (ner_type or ""),
            "ner_keywords": "|".join(ner_kw) if ner_kw else "",
            "decision": decision,
            "matched_uni": "",
            "matched_type": "",
            "matched_keywords": "",
            "doc_path": "",
            "page_index": -1,
            "score": 0.0,
            "snippet": f"(최종 분류: {decision} — 문서탐색 아님)",
        })
        ner_dump = {
            "uni": ner_uni,
            "type": ner_type,
            "keywords": ner_kw,
            "decision": decision,
        }
        return rows, stats, ner_dump

    # ---- decision == "문서탐색" 인 경우: 페어별로 실제 문서 검색

    # 3) (UNI, TYPE) 페어 생성
    # - 우선: 문장 내 근접도 기반 스마트 매칭
    # - 폴백: gemini_sort 결과
    # - 최종 폴백: 카테시안 곱(모든 조합)
    smart_pairs = build_pairs_smart(text, ner_uni, ner_type, ner_kw)
    if smart_pairs:
        pairs = smart_pairs
    else:
        pairs = gemini_sort(api_key, gemini_model, ner_uni, ner_type, ner_kw)

        uni_list = ner_uni if isinstance(ner_uni, list) else ([ner_uni] if ner_uni else [])
        type_list = ner_type if isinstance(ner_type, list) else ([ner_type] if ner_type else [])

        # gemini_sort가 일부 페어만 반환하거나 비어있을 때 대비
        if (not pairs) and uni_list and type_list:
            pairs = [{"UNI": u, "TYPE": t, "KEYWORD": ner_kw} for u in uni_list for t in type_list]

    stats["pairs"] = len(pairs)

    for p in pairs:
        u = p.get("UNI")
        t = p.get("TYPE", "")
        klist = p.get("KEYWORD", [])
        if not isinstance(klist, list):
            klist = [klist] if klist else []

        uni_slug = get_uni_slug(u) if u else ""
        type_folder = get_type_folder(t)
        doc_path = resolve_text_path(uni_slug, type_folder)

        # (1) 경로 없거나 파일이 없으면 에러 메시지 row 추가
        if not doc_path or not os.path.exists(doc_path):
            rows.append({
                "input_query": text,
                "ner_uni": "|".join(ner_uni) if ner_uni else "",
                "ner_type": "|".join(ner_type) if isinstance(ner_type, list) else (ner_type or ""),
                "ner_keywords": "|".join(ner_kw) if ner_kw else "",
                "decision": decision,
                "matched_uni": u or "",
                "matched_type": t or "",
                "matched_keywords": "|".join(klist),
                "doc_path": doc_path or "(경로 생성 실패)",
                "page_index": -1,
                "score": 0.0,
                "snippet": "(문서 없음 — university/<uni_slug>/<type>_text.txt 확인)",
            })
            continue

        # (2) 문서 존재 → 페이지 분할 후 스코어링
        stats["docs_found"] += 1
        with open(doc_path, "r", encoding="utf-8") as f:
            raw = f.read()
        pages = split_text_into_pages(raw)
        stats["pages_scored"] += len(pages)

        # 여기서 top_pages는 내부 후보 수. 너무 작으면 p13 같은 페이지가 밀릴 수 있어 여유를 둔다.
        internal_k = max(top_pages, 12)

        ranking = score_pages(pages, klist, k=internal_k)
        ranking = ranking[:3] if ranking else []

        # (3) 적합 페이지 없음
        if not ranking:
            rows.append({
                "input_query": text,
                "ner_uni": "|".join(ner_uni) if ner_uni else "",
                "ner_type": "|".join(ner_type) if isinstance(ner_type, list) else (ner_type or ""),
                "ner_keywords": "|".join(ner_kw) if ner_kw else "",
                "decision": decision,
                "matched_uni": u or "",
                "matched_type": t or "",
                "matched_keywords": "|".join(klist),
                "doc_path": doc_path,
                "page_index": -1,
                "score": 0.0,
                "snippet": "(적합 페이지 없음)",
            })
            continue

        # (4) 상위 페이지들 rows에 추가
        for (idx, sc) in ranking:
            rows.append({
                "input_query": text,
                "ner_uni": "|".join(ner_uni) if ner_uni else "",
                "ner_type": "|".join(ner_type) if isinstance(ner_type, list) else (ner_type or ""),
                "ner_keywords": "|".join(ner_kw) if ner_kw else "",
                "decision": decision,
                "matched_uni": u or "",
                "matched_type": t or "",
                "matched_keywords": "|".join(klist),
                "doc_path": doc_path,
                "page_index": idx + 1,  # 1-based index
                "score": round(float(sc), 6),
                # 기존 300자 고정 -> 키워드 주변 스니펫
                "snippet": extract_snippet_around_keywords(pages[idx], klist, window=700),
            })

    ner_dump = {
        "uni": ner_uni,
        "type": ner_type,
        "keywords": ner_kw,
        "decision": decision,
    }
    return rows, stats, ner_dump


# ---------------------------
# CSV & I/O
# ---------------------------

def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = [
        "input_query", "ner_uni", "ner_type", "ner_keywords",
        "decision", "matched_uni", "matched_type", "matched_keywords",
        "doc_path", "page_index", "score", "snippet",
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
    ap.add_argument("-i", "--input", default="test.txt")
    ap.add_argument("-o", "--output", default="hits.csv")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--pages", type=int, default=5)
    args = ap.parse_args()

    api_key, gemini_model = load_env()
    uni_ex = UniExtractor(max_len=128)
    type_ex = TypeExtractor()
    kw_ex = KeywordExtractorBridge(topn=args.topn)

    queries = read_questions(args.input)
    all_rows, per_durations = [], []
    total_start = time.perf_counter()

    for qi, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        rows, stats, ner = search_top_pages_for_query(
            q, uni_ex, type_ex, kw_ex, api_key, gemini_model, top_pages=args.pages
        )
        all_rows.extend(rows)
        dt = time.perf_counter() - t0
        per_durations.append(dt)

        print(f"\n[{qi:03d}] 입력 문장: {q}")
        print(
            f"     NER 추출 → UNI:{ner.get('uni')}  "
            f"TYPE:{ner.get('type')}  KEYWORD:{ner.get('keywords')}"
        )
        print(f"     최종 분류: {ner.get('decision')}")
        print(
            f"     매칭쌍: {stats['pairs']}개, "
            f"문서 발견: {stats['docs_found']}개, "
            f"스코어링 대상 페이지: {stats['pages_scored']}장"
        )

        if ner.get("decision") != "문서탐색":
            print("     (문서탐색이 아니므로 검색 스킵)")
        else:
            preview = [r for r in rows if r["page_index"] != -1][:12]
            for i, r in enumerate(preview, 1):
                print(
                    f"       - Top{i}: {os.path.basename(r['doc_path'])} | "
                    f"p.{r['page_index']} | score={r['score']:.4f} | kw={r['matched_keywords']}"
                )

        print(f"     처리 시간: {fmt_sec(dt)}")

    # CSV 저장
    csv_t0 = time.perf_counter()
    write_csv(args.output, all_rows)
    csv_dt = time.perf_counter() - csv_t0

    total_dt = time.perf_counter() - total_start
    n = len(per_durations)
    avg_dt = sum(per_durations) / n if n else 0.0
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
