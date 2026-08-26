# search_and_export.py
# -*- coding: utf-8 -*-
import os, re, sys, csv, time, argparse
from typing import List, Tuple, Dict, Any, Optional

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

# === extract_all.py의 파이프라인 구성요소 재사용 ===
from extract_all import (
    UniExtractor, TypeExtractor, KeywordExtractorBridge,
    load_env, gemini_sort, final_bucket
)

# === mapping_loader의 표준화/슬러그 사용 ===
from utils.mapping_loader import (
    normalize_uni, normalize_type, uni_to_slug, type_to_slug
)
from utils.major_aliases import expand_major_keywords, discover_document_major_aliases

# ---------------------------
# (UNI, TYPE) 페어 생성 보정
# ---------------------------

def _find_all_positions(text: str, token: str) -> List[int]:
    if not text or not token:
        return []
    return [m.start() for m in re.finditer(re.escape(token), text)]


def build_pairs_smart(
    question: str,
    ner_uni: Any,
    ner_type: Any,
    ner_kw: List[str],
) -> List[Dict[str, Any]]:
    uni_list = ner_uni if isinstance(ner_uni, list) else ([ner_uni] if ner_uni else [])
    type_list = ner_type if isinstance(ner_type, list) else ([ner_type] if ner_type else [])

    if not uni_list or not type_list:
        return []

    # uni 다수 + type 단일
    if len(uni_list) > 1 and len(type_list) == 1:
        t = type_list[0]
        return [{"UNI": u, "TYPE": t, "KEYWORD": ner_kw} for u in uni_list]

    # uni 단일 + type 다수
    if len(uni_list) == 1 and len(type_list) > 1:
        u = uni_list[0]
        return [{"UNI": u, "TYPE": t, "KEYWORD": ner_kw} for t in type_list]

    # uni 다수 + type 다수 -> 근접도 매칭
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

    return [{"UNI": uni_list[0], "TYPE": type_list[0], "KEYWORD": ner_kw}]


# ---------------------------
# 라벨 기반 페이지 분할 (==== Page N ====)
# ---------------------------

PAGE_LABEL_RE = re.compile(r"^\s*={2,}\s*Page\s*(\d+)\s*={2,}\s*$", re.IGNORECASE | re.MULTILINE)

def split_text_into_labeled_pages(raw: str, fallback_chars: int = 1200) -> List[Dict[str, Any]]:
    """
    파일 내 '==== Page N ====' 라벨을 기준으로 페이지 분할.
    반환: [{"label": N(int) 또는 None, "text": chunk_text(str)}...]
    - label이 없는 구간이 존재할 수 있어 None 처리
    - 라벨이 아예 없으면 fallback_chars로 등분하고 label=None
    """
    raw = raw or ""
    matches = list(PAGE_LABEL_RE.finditer(raw))

    # 라벨이 없으면 기존처럼 문자수 기준으로 분할
    if not matches:
        txt = raw.strip()
        if not txt:
            return []
        chunks = [txt[i:i + fallback_chars].strip() for i in range(0, len(txt), fallback_chars)]
        return [{"label": None, "text": c} for c in chunks if c]

    pages: List[Dict[str, Any]] = []
    for i, m in enumerate(matches):
        label = None
        try:
            label = int(m.group(1))
        except Exception:
            label = None

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        chunk = raw[start:end].strip()

        # 라벨 아래 내용이 비어있는 경우도 있을 수 있으니 방어
        if chunk:
            pages.append({"label": label, "text": chunk})
        else:
            pages.append({"label": label, "text": ""})

    # 빈 text는 제거(인덱스 혼선 최소화)
    pages = [p for p in pages if (p.get("text") or "").strip()]
    return pages


# ---------------------------
# 추가: 정규화 / 키워드 포함 보너스 / 키워드 주변 스니펫
# ---------------------------

_NORM_REMOVE = re.compile(r"[\s\.\,\(\)\[\]\{\}\-_/\\:;\"'`~!@#$%^&*+=|<>?·•ㆍ…]+")

def normalize_for_match(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _NORM_REMOVE.sub("", s)
    return s


def keyword_bonus_score(page_text: str, keywords: List[str]) -> float:
    if not page_text or not keywords:
        return 0.0

    p_norm = normalize_for_match(page_text)
    bonus = 0.0

    for kw in keywords:
        kw = (kw or "").strip()
        if not kw:
            continue
        k_norm = normalize_for_match(kw)
        if not k_norm:
            continue

        if k_norm in p_norm:
            bonus += 0.030
            cnt = p_norm.count(k_norm)
            cnt = min(cnt, 3)
            bonus += 0.008 * cnt

    return min(bonus, 0.08)


def extract_snippet_around_keywords(page_text: str, keywords: List[str], window: int = 700) -> str:
    if not page_text:
        return ""

    best_pos = None
    for kw in keywords:
        kw = (kw or "").strip()
        if not kw:
            continue

        pos = page_text.find(kw)
        if pos != -1:
            if best_pos is None or pos < best_pos:
                best_pos = pos
            continue

        p_norm = normalize_for_match(page_text)
        k_norm = normalize_for_match(kw)
        if k_norm and k_norm in p_norm:
            best_pos = 0 if best_pos is None else min(best_pos, 0)

    if best_pos is None:
        snippet = page_text[:window]
    else:
        half = window // 2
        start = max(best_pos - half, 0)
        end = min(best_pos + half, len(page_text))
        snippet = page_text[start:end]

    return snippet.replace("\n", " ").strip()


def score_pages(pages_text: List[str], keywords: List[str], k: int = 5) -> List[Tuple[int, float]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not pages_text:
        return []

    kw_clean = [kw.strip() for kw in keywords if kw and kw.strip()]
    if not kw_clean:
        return []

    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    X = vect.fit_transform(pages_text)

    query_joined = " ".join(kw_clean)
    s_join = cosine_similarity(vect.transform([query_joined]), X)[0]
    indiv = [cosine_similarity(vect.transform([kw]), X)[0] for kw in kw_clean]
    indiv_avg = (sum(indiv) / len(indiv)) if indiv else s_join

    base = 0.6 * s_join + 0.4 * indiv_avg

    final_scores = []
    for i, b in enumerate(base.tolist()):
        bonus = keyword_bonus_score(pages_text[i], kw_clean)
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

    ner_uni = uni_ex.extract_uni(text)
    ner_type = type_ex.extract_type(text)
    ner_kw = kw_ex.extract_keywords(text)

    decision = final_bucket(ner_uni, ner_type, ner_kw)

    rows: List[Dict[str, Any]] = []
    stats = {"pairs": 0, "docs_found": 0, "pages_scored": 0}

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
            "split_index": "",
        })
        ner_dump = {"uni": ner_uni, "type": ner_type, "keywords": ner_kw, "decision": decision}
        return rows, stats, ner_dump

    # 페어 생성
    smart_pairs = build_pairs_smart(text, ner_uni, ner_type, ner_kw)
    if smart_pairs:
        pairs = smart_pairs
    else:
        pairs = gemini_sort(api_key, gemini_model, ner_uni, ner_type, ner_kw)

        uni_list = ner_uni if isinstance(ner_uni, list) else ([ner_uni] if ner_uni else [])
        type_list = ner_type if isinstance(ner_type, list) else ([ner_type] if ner_type else [])

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
        original_klist = list(klist)
        klist, alias_notes = expand_major_keywords(uni_slug, klist, question=text)
        alias_terms = [term for term in klist if term not in original_klist]
        type_folder = get_type_folder(t)
        doc_path = resolve_text_path(uni_slug, type_folder)

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
                "major_alias_notes": "|".join(alias_notes),
                "major_alias_terms": "|".join(alias_terms),
                "doc_path": doc_path or "(경로 생성 실패)",
                "page_index": -1,
                "score": 0.0,
                "snippet": "(문서 없음 — university/<uni_slug>/<type>_text.txt 확인)",
                "split_index": "",
            })
            continue

        stats["docs_found"] += 1
        with open(doc_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Curated mapping이 없는 대학도 해당 대학의 실제 모집요강에 존재하는
        # 관련 모집단위 명칭을 찾아 검색어를 확장한다.
        if not alias_terms:
            discovered_klist, discovered_notes = discover_document_major_aliases(
                klist,
                question=text,
                document_text=raw,
            )
            discovered_terms = [term for term in discovered_klist if term not in klist]
            if discovered_terms:
                klist = discovered_klist
                alias_terms.extend(discovered_terms)
                alias_notes.extend(note for note in discovered_notes if note not in alias_notes)

        labeled_pages = split_text_into_labeled_pages(raw)

        # 스코어링 대상 페이지 수는 "라벨 페이지 개수" 기준으로 집계
        stats["pages_scored"] += len(labeled_pages)

        pages_text = [pp["text"] for pp in labeled_pages]
        labels = [pp.get("label") for pp in labeled_pages]  # label may be None

        internal_k = max(top_pages, 12)
        ranking = score_pages(pages_text, klist, k=internal_k)

        # 대학별 실제 모집단위 별칭이 적용된 모집인원 질문은 별칭이 직접
        # 등장하고 숫자 근거가 풍부한 페이지를 우선한다. 일반 질문의
        # 기존 TF-IDF 순서는 변경하지 않는다.
        compact_query = re.sub(r"\s+", "", text or "")
        is_quota_query = any(term in compact_query for term in ("모집인원", "모집인원은", "몇명", "몇 명", "뽑아", "선발인원"))
        if alias_terms and is_quota_query:
            def alias_page_priority(item):
                idx, score = item
                page = pages_text[idx]
                alias_hit = any(term in page for term in alias_terms)
                number_count = len(re.findall(r"\d+", page))
                return (1 if alias_hit else 0, min(number_count, 100), float(score))

            ranking = sorted(ranking, key=alias_page_priority, reverse=True)
        ranking = ranking[:3] if ranking else []

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
                "major_alias_notes": "|".join(alias_notes),
                "major_alias_terms": "|".join(alias_terms),
                "major_alias_priority": 0,
                "doc_path": doc_path,
                "page_index": -1,
                "score": 0.0,
                "snippet": "(적합 페이지 없음)",
                "split_index": "",
            })
            continue

        for (idx, sc) in ranking:
            label = labels[idx]
            # label이 None이면 fallback: split 인덱스(1-based)를 사용
            page_out = int(label) if isinstance(label, int) else (idx + 1)

            rows.append({
                "input_query": text,
                "ner_uni": "|".join(ner_uni) if ner_uni else "",
                "ner_type": "|".join(ner_type) if isinstance(ner_type, list) else (ner_type or ""),
                "ner_keywords": "|".join(ner_kw) if ner_kw else "",
                "decision": decision,
                "matched_uni": u or "",
                "matched_type": t or "",
                "matched_keywords": "|".join(klist),
                "major_alias_notes": "|".join(alias_notes),
                "major_alias_terms": "|".join(alias_terms),
                "major_alias_priority": 1 if any(term in pages_text[idx] for term in alias_terms) else 0,
                "doc_path": doc_path,
                # 핵심: 이제 page_index는 "==== Page N ===="의 N(라벨)
                "page_index": page_out,
                "score": round(float(sc), 6),
                "snippet": extract_snippet_around_keywords(
                    pages_text[idx], alias_terms or klist, window=700
                ),
                # 디버그용 split 인덱스도 함께 보관(필요하면 generate_answers에서 출력 가능)
                "split_index": idx + 1,
            })

    ner_dump = {"uni": ner_uni, "type": ner_type, "keywords": ner_kw, "decision": decision}
    return rows, stats, ner_dump


# ---------------------------
# CSV & I/O
# ---------------------------

def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = [
        "input_query", "ner_uni", "ner_type", "ner_keywords",
        "decision", "matched_uni", "matched_type", "matched_keywords",
        "major_alias_notes", "major_alias_terms", "major_alias_priority",
        "doc_path", "page_index", "score", "snippet",
        "split_index",
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
            f"스코어링 대상 페이지(라벨): {stats['pages_scored']}장"
        )

        if ner.get("decision") != "문서탐색":
            print("     (문서탐색이 아니므로 검색 스킵)")
        else:
            preview = [r for r in rows if r["page_index"] != -1][:12]
            for i, r in enumerate(preview, 1):
                print(
                    f"       - Top{i}: {os.path.basename(r['doc_path'])} | "
                    f"p.{r['page_index']} (split={r.get('split_index','')}) | "
                    f"score={r['score']:.4f} | kw={r['matched_keywords']}"
                )

        print(f"     처리 시간: {fmt_sec(dt)}")

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
