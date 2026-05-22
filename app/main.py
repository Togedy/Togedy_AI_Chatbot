# app/main.py
# -*- coding: utf-8 -*-
from __future__ import annotations

# VSCode Run(▶)에서도 루트 임포트 되도록 보정
import sys, logging, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, List, Any, Optional
from dotenv import load_dotenv, find_dotenv

ENV_PATH = find_dotenv(usecwd=True) or str(Path(__file__).resolve().parents[1] / ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("app.main")

from ner.infer_runtime import run_ner
from ner.llm_validate_gemini import validate_with_gemini
from document_retrieval.retriever import TfidfRetriever
from utils.mapping_loader import normalize_uni, uni_to_slug
from llm.answer_gemini import generate_final_answer

def _force_uni_normalized_to_slug(validated: Dict[str, Any], uni_keys: List[str]) -> Dict[str, Any]:
    ents = validated.get("entities", {})
    slug = uni_keys[0] if uni_keys else None
    if not slug:
        for u in ents.get("UNI", []):
            can = normalize_uni(u.get("normalized") or u.get("text") or "")
            s = uni_to_slug(can)
            if s:
                slug = s; break
    if slug:
        for u in ents.get("UNI", []):
            u["normalized"] = slug
            u.setdefault("source", "llm")
        log.info("UNI.normalized → slug '%s' 로 강제 통일", slug)
    else:
        log.warning("UNI slug를 계산하지 못했습니다 (매핑 파일 확인).")
    return validated

def _log_ner_diff(ner_out: Dict[str, List[str]], validated: Dict[str, Any], logger):
    ents = validated.get("entities", {})
    def arr(k): return [x.get("normalized") or x.get("text") for x in ents.get(k, []) if (x.get("normalized") or x.get("text"))]
    before = {k: ner_out.get(k, []) for k in ("UNI","TYPE","KEYWORD")}
    after  = {k: arr(k) for k in ("UNI","TYPE","KEYWORD")}
    logger.info("[2] NER→LLM 비교: BEFORE=%s | AFTER=%s", before, after)

def main(
    question: str,
    frist: bool,
    *,
    ner_model_dir: Optional[str] = None,
    validate_model: Optional[str] = None,
    top_k: int = 3,
    ner_out_override: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    log.info("=== Pipeline start ===")
    log.info("CWD=%s | ROOT=%s | ENV=%s", os.getcwd(), ROOT, ENV_PATH)
    log.info("PYTHON=%s", sys.executable)
    log.info("QUESTION='%s' | first=%s", question, frist)

    # 1) NER
    if ner_out_override is None:
        ner_model_dir = ner_model_dir or os.getenv("NER_MODEL_DIR") or "results/final_model"
        log.info("[1] NER 시작 (model_dir=%s)", ner_model_dir)
        _, _, ner_out = run_ner(question, model_dir=ner_model_dir)
    else:
        log.info("[1] 외부 NER 결과 사용")
        ner_out = ner_out_override
    log.info("[1] NER 결과: %s", ner_out)

    # 2) LLM 검증
    validate_model = validate_model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    log.info("[2] LLM 검증 시작 (model=%s)", validate_model)
    validated = validate_with_gemini(question=question, ner_entities=ner_out, model_name=validate_model)
    log.info("[2] LLM 검증 결과: verdict=%s entities_keys=%s",
             validated.get("verdict"), list(validated.get("entities", {}).keys()))
    _log_ner_diff(ner_out, validated, log)

    # 3) 필터 추출
    rt = TfidfRetriever()
    uni_keys, type_keys = rt.extract_filters(validated)
    log.info("[3] filters: uni_keys=%s, type_keys=%s", uni_keys, type_keys)

    # 4) UNI.normalized → slug 강제
    validated = _force_uni_normalized_to_slug(validated, uni_keys)

    # 5) TXT 우선 검색
    log.info("[5] TXT 코퍼스 로딩")
    docs_txt = rt.load_txt_corpus(uni_keys=uni_keys or None, type_keys=type_keys or None)
    log.info("[5] TXT docs=%d", len(docs_txt))
    if not docs_txt:
        log.warning("[5] TXT가 없어 혼합 코퍼스로 폴백")
        docs_txt = rt.load_corpus(uni_keys=uni_keys or None, type_keys=type_keys or None)
        log.info("[5] Mixed docs=%d", len(docs_txt))

    # 6) 인덱싱 & 검색(+컷오프)
    min_score = float(os.getenv("TFIDF_MIN_SCORE", "0.0"))
    log.info("[6] TF-IDF 인덱싱 & 검색 (top_k=%d, min_score=%.2f)", top_k, min_score)
    rt.build(docs_txt)
    hits = rt.search(validated, top_k=top_k, min_score=min_score)
    log.info("[6] 검색 결과 개수=%d", len(hits))
    if not hits and min_score > 0.0:
        log.warning("[6] 컷오프(min_score=%.2f)로 모두 탈락 → 임계값 낮춰 재시도", min_score)
        hits = rt.search(validated, top_k=top_k, min_score=0.0)

    # 6.1) 묶음/페이지 랭킹/표 추출
    query_for_pages = TfidfRetriever._compose_query(validated)
    page_hits_for_answer = []
    top_docs = []

    for i, (d, score) in enumerate(hits, 1):
        bundle = rt.bundle_neighbors(d.source)

        pages = []
        if bundle.get("text"):
            pages = rt.rank_pages(bundle["text"], query=query_for_pages, top_n=3)
            for p in pages:
                log.info("[6.%d] page=%s score=%.3f excerpt=%.120s", i, p["page"], p["score"], p["excerpt"])

        # (선택) 표 추출
        table_rows = []
        table_path = bundle.get("tables")
        if table_path:
            try:
                from document_retrieval.table_pick import pick_rows_by_keywords
                must = [kw.get("normalized") or kw.get("text")
                        for kw in validated.get("entities", {}).get("KEYWORD", [])
                        if (kw.get("normalized") or kw.get("text"))]
                anys = ["모집인원","정원","인원","컴퓨터","컴퓨터공학","컴퓨터공학부"]
                table_rows = pick_rows_by_keywords(table_path, must_keywords=must, any_keywords=anys, max_rows=10)
            except Exception as e:
                log.warning("[6.%d] 표 파싱 실패: %s", i, e)

        top_docs.append({
            "rank": i,
            "score": float(score),
            "doc_id": d.doc_id,
            "uni_key": d.uni_key,
            "type_key": d.type_key,
            "source_text": bundle.get("text"),
            "source_tables": table_path,
            "source_pdf": bundle.get("pdf"),
            "preview": (d.text[:220] + "…") if len(d.text) > 220 else d.text,
            "page_hits": pages,
            "table_hits": table_rows
        })

        page_hits_for_answer.append({
            "source_text": bundle.get("text"),
            "uni_key": d.uni_key,
            "type_key": d.type_key,
            "pages": pages
        })

    # 7) 최종 답변
    final_answer = generate_final_answer(
        question=question,
        validated=validated,
        page_hits=page_hits_for_answer[:2],
        table_rows=(top_docs[0].get("table_hits") if top_docs else None)
    )
    log.info("[7] 최종 답변 생성 완료")

    out = {
        "request": {"question": question, "first": bool(frist)},
        "ner_out": ner_out,
        "validated": validated,
        "top_docs": top_docs,
        "final_answer": final_answer
    }
    log.info("=== Pipeline done ===")
    return out

if __name__ == "__main__":
    sentence = os.getenv("QUESTION", "건국대 수시 컴퓨터공학부 모집정원은?")
    result = main(sentence, True)
    print(json.dumps(result, ensure_ascii=False, indent=2))