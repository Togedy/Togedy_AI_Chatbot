"""Grounded answers with one question plan and a shared evidence gate."""
import json
from collections import defaultdict
from pathlib import Path

from question_plan import build_question_plan, local_response
from llm.evidence import EVIDENCE_INSTRUCTIONS, NO_EVIDENCE, evidence_blocks, validate_document_answer
from llm.table_evidence import quota_answer, english_grade_answer, calendar_table_answer
from llm.reviewed_facts import discovery_answer
from llm.major_resolution import major_name_clarification
from llm.semantic_review import review_document_answer
from llm.reviewed_topics import load_topics, select_topic, render_topic, source_rows as reviewed_source_rows


def answer_with_plan(text, uni_ex, type_ex, kw_ex, api_key, gemini_model,
                     llm_model="gpt-4o-mini", top_pages=3, quota_pages_per_pair=2,
                     ner_override=None):
    # Import at call time: CLI and API use the same adapters; test doubles remain usable.
    import generate_answers as ans

    if ner_override is None and all(ex is not None for ex in (uni_ex, type_ex, kw_ex)):
        ner_override = ans._extract_ner_direct(text, uni_ex, type_ex, kw_ex)
    plan = build_question_plan(text, ner_override)

    def result(answer, decision="답변 생성", pairs=None, stats=None):
        pairs = pairs or {}
        return {
            "input": text, "ner_uni": plan.universities, "ner_type": plan.rounds,
            "ner_kw": plan.keywords, "decision": decision, "answer": answer,
            "pair_to_rows": pairs, "stats": stats or {},
            "sources": ans.build_sources({p: rows for p, rows in pairs.items() if rows}),
        }

    local = local_response(plan)
    if local:
        return result(*local, stats={"handler": plan.intent})
    clarification = major_name_clarification(plan)
    if clarification:
        return result(clarification, "재질문", stats={"handler": "major_clarification"})

    reviewed = discovery_answer(plan)
    if reviewed:
        return result(reviewed[0], "문서탐색", reviewed[1], {"handler": "reviewed_table", "scope": "verified_examples"})

    page_limit = max(top_pages, len(plan.topics) * 3)
    rows, stats, _ = ans.search_top_pages_for_query(
        text, uni_ex, type_ex, kw_ex, api_key, gemini_model,
        top_pages=page_limit, query_plan=plan,
    )
    groups = defaultdict(list)
    for row in rows:
        if row.get("matched_uni") and row.get("matched_type"):
            groups[(row["matched_uni"], row["matched_type"])].append(row)
    if not groups:
        return result("확인할 대학과 전형의 연결이 불명확합니다. 대학별로 어떤 전형을 확인할지 알려주시겠어요?", "재질문", stats=stats)

    if plan.intent == "discovery":
        ranked = sorted(groups, key=lambda p: max(float(r.get("score", 0)) for r in groups[p]), reverse=True)
        groups = {pair: groups[pair] for pair in ranked[:6]}
        stats["discovery_pairs_considered"] = len(ranked)
        stats["discovery_pairs_answered"] = len(groups)
    answers = [plan.advice] if plan.advice else []
    source_rows = {}
    topic_records = load_topics()
    stats["evidence"] = []
    for (uni, typ), candidates in groups.items():
        direct_topics = {topic: select_topic(topic_records, plan, uni, typ, topic) for topic in plan.topics}
        candidate_paths = {Path(r["doc_path"]).resolve() for r in candidates if r.get("doc_path")}
        direct_topics = {topic: record if record and all(
            Path(row["doc_path"]).resolve() in candidate_paths for row in reviewed_source_rows(record)) else None
                        for topic, record in direct_topics.items()}
        extra_rows = [row for record in direct_topics.values() if record for row in reviewed_source_rows(record)]
        valid = ans.usable_document_rows(candidates)
        # Keep retrieval's per-topic selection; a second score sort loses coverage.
        valid = valid[:page_limit]
        known = {(r["doc_path"], r["page_index"]) for r in valid}
        valid.extend(r for r in extra_rows if (r["doc_path"], r["page_index"]) not in known)
        if "모집인원" in plan.topics and plan.majors:
            target_terms = plan.majors + ans.collect_major_alias_info(valid)[0]
            valid = [r for r in valid if any(
                ans.normalize_for_match(m) in ans.normalize_for_match(ans.load_page_text(r["doc_path"], r["page_index"]))
                for m in target_terms)]
        source_rows[(uni, typ)] = []
        if not valid:
            if plan.intent != "discovery":
                answers.append(f"{uni} {typ}\n{NO_EVIDENCE}")
            continue
        pages = {r["page_index"]: ans.load_page_text(r["doc_path"], r["page_index"]) for r in valid}
        if plan.topics == ["모집인원"] and plan.majors:
            table_answer, table_pages = quota_answer(pages, plan.majors)
            if table_pages:
                source_rows[(uni, typ)] = [r for r in valid if r["page_index"] in table_pages]
                answers.append(f"{uni} {typ}\n{table_answer}")
                stats["evidence"].append({"university": uni, "round": typ, "cited_pages": table_pages, "handler": "verified_table"})
                continue
        bank = evidence_blocks(pages)
        topic_answers, all_cited = [], []
        for topic in plan.topics or [""]:
            if direct_topics.get(topic):
                record = direct_topics[topic]
                topic_answers.append(f"[{topic}]\n{render_topic(record)}")
                all_cited.extend(s["page"] for s in record["sources"])
                stats.setdefault("coverage", []).append({"university": uni, "round": typ, "topic": topic, "status": "reviewed_table"})
                continue
            if topic == "전형일정":
                direct, direct_pages = calendar_table_answer(valid[0]["doc_path"], pages, plan.tracks)
                if direct_pages:
                    topic_answers.append(f"[전형일정]\n{direct}")
                    all_cited.extend(direct_pages)
                    stats.setdefault("coverage", []).append({"university": uni, "round": typ, "topic": topic, "status": "verified_csv_excerpt"})
                    continue
            if topic == "영어등급":
                direct, direct_pages = english_grade_answer(pages)
                if direct_pages:
                    topic_answers.append(f"[영어등급]\n{direct}")
                    all_cited.extend(direct_pages)
                    stats.setdefault("coverage", []).append({"university": uni, "round": typ, "topic": topic, "status": "verified_table"})
                    continue
            prompt = (
                f"사용자 질문: {text}\n이번 답변 대상: {uni} {typ}\n"
                f"이번에 답할 항목: {topic}\n"
                f"세부 전형: {', '.join(plan.tracks)}\n"
                f"모집단위: {', '.join(plan.majors)}\n"
                f"필수 조건: {json.dumps(plan.constraints, ensure_ascii=False)}\n"
                "사용자의 배경 설명을 지원 대상으로 바꾸지 마세요. 질문에 있는 대학 우열이나 합격 가능성은 추측하지 마세요.\n"
                "여러 전형의 표는 열 제목을 확인하고, 총인원과 전형별 인원을 구분하세요. "
                "실기 없는 전형은 해당 미술 모집단위가 그 전형으로 선발되는 근거도 있어야 합니다.\n"
                "각 요청 항목에 해당하는 확인된 사실을 모두 답하세요. 서류는 지원자 유형별로 구분하세요.\n"
                "수능최저 유무 질문은 적용 여부를 별도 문장으로 먼저 답하세요. 세부 등급은 정확한 계열 행을 확인했을 때만 별도 문장으로 답하세요.\n"
                "미술 전형의 반영비율은 원문에서 평가요소와 퍼센트가 직접 연결된 경우만 답하세요. 여러 단계 표의 100을 최종 선발 비율로 해석하지 마세요.\n"
                "공통 원서접수·일정은 대학 공통 일정이라고 명시하세요. 원문에 없는 학과명을 공통 일정 문장에 붙이지 마세요.\n"
                "전형방법에는 반드시 세부 전형명과 적용 계열을 명시하고, 모집인원을 묻지 않았다면 인원 숫자를 덧붙이지 마세요.\n"
                "제출서류는 국내고·검정고시·국외고 및 온라인 제공 동의 여부의 차이를 원문에서 확인해 구분하세요.\n"
                "<source_blocks>\n" + json.dumps(bank, ensure_ascii=False) + "\n</source_blocks>"
            )
            answer, cited = NO_EVIDENCE, []
            for attempt in range(2):
                raw = ans.call_llm(user_prompt=prompt, model=llm_model, temperature=0.0,
                                   system_prompt=ans.EXPERT_SYSTEM_PROMPT + EVIDENCE_INSTRUCTIONS)
                raw, checks = review_document_answer(raw, bank, plan, topic, uni, typ, ans.call_llm, llm_model)
                stats.setdefault("semantic_review", []).append({"university": uni, "round": typ, "topic": topic, "attempt": attempt + 1, "checks": checks})
                candidate, candidate_pages = validate_document_answer(raw, pages, blocks=bank, university=uni, constraints=plan.constraints,
                                                                     requested_topics=[topic], requested_majors=plan.majors)
                if candidate_pages:
                    answer, cited = candidate, candidate_pages
                rejected = [c for c in checks if c.get("supported") is not True]
                if not rejected:
                    break
                prompt += "\n원문 대조에서 발견된 문제: " + json.dumps(rejected, ensure_ascii=False) + "\n이번 항목만 다시 답하세요. 세부 계열과 예외를 명시하고 확인할 수 없는 주장은 생략하세요."
            topic_answers.append((f"[{topic}]\n" if topic else "") + answer)
            all_cited.extend(cited)
            stats.setdefault("coverage", []).append({"university": uni, "round": typ, "topic": topic, "status": "answered" if cited else "unanswered"})
        source_rows[(uni, typ)] = [r for r in valid if r["page_index"] in all_cited]
        stats["evidence"].append({"university": uni, "round": typ, "candidate_pages": list(pages), "cited_pages": sorted(set(all_cited))})
        if all_cited or plan.intent != "discovery":
            answers.append(f"{uni} {typ}\n" + "\n\n".join(topic_answers))
    if plan.intent == "discovery" and any(source_rows.values()):
        answers.insert(0, "보유 모집요강에서 확인한 전형 사례입니다.")
    return result("\n\n".join(answers) or NO_EVIDENCE, "문서탐색", source_rows, stats)
