"""Summarize human-reviewed outcomes; never calls an external service."""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_answers import load_page_text
from llm.evidence import _compact, _numbers, validate_document_answer

HERE = Path(__file__).resolve().parent
LABELS = {"complete": "완전 정답", "partial": "부분 충족", "abstained": "답변 보류/미해결", "incorrect": "오답/근거 없는 답변", "excluded": "평가 제외"}
REVIEW = {
    1: ("partial", "서류제출·합격자 발표 일부만 답함. 원서접수·논술·면접 일정이 빠지고 전형별 예외도 구분하지 않음."),
    2: ("abstained", "전형방법과 제출서류를 검색했으나 생성된 두 문장 모두 인용 검증에서 탈락."),
    3: ("abstained", "KU일반학생을 건국대 전형으로 연결하지 못해 문서 검색 없이 확인 불가로 응답."),
    4: ("partial", "4개 대학은 인식. 건국대 일부 일정과 서울대 지역균형 접수·서류 일정만 답하고 연세대·고려대는 보류."),
    5: ("partial", "수시 컴퓨터공학부 49명은 원문과 일치. 정시 36명은 원문에 있으나 인용문 재작성으로 검증 탈락."),
    6: ("abstained", "부산 거주자의 주거·비용·전공 선택 기준 없이 주관적이라 답할 수 없다고만 함."),
    7: ("abstained", "대학 비교를 보류하고 전공·진로·선호 기준을 안내하지 않음."),
    8: ("abstained", "연고대 인식은 성공했으나 지원 판단에 필요한 조건 안내 없이 거절."),
    9: ("abstained", "연세대 정시·고려대 수시 매칭은 맞지만 일정 문장이 모두 검증 탈락. 합격 가능성 비교도 해결하지 못함."),
    10: ("abstained", "합격 가능성 단정은 피했지만 본인 성적·환산점수 등 비교 기준은 제시하지 않음."),
    11: ("incorrect", "'논술 최저학력기준이 없습니다'라고 답함. 보유 건국대 수시 요강 p.43은 '수능최저학력기준 있음'을 명시. 검색 경로에 진입하지 않음."),
    12: ("excluded", "'작년' 자료를 묻는 질문이므로 사용자 지정 제외 범위. 실제 실행 응답은 확인 불가."),
    13: ("abstained", "실기 없는 미대의 대학·학과·전형 예시를 하나도 제시하지 못함."),
    14: ("abstained", "예비번호 비교를 보류. 실제 모집단위 확인이나 충원자료 비교 방법을 제공하지 않음."),
    15: ("abstained", "6:1과 유사한 모집단위나 경쟁률 자료를 제시하지 못함."),
    16: ("abstained", "실기 60% 이상 조건에 맞는 전형을 제시하지 못함."),
    17: ("abstained", "수능 60% 이상 조건에 맞는 전형을 제시하지 못함."),
    18: ("abstained", "전형 분류가 비어 문서 검색 없이 확인 불가. 실제 요강의 공과대학자유전공학부로도 연결하지 못함."),
    19: ("incorrect", "'타학교 산업공학과 목표'를 건국대 산업공학과 질문으로 바꿈. 문서 근거 없이 물리 미이수 불이익이 클 수 있다고 답함. 허위로 확정된 사실과 구분하여 근거 없는 답변으로 판정."),
    20: ("abstained", "합격 백분위를 임의 생성하지 않았지만 입결·환산점수 정보를 제공하지 못함."),
    21: ("abstained", "위치·시설은 보류하고 공식 사이트 문의를 권함. 안전한 보류이지만 실제 질문 해결은 아님."),
    22: ("abstained", "대학과 전형을 특정하지 못해 기타과목 중요도에 대한 근거를 제공하지 못함."),
    23: ("abstained", "심심하다는 말에 '확인할 수 없는 사항'이라고 응답. 자연스러운 입시 안내로 전환하지 못함."),
}

# Document pairs in the actual captured LLM call order.
DOC_CALLS = {
    1: [("konkuk", "susi")], 2: [("yonsei", "jungsi")],
    4: [("konkuk", "susi"), ("seoul", "susi"), ("yonsei", "susi"), ("korea", "susi")],
    5: [("seoul", "jungsi"), ("seoul", "susi")],
    9: [("korea", "susi"), ("yonsei", "jungsi")],
}


def main():
    result = json.loads((HERE / "image_questions_results.json").read_text(encoding="utf-8"))
    rubric = json.loads((HERE / "image_questions_rubric.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in result["items"]] == list(range(1, 24))
    assert not any(item.get("error_type") for item in result["items"])
    reviewed = []
    diagnostics = []
    for item in result["items"]:
        number = item["id"]
        verdict, reason = REVIEW[number]
        reviewed.append(dict(id=number, verdict=verdict, reason=reason))
        for call, (uni, typ) in zip(item["llm_calls"], DOC_CALLS.get(number, [])):
            payload = json.loads(call["output"])
            path = ROOT / "university" / uni / f"{typ}_text.txt"
            for claim in payload["claims"]:
                pages = {e["page"]: load_page_text(str(path), e["page"]) for e in claim["evidence"]}
                text, accepted_pages = validate_document_answer(json.dumps({"claims": [claim]}), pages)
                quote_matches = all(_compact(e["quote"]) in _compact(pages[e["page"]]) for e in claim["evidence"])
                numbers_match = _numbers(claim["text"]) <= _numbers("\n".join(e["quote"] for e in claim["evidence"]))
                diagnostics.append(dict(question_id=number, text=claim["text"],
                                        accepted=bool(accepted_pages), quote_matches=quote_matches,
                                        numbers_match=numbers_match, document=str(path)))
    counts = Counter(item["verdict"] for item in reviewed)
    included = [item for item in result["items"] if item["id"] != 12]
    denominator = len(included)
    summary = {
        "denominator": denominator, "counts": dict(counts),
        "strict_complete_percent": counts["complete"] / denominator * 100,
        "at_least_partial_percent": (counts["complete"] + counts["partial"]) / denominator * 100,
        "abstained_percent": counts["abstained"] / denominator * 100,
        "incorrect_or_unsupported_percent": counts["incorrect"] / denominator * 100,
        "retrieval_route_ids": [i["id"] for i in included if i["decision"] == "문서탐색"],
        "llm_calls": sum(len(i["llm_calls"]) for i in result["items"]),
        "document_claims": len(diagnostics),
        "accepted_document_claims": sum(d["accepted"] for d in diagnostics),
        "seconds_total_excluding_pipeline_init": sum(i["elapsed_seconds"] for i in result["items"]),
        "median_seconds": statistics.median(i["elapsed_seconds"] for i in result["items"]),
    }
    (HERE / "image_questions_scores.json").write_text(json.dumps(dict(summary=summary, reviews=reviewed, claim_diagnostics=diagnostics), ensure_ascii=False, indent=2), encoding="utf-8")

    report = ["# 이미지 23문항 실제 실행 평가", "", "## 결과", "",
              f"- 실행 모델: `{result['openai_model']}`. 질문별 독립적인 첫 질문으로 1회씩 실행했습니다.",
              "- 9번의 두 문장은 한 문항으로 처리했습니다. 12번은 사용자 지정 연도 질문 제외 범위이므로 분모에서 제외했습니다.",
              "- 이 평가는 보유 모집요강과 질문 충족 여부에 대한 수동 검토입니다. 최신 공식 모집요강 검증이나 독립 평가자의 채점은 아닙니다.",
              f"- **완전 정답률: {summary['strict_complete_percent']:.1f}% ({counts['complete']}/{denominator})**.",
              f"- **일부라도 유효한 정보를 제공한 비율: {summary['at_least_partial_percent']:.1f}% ({counts['partial']}/{denominator})**. 완전 정답률과 다릅니다.",
              f"- 답변 보류/미해결: {counts['abstained']}/{denominator} ({summary['abstained_percent']:.1f}%). 오답/근거 없는 답변: {counts['incorrect']}/{denominator} ({summary['incorrect_or_unsupported_percent']:.1f}%).",
              "- 답변 보류를 정답으로 세지 않았습니다. 보류에는 타당한 보류와 검색·생성 실패가 섞여 있으므로 모두 사실 오류라는 뜻도 아닙니다.",
              "- 기존 회귀 테스트 46개 통과는 기능 검증 결과이며, 이 질문 세트의 답변 정확도가 아닙니다.", "",
              "## 판정 기준", "",
              "완전 정답은 질문의 핵심 요구와 명시한 모든 대상을 충족한 경우입니다. 부분 충족은 검증 가능한 정보를 일부 제공했지만 핵심 항목이나 대상이 누락된 경우입니다. 주관적 상담은 근거 없는 우열·합격 예측을 피하면서 질문에 맞는 판단 기준을 안내해야 해결로 인정합니다. 단순 거절은 보류로 분류합니다.", "",
              "## 주요 원인", "",
              "1. 문서 검색에 진입한 문항은 1·2·4·5·9번뿐입니다. 논술·학생부종합처럼 수시/정시가 직접 쓰이지 않은 표현, 대학 미지정 탐색, 일부 학과·항목에서 슬롯이 비어 일반 답변으로 빠집니다.",
              "2. 검색 없이 일반 답변으로 간 경우 인용 검증을 거치지 않습니다. 11번에서 최저기준 유무를 틀렸고, 19번에서 물리 미이수 불이익을 근거 없이 추측했습니다.",
              "3. 인용 검증이 일부 잘못된 생성은 막았지만 정상 정보도 탈락시켰습니다. 5번 정시 36명은 원문에 있으나 모델이 표를 '모집단위: 컴퓨터공학부, 정원 내: 36'으로 재작성해 원문 일치에 실패했습니다.",
              "4. 1번 원서접수 문장은 본문에 학년도를 넣고 인용에는 넣지 않아 숫자 검증에서 탈락했습니다. 9번은 날짜의 점 표기와 자연어 표기, 축약 인용 등이 겹쳐 최종 문장이 전부 제외되었습니다.",
              f"5. 문서 답변 호출 {sum(len(v) for v in DOC_CALLS.values())}건에서 생성된 주장 {len(diagnostics)}개 중 인용 검증을 통과한 주장은 {summary['accepted_document_claims']}개입니다. 이는 문장 단위 내부 지표이며 문항 정확도가 아닙니다.", "",
              "## 원문 대조 근거", "",
              "- 건국대 수시 요강 PDF p.43~44: 논술전형의 '수능최저학력기준 있음', 최저기준 충족자 중 선발. [텍스트](../university/konkuk/susi_text.txt)",
              "- 서울대 수시 요강 PDF p.10: 컴퓨터공학부 지역균형 9, 일반 36, 기회균형 4, 합계 49. [텍스트](../university/seoul/susi_text.txt)",
              "- 서울대 정시 요강 PDF p.13: 컴퓨터공학부 정원 내 일반전형 36. [텍스트](../university/seoul/jungsi_text.txt)",
              "- 건국대 수시 요강 PDF p.19: 전체 전형 접수·서류 제출, 논술·면접, 전형별 합격자 발표 일정. [텍스트](../university/konkuk/susi_text.txt)",
              "- 서울대 수시 요강 PDF p.5: 지역균형 접수·서류 제출·면접·발표 일정. [텍스트](../university/seoul/susi_text.txt)", "",
              "## 문항별 판정", "", "| 번호 | 판정 | 근거 |", "|---|---|---|"]
    report.extend(f"| {i['id']} | {LABELS[i['verdict']]} | {i['reason']} |" for i in reviewed)
    report += ["", "## 실제 질문과 최종 응답", ""]
    for item in result["items"]:
        report += [f"### {item['id']}. {LABELS[REVIEW[item['id']][0]]}", "", item["question"], "", "**최종 응답**", "", item["answer"], ""]
    report += ["## 재현 자료", "", "- `run_image_questions.py`: 실제 실행 및 외부 호출 없는 `--offline` 검색 평가.",
               "- `image_questions_results.json`: NER·최종 답변·출처·LLM 원응답·실행시간.",
               "- `image_questions_offline.json`: 로컬 검색 단계 결과.",
               "- `image_questions_rubric.json`: 문항별 기대 동작과 채점 기준.",
               "- `image_questions_scores.json`: 수동 판정·계산 결과·인용 검증 진단.",
               "- `score_image_questions.py`: 저장된 결과로 보고서와 집계를 재생성. 외부 API 호출 없음.", "",
               "이번 평가는 현재 동작을 측정했으며 서비스 코드나 모델 설정을 수정하지 않았습니다. 1회 실행의 결과이므로 재실행 시 생성 결과가 달라질 수 있습니다."]
    (HERE / "image_questions_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
