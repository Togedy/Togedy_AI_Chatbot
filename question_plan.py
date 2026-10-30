"""One interpretation of a question, shared by retrieval and generation."""
from dataclasses import dataclass, field
import re
from query_constraints import parse_percentages, invalid_conditions


UNIVERSITIES = {
    "건국대": ("건국대학교", "건국대", "건대"),
    "서울대": ("서울대학교", "서울대"),
    "연세대": ("연세대학교", "연세대", "연대"),
    "고려대": ("고려대학교", "고려대", "고대"),
    "서강대": ("서강대학교", "서강대"),
    "성균관대": ("성균관대학교", "성균관대", "성대"),
    "한양대": ("한양대학교", "한양대"),
}
TOPICS = {
    "모집인원": ("모집인원", "선발인원", "몇명", "뽑", "정원"),
    "전형일정": ("일정", "원서접수", "합격자발표", "등록기간"),
    "제출서류": ("제출서류", "필요서류", "구비서류", "서류도", "서류는"),
    "전형방법": ("전형방법", "평가방법", "평가방식", "반영비율", "들어가는방법", "전형안내"),
    "지원자격": ("지원자격", "자격요건"),
    "수능최저": ("수능최저", "최저학력", "논술최저"),
    "영어등급": ("영어등급", "영어환산", "등급환산"),
    "서류평가": ("서류평가", "물리", "기타과목", "과목중요", "이수", "불이익"),
}


TRACKS = (
    "논술", "학생부종합", "학생부교과", "KU일반학생", "KU자기추천",
    "지역균형", "일반전형", "고른기회전형", "학교추천전형",
    "학업우수전형", "계열적합전형", "미술특기자",
)


def compact(text):
    return re.sub(r"\s+", "", text or "")


def normalize_question_text(question):
    """Remove retry-form label help before both NER and rule interpretation."""
    return re.sub(
        r"(?m)^(\s*(?:대학|전형|학과·모집단위|희망 학과·전공|궁금한 내용))\([^\n]*?\)(?=\s*[:：])",
        r"\1", question,
    )


def values(ner, key):
    value = (ner or {}).get(key, (ner or {}).get({"UNI": "uni", "TYPE": "type", "KEYWORD": "keywords"}[key], []))
    return list(dict.fromkeys(str(x).strip() for x in (value if isinstance(value, list) else [value]) if x))


@dataclass
class QuestionPlan:
    question: str
    universities: list = field(default_factory=list)
    rounds: list = field(default_factory=list)
    topics: list = field(default_factory=list)
    majors: list = field(default_factory=list)
    keywords: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    intent: str = "fact"
    advice: str = ""
    constraints: dict = field(default_factory=dict)
    conversation_context: object = None

    def ner(self):
        return {"UNI": self.universities, "TYPE": self.rounds, "KEYWORD": self.keywords}


def build_question_plan(question, ner=None):
    # The retry form's examples are label help, not selected values. Leaving
    # '전형(수시·정시):' blank must not count as requesting both rounds.
    question = normalize_question_text(question)
    text = compact(question)
    plan = QuestionPlan(question)
    explicit = [u for u, names in UNIVERSITIES.items() if any(name in text for name in names)]
    if "서연고" in text:
        explicit += ["서울대", "연세대", "고려대"]
    elif "연고대" in text:
        explicit += ["연세대", "고려대"]
    excluded_unis = [u for u, names in UNIVERSITIES.items() if any(
        re.search(re.escape(name) + r"(?:은|는|을|를)?(?:제외|말고|빼고)", text) for name in names)]
    if excluded_unis:
        plan.constraints["excluded_universities"] = excluded_unis
    plan.universities = [u for u in dict.fromkeys(explicit or values(ner, "UNI")) if u not in excluded_unis]
    plan.rounds = [r for r in ("수시", "정시") if r in text] or values(ner, "TYPE")
    plan.tracks = [t for t in TRACKS if t.lower() in text.lower()]
    if "ku일반학생" in text.lower():
        plan.universities = plan.universities or ["건국대"]
        plan.rounds = plan.rounds or ["정시"]
    if not plan.rounds and any(t in plan.tracks for t in ("논술", "학생부종합", "학생부교과", "KU자기추천", "학교추천전형", "학업우수전형", "계열적합전형", "미술특기자")):
        plan.rounds = ["수시"]
    plan.topics = [topic for topic, variants in TOPICS.items() if any(v in text for v in variants)]
    if not plan.topics:
        plan.topics = [k for k in values(ner, "KEYWORD") if k in TOPICS]

    # Background goals and explicitly excluded units must not become targets.
    target_text = re.sub(r"타\s*학교.*?목표[^,?.]*?건국", "건국", question)
    majors = re.findall(r"[가-힣A-Za-z·]+(?:학과|학부|전공|예과|미술과|화과|조소과|디자인과|교육과)", target_text)
    excluded_majors = [m for m in majors if re.search(re.escape(m) + r"\s*(?:은|는|을|를)?\s*(?:말고|제외|빼고)", target_text)]
    if excluded_majors:
        plan.constraints["excluded_majors"] = list(dict.fromkeys(excluded_majors))
    majors = [m for m in majors if m not in excluded_majors]
    if "컴공" in text:
        majors.append("컴퓨터공학부")
    plan.majors = list(dict.fromkeys(majors))
    original = [k for k in values(ner, "KEYWORD") if k not in excluded_majors]
    if target_text != question:
        original = [k for k in original if k in target_text]
    plan.keywords = list(dict.fromkeys(plan.majors + plan.topics + original))

    art = any(term in text for term in ("미대", "미술대학"))
    if art:
        plan.intent = "discovery"
        plan.topics = list(dict.fromkeys(plan.topics + ["전형방법"]))
        plan.keywords = list(dict.fromkeys(plan.keywords + ["실기", "전형방법"]))
        plan.constraints["art"] = True
        if any(s in text for s in ("실기없는", "실기없이", "실기를안보", "실기를보지않")):
            plan.constraints["no_practical"] = True
        percentages = parse_percentages(text)
        if percentages:
            plan.constraints["percentages"] = percentages
            minimum = next((c for c in percentages if c["operator"] == "ge"), None)
            if minimum:
                plan.constraints.update(component=minimum["component"], minimum_percent=minimum["value"])
        missing_component = bool(re.search(r"\d+(?:\.\d+)?(?:%|퍼센트|프로)", text)) and not percentages
        incompatible = invalid_conditions(percentages)
        stages = list(dict.fromkeys(re.findall(r"1단계|2단계|일괄합산", text)))
        if percentages and len(stages) == 1:
            plan.constraints["stage"] = stages[0]
        ambiguous_stages = bool(percentages) and len(stages) > 1
        if plan.constraints.get("no_practical") and any(
                c["component"] == "실기" and ((c["operator"] in ("ge", "eq") and c["value"] > 0)
                                            or (c["operator"] == "gt" and c["value"] >= 0)) for c in percentages):
            incompatible = True
        if missing_component or incompatible or ambiguous_stages:
            plan.intent = "clarify_constraints"
            plan.constraints["clarification"] = (
                "어느 반영 항목의 비율을 말씀하신 건가요? 실기 또는 수능과 원하는 비율 범위를 함께 알려주세요."
                if missing_component else
                "반영 비율을 확인할 단계를 구분해 주세요. 1단계와 2단계 중 원하는 단계와 조건을 각각 알려주세요."
                if ambiguous_stages else
                "입력하신 반영 비율 조건을 함께 만족할 수 없습니다. 0~100% 범위에서 실기·수능 조건을 다시 말씀해 주세요."
            )

    counseling = any(s in text for s in ("메리트", "더좋", "고민", "붙기쉬", "잘붙", "불이익", "불리", "괜찮을까", "가능할까", "힘들까"))
    if counseling:
        if "물리" in text or "불이익" in text:
            plan.advice = "특정 과목을 이수하지 않은 점이 지원에 걸림돌이 될지 고민하고 계시는군요. 다만 과목 미이수나 내신 등급만으로 합격·불합격 또는 불이익의 크기를 판단할 수 없습니다. 어떤 과목을 선택했고 그 안에서 진로와 관련된 학습을 어떻게 이어왔는지, 지원 전형의 평가 기준과 함께 살펴보는 게 좋아요."
        elif any(s in text for s in ("붙기쉬", "잘붙", "정시")):
            plan.advice = "지원 가능성을 비교하고 싶으시군요. 어느 대학이 더 합격하기 쉬운지는 현재 정보만으로 판단할 수 없습니다. 먼저 영역별 성적을 대학별 반영 방식으로 환산한 뒤, 같은 전형과 모집단위의 공식 입시결과를 함께 비교해 보는 게 좋아요."
        else:
            plan.advice = "대학을 고를 때는 희망 전공과 진로에 맞는 교육과정부터 비교해 보면 좋아요. 생활비와 통학·이주 가능성도 함께 생각해 보고요. 지원 가능성은 성적과 지원 전형을 확인한 뒤 따로 살펴볼게요."
        if not plan.topics:
            plan.intent = "counseling"
    if any(s in text for s in ("경쟁률", "백분위", "예비번호", "미달")) and not plan.topics:
        plan.intent = "external_data"
    if any(s in text for s in ("경영관", "시설", "캠퍼스위치")) and not plan.topics:
        plan.intent = "campus"
    if re.fullmatch(r"(?:나)?(?:심심해|안녕|안녕하세요|고마워)[.!?~]*", text):
        plan.intent = "smalltalk"
    if not plan.keywords and plan.intent == "fact":
        plan.intent = "clarify"
    if not plan.universities and plan.intent == "fact":
        plan.intent = "clarify"
    return plan


def local_response(plan):
    """No generated university facts may escape the document evidence path."""
    if plan.intent == "clarify_constraints":
        return plan.constraints["clarification"], "재질문"
    if plan.intent == "smalltalk":
        return "궁금했던 학과나 대학 이야기를 해볼까요? 저는 대학 입시와 전형 정보를 함께 살펴드릴 수 있어요.", "답변 생성"
    if plan.intent == "counseling":
        return plan.advice, "답변 생성"
    if plan.intent == "external_data":
        return ("모집요강만으로는 경쟁률·충원 결과·합격 백분위나 미달 여부를 비교하기 어렵습니다. 해당 대학 입학처의 전형·모집단위별 공식 입시결과가 필요합니다. "
                "합격 가능성은 본인 성적을 대학별 반영 방식으로 환산한 뒤 비교해야 합니다."), "답변 생성"
    if plan.intent == "campus":
        return "건물 위치와 시설은 모집요강으로 확인하기 어렵습니다. 해당 대학의 공식 캠퍼스 지도와 시설 안내에서 확인해야 합니다.", "답변 생성"
    if plan.intent == "clarify":
        if not plan.universities:
            return "어느 대학의 전형을 확인해 드릴까요?", "재질문"
        return "궁금한 학과나 항목을 말씀해 주세요. 모집인원, 전형방법, 지원자격, 제출서류 등을 확인할 수 있습니다.", "재질문"
    return None
