"""Request-local student context; never a source for university requirements."""
from dataclasses import dataclass, field
import re


@dataclass
class ConversationContext:
    profile: dict = field(default_factory=dict)
    current_profile: dict = field(default_factory=dict)
    continuing_counseling: bool = False


def student_details(text):
    details = {}
    # Preserve the student's wording, including negatives. Do not infer scores
    # or achievement. Decimal points in grades are not sentence boundaries.
    sentences = re.split(r"(?<=[!?。])\s*|\n+|(?<=[요다])\.\s*", text or "")
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or any(t in sentence for t in ("예를 들어", "친구", "다른 학생")):
            continue
        for name in ("내신", "수능", "모의고사", "백분위"):
            match = re.search(re.escape(name) + r"(?:\s*(?:성적|등급|점수|평균))?\s*(?:은|는|이|가)?\s*\d+(?:\.\d+)?(?:\s*(?:등급|점|이에요|예요|이야|야))?", sentence)
            if match and not re.match(r"\s*(?:%|퍼센트|프로)", sentence[match.end():]):
                hypothetical = re.match(r"\s*(?:이면|이라면|일\s*경우|정도|맞아야|받아야|필요)", sentence[match.end():])
                asserted = "?" not in sentence or any(t in sentence for t in ("인데", "이고", "이에요", "예요", "입니다"))
                if not hypothetical and asserted:
                    details[name] = match[0]
        if re.search(r"(?:물리|화학|생명|지구과학|수학|과학|과목).{0,45}(?:들었|듣지|안\s*들|이수|선택했|배웠)", sentence):
            details["과목"] = sentence[:300]
        if re.search(r"(?:탐구|실험|프로젝트|동아리|보고서).{0,50}(?:했|진행|작성|안\s*했|없)", sentence):
            details["활동"] = sentence[:300]
        if re.search(r"(?:학과|학부|전공)(?:를|을)?\s*(?:희망|생각|목표|가고|지원하고|요[.!]*$)", sentence):
            details["희망전공"] = sentence[:300]
    return details


def build_conversation_context(previous, current):
    current = current or ""
    if any(t in current.replace(" ", "") for t in ("새질문", "다른질문", "이전내용잊", "내얘기아니", "다른사람")):
        return ConversationContext()
    old, new = student_details(previous), student_details(current)
    counseling = any(t in (previous or "") for t in (
        "불리", "불이익", "괜찮", "고민", "힘들", "가능", "물리", "내신", "탐구", "성적",
    ))
    return ConversationContext({**old, **new}, new, bool(new and (old or counseling)))


def next_counseling_question(plan):
    context = getattr(plan, "conversation_context", None)
    profile = context.profile if context else {}
    if "물리" in plan.question and "과목" not in profile:
        return "물리 대신 어떤 과목을 이수했고, 그 과목에서 어떤 탐구 활동을 했는지 알려주시겠어요?"
    if "과목" in profile and "활동" not in profile:
        return "그 과목에서 진행한 탐구나 활동이 있다면, 어떤 내용을 다뤘는지 알려주시겠어요?"
    if "활동" in profile:
        if not plan.majors and not profile.get("희망전공"):
            return "지원하려는 학과나 전공은 무엇인가요?"
        if re.search(r"안\s*했|없", profile["활동"]):
            return "별도 탐구 활동 외에, 수업에서 관심을 갖고 공부한 내용이 있나요?"
        return "그 활동에서 직접 세운 질문과 조사·실험한 과정을 조금 더 설명해 주시겠어요?"
    if not any(k in profile for k in ("내신", "수능", "모의고사", "백분위")):
        return "현재 성적과 희망하는 전공을 알려주시겠어요?"
    return "지원하려는 학과와 전형은 무엇인가요?"


def counseling_followup(plan):
    context = getattr(plan, "conversation_context", None)
    if not context or not context.continuing_counseling:
        return None
    # An explicit factual request must still run through grounded retrieval.
    if any(t != "서류평가" for t in plan.topics):
        return None
    if any(t in plan.question for t in ("평가 기준", "평가기준", "반영", "불이익", "불리", "지원자격")):
        return None
    statements = " / ".join(dict.fromkeys(context.current_profile.values()))
    answer = f"말씀해 주신 ‘{statements}’ 내용을 바탕으로 이어서 살펴볼게요."
    previous_scores = [v for k, v in context.profile.items()
                       if k in ("내신", "수능", "모의고사", "백분위") and k not in context.current_profile]
    if previous_scores:
        answer += " 앞서 말씀하신 성적도 함께 고려할게요: " + " / ".join(previous_scores) + "."
    if "활동" in context.profile and not re.search(r"안\s*했|없", context.profile["활동"]):
        answer += " 활동 이름만으로 유불리를 판단하기보다, 어떤 질문을 세우고 무엇을 조사하거나 실험했는지 정리해 보면 좋아요."
    else:
        answer += " 성적이나 이수 과목만으로 합격 가능성을 단정하지 않고, 희망 전공과 준비 과정을 함께 살펴볼게요."
    return answer + "\n\n" + next_counseling_question(plan), "답변 생성"
