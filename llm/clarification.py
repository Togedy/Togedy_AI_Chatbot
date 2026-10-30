"""Clarify missing document targets without changing the public API schema."""


def missing_document_fields(plan):
    # Counseling and cross-university discovery have different requirements.
    if plan.intent not in ("fact", "clarify") or plan.advice:
        return []
    missing = []
    if not plan.universities:
        missing.append("대학")
    if not plan.rounds:
        missing.append("전형(수시·정시)")
    if not plan.topics:
        missing.append("궁금한 내용")
    return missing


def clarification_answer(plan, repeated=False):
    missing = missing_document_fields(plan)
    if not missing:
        return None
    summary = (
        f"대학: {', '.join(plan.universities) or '미확인'} / "
        f"전형: {', '.join(plan.rounds) or '미확인'} / "
        f"학과·모집단위: {', '.join(plan.majors) or '미지정'} / "
        f"궁금한 내용: {', '.join(plan.topics) or '미확인'}"
    )
    if repeated:
        return (
            "아직 " + ", ".join(missing) + " 정보가 부족해 확인할 모집요강을 정하기 어려워요. "
            "질문을 한 번에 정리할 수 있도록 아래 내용을 직접 채워서 알려주세요. 이미 적힌 내용도 바꾸셔도 돼요.\n\n"
            f"현재까지 확인한 내용: {summary}\n\n"
            f"대학: {', '.join(plan.universities)}\n"
            f"전형(수시·정시, 알면 세부 전형명도): {', '.join(plan.rounds)}\n"
            f"학과·모집단위(전체 안내를 원하면 '전체'): {', '.join(plan.majors)}\n"
            f"궁금한 내용(예: 제출서류·일정·모집인원): {', '.join(plan.topics)}\n\n"
            "위 내용을 한 메시지로 보내주시겠어요?"
        )
    return (
        f"현재까지 확인한 내용: {summary}\n\n"
        + ", ".join(missing) + " 정보를 알려주시겠어요? 확인할 모집요강을 정한 뒤 안내해 드릴게요."
    )


def counseling_reprompt(previous_plan, current_plan, resolved_plan, context, previous_text):
    """Infer outstanding counseling context from the two supplied questions."""
    from llm.conversation import student_details
    old = student_details(previous_text)
    if not old or (previous_plan.topics and previous_plan.topics != ["서류평가"] and not previous_plan.advice):
        return None
    # A new concrete document request takes priority over a counseling form.
    if current_plan.topics or current_plan.intent in ("smalltalk", "external_data", "campus", "discovery"):
        return None
    profile = context.profile
    known_major = resolved_plan.majors or profile.get("희망전공")
    missing = []
    if "과목" in old and "활동" not in old and "활동" not in profile:
        missing.append("수업에서 한 탐구·활동(없으면 '없음')")
    elif "활동" in old and not previous_plan.majors and not old.get("희망전공") and not known_major:
        missing.append("희망 학과·전공")
    elif not context.current_profile and not (current_plan.universities or current_plan.rounds or current_plan.majors or current_plan.tracks):
        missing.append("상담을 이어갈 구체적인 상황")
    if not missing:
        return None
    return (
        "아직 " + ", ".join(missing) + " 정보가 부족해요. 내용을 한 번에 정리할 수 있도록 아래 항목을 직접 채워서 알려주세요. "
        "모르는 항목은 '미정', 하지 않은 활동은 '없음'이라고 적으셔도 돼요.\n\n"
        f"대학: {', '.join(resolved_plan.universities)}\n"
        f"전형(수시·정시, 세부 전형): {', '.join(resolved_plan.rounds)}\n"
        f"희망 학과·전공: {', '.join(resolved_plan.majors) or profile.get('희망전공', '')}\n"
        "현재 성적: " + ' / '.join(v for k, v in profile.items() if k in ("내신", "수능", "모의고사", "백분위")) + "\n"
        f"이수 과목: {profile.get('과목', '')}\n"
        f"탐구·활동: {profile.get('활동', '')}\n"
        "궁금한 점: \n\n위 내용을 한 메시지로 보내주시겠어요?"
    )
