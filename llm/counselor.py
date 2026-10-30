"""Conversation framing around verified facts; never rewrite evidence or numbers."""


def present_answer(answer, plan, decision, has_sources=False):
    if decision == "재질문" or not answer:
        return answer

    # These prompts ask for the student's context, not an ungrounded prediction.
    if plan.intent == "counseling" or plan.advice:
        if getattr(plan, "conversation_context", None):
            from llm.conversation import next_counseling_question
            return answer + "\n\n" + next_counseling_question(plan)
        question = (
            "물리 대신 어떤 과목을 이수했고, 그 과목에서 어떤 탐구 활동을 했는지 알려주시겠어요?"
            if "물리" in plan.question else
            "현재 성적과 희망하는 전공을 알려주시겠어요?"
            if any(term in plan.question for term in ("붙기", "잘붙", "정시", "성적")) else
            "지금 가장 고민되는 건 전공 선택인가요, 성적에 맞는 지원 대학을 찾는 건가요?"
        )
        return answer + "\n\n" + question
    if plan.intent == "external_data":
        if getattr(plan, "conversation_context", None):
            from llm.conversation import next_counseling_question
            return answer + "\n\n" + next_counseling_question(plan)
        return answer + "\n\n희망하는 모집단위와 현재 성적을 알려주시겠어요? 내신인지 수능 성적인지도 함께 말씀해 주세요."
    if not has_sources:
        return answer

    # Keep all verified content intact, including exceptions and source scope.
    introductions = {
        "제출서류": "서류는 본인의 출신 학교와 온라인 제공 동의 여부에 맞춰 확인하는 게 중요해요. 확인된 내용을 함께 볼게요.",
        "전형일정": "지원 준비 순서를 잡을 수 있도록, 확인된 일정부터 함께 볼게요. 세부 전형에 따라 적용되는 일정은 구분해서 봐주세요.",
        "전형방법": "지원 방식을 결정할 때는 어떤 요소로 평가하는지부터 살펴보면 좋아요. 확인된 전형의 기준을 함께 볼게요.",
        "수능최저": "수능최저는 지원 준비에서 따로 확인해야 할 조건이에요. 질문하신 전형에서 확인된 기준을 볼게요.",
        "모집인원": "확인된 모집인원을 볼게요. 인원만으로 합격 가능성을 판단하기보다는 지원 전형과 함께 살펴보는 게 좋아요.",
        "영어등급": "영어 성적이 어떻게 반영되는지 확인된 환산 기준을 함께 볼게요.",
    }
    intro = next((introductions[t] for t in plan.topics if t in introductions), "")
    if plan.intent == "discovery":
        intro = "말씀하신 조건에 맞는지 검토할 수 있도록, 자료에서 확인된 전형 사례를 함께 볼게요."
    return (intro + "\n\n" if intro else "") + answer
