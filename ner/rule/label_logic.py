def resolve_labels(current_labels: dict, previous_labels: dict = None):
    """
    모델에서 추출된 라벨(current_labels)을 Flow Chart 규칙에 따라 보정 및 결합

    Args:
        current_labels (dict): 모델 결과 예: {"UNI": "연세대학교", "TYPE": "", "KEYWORD": "논술"}
        previous_labels (dict): 이전 질문 저장소에서 불러온 값

    Returns:
        dict: 확정된 라벨 딕셔너리 {"UNI": ..., "TYPE": ..., "KEYWORD": ...}
    """

    resolved = {
        "UNI": current_labels.get("UNI", "").strip(),
        "TYPE": current_labels.get("TYPE", "").strip(),
        "KEYWORD": current_labels.get("KEYWORD", "").strip()
    }

    # 이전 질문 존재 시 보완
    if previous_labels:
        if not resolved["UNI"]:
            resolved["UNI"] = previous_labels.get("UNI", "")
        if not resolved["TYPE"]:
            resolved["TYPE"] = previous_labels.get("TYPE", "")
        # KEYWORD는 항상 새로 받아야 함 (질문 의도는 다를 수 있음)

    return resolved
