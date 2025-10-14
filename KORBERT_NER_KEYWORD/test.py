# -*- coding: utf-8 -*-
# 사용 예: python test_examples_keyword.py
from predict_keyword import predict

examples = [
    "컴퓨터공학부 모집인원 알려줘",
    "경제학부 전형 일정",
    "컴퓨터공학부와 경제학부 모두 모집일정 궁금해",
    "모집일정과 전공명을 함께 말했어",
    "건대랑 서연고 수시 모집 일정 알려줘",
    "건국대랑 연세대랑 붙으면 누가 이겨?",
    "서성한에서 입결 누가 더 높아?",
    "연세대 출신 연예인 누구 있어?",
    "자율전공 입결이 어느 정도야?"
]

for s in examples:
    out = predict(s)
    print("Sentence:", s)
    print("Tokens:", out["tokens"])
    print("Tags:  ", out["tags"])
    print("KEYWORD:", out["KEYWORD"])
    print()
 