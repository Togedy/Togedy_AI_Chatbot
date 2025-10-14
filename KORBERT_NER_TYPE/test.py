from predict import predict

examples = [
    "수시 모집 일정 알려줘",
    "정시 원서 접수 기간이 언제야",
    "수시와 정시 둘 다 일정 알려줘",
    "모집일정 궁금해",
    "건대랑 서연고 수시 모집 일정 알려줘",
    "건대 수시 일정이랑 연세대 정시 일정 궁금해"
]

for s in examples:
    out = predict(s)
    print("Sentence:", s)
    print("Tokens:", out["tokens"])
    print("Tags:  ", out["tags"])
    print("TYPE:  ", out["TYPE"])
    print()
