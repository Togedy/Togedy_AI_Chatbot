from predict import predict_ner

sentence = input("문장을 입력하세요: ")
results = predict_ner(sentence)

print("\n[단어]       [예측 라벨]")
for token, label in results:
    print(f"{token:<10} {label}")
