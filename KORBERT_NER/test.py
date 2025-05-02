# test.py
from predict import predict_ner

if __name__ == "__main__":
    sentence = input("문장을 입력하세요: ")
    results = predict_ner(sentence)
    print(f"\n{'[단어]':<10} {'[예측 라벨]'}")
    for token, label in results:
        print(f"{token:<10} {label}")
