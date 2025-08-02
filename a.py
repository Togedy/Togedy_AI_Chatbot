from data_loader import read_tsv_file

data = read_tsv_file("./data/train.tsv")

print(f"총 문장 수: {len(data['tokens'])}\n")
print("📌 첫 번째 문장:")

for token, label in zip(data["tokens"][1], data["labels"][1]):
    print(f"{token}\t{label}")
