from datasets import load_dataset
from torch.utils.data import Dataset
from utils import encode_examples

class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

def load_and_cache_examples(tokenizer, label_list, mode="train"):
    path = f"./KORBERT_NER/data/{mode}.tsv"

    # 헤더 없이 TSV 파일 읽기 + 열 이름 수동 지정
    dataset = load_dataset("csv", data_files=path, delimiter="\t", split="train", column_names=["word", "tag"])

    grouped = []
    sentence, tags = [], []

    for row in dataset:
        word, tag = row["word"], row["tag"]

        #  None 값 제거
        if word is None or tag is None:
            continue

        #  문장 경계 (빈 줄)
        if word == "":
            if sentence:
                grouped.append((sentence, tags))
                sentence, tags = [], []
        else:
            sentence.append(word)
            tags.append(tag)

    # 마지막 문장 처리
    if sentence:
        grouped.append((sentence, tags))

    if len(grouped) == 0:
        raise ValueError(f"No valid sentence/tag pairs were loaded from {path}")

    texts, ner_tags = zip(*grouped)

    # 튜플 → 리스트 변환 (tokenizer 요구 형식)
    texts = [list(t) for t in texts]
    ner_tags = [list(t) for t in ner_tags]

    encodings, encoded_labels = encode_examples(texts, ner_tags, tokenizer, label_list)
    encodings["labels"] = encoded_labels
    return NERDataset(encodings)
