from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


def read_ner_data(file_path):
    sentences, labels = [], []
    words, tags = [], []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                splits = line.split('\t')
                if len(splits) == 2:
                    word, tag = splits
                    words.append(word)
                    tags.append(tag)

    if words:
        sentences.append(words)
        labels.append(tags)

    return sentences, labels


def get_label_list(label_path):
    with open(label_path, encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)
    pred_list, label_list = [], []

    for pred, lab in zip(predictions, labels):
        for p_, l_ in zip(pred, lab):
            if l_ != -100:
                pred_list.append(p_)
                label_list.append(l_)

    report = classification_report(label_list, pred_list, output_dict=True, zero_division=0)
    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }
