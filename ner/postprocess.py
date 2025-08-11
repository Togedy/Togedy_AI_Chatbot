from konlpy.tag import Okt

okt = Okt()

def is_josa(word):
    """조사 여부 판별"""
    pos_tags = okt.pos(word)
    return all(pos.startswith("J") for _, pos in pos_tags)

# postprocess.py
def postprocess_ner_output(*args):
    # 지원: (tokens, tags) 또는 (sentence, tags, tokens)
    if len(args) == 2:
        tokens, tags = args
    elif len(args) == 3:
        sentence, tags, tokens = args
    else:
        raise TypeError("postprocess_ner_output() expects (tokens, tags) or (sentence, tags, tokens)")

    entities = []
    cur_label, cur_tokens = None, []

    for tok, tag in zip(tokens, tags):
        if not tag or tag == "O":
            if cur_label:
                entities.append((" ".join(cur_tokens), cur_label))
                cur_label, cur_tokens = None, []
            continue

        parts = tag.split("-", 1)
        if len(parts) != 2:
            if cur_label:
                entities.append((" ".join(cur_tokens), cur_label))
                cur_label, cur_tokens = None, []
            continue

        bio, label = parts[0], parts[1]

        if bio == "B" or label != cur_label:
            if cur_label:
                entities.append((" ".join(cur_tokens), cur_label))
            cur_label, cur_tokens = label, [tok]
        else:  # I
            cur_tokens.append(tok)

    if cur_label:
        entities.append((" ".join(cur_tokens), cur_label))

    out = {"UNI": [], "TYPE": [], "KEYWORD": []}
    for text, label in entities:
        if label in out:
            out[label].append(text)
    return out
