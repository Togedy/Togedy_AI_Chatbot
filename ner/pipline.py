from ner.model.predict import run_ner_model
from ner.rule.label_logic import resolve_labels
from ner.rule.label_store import update_label_store, get_label_store

def ner_pipeline(question: str, first: bool):
    model_labels = run_ner_model(question)
    
    if first:
        resolved = resolve_labels(model_labels, previous_labels=None)
        update_label_store(resolved)
    else:
        prev_labels = get_label_store()
        resolved = resolve_labels(model_labels, previous_labels=prev_labels)
        update_label_store(resolved)

    return resolved
