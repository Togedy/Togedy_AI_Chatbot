import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# 인덱스 파일 위치
INDEX_PATH = os.path.join("document_retrieval", "tfidf_index.pkl")

def load_index():
    with open(INDEX_PATH, "rb") as f:
        index_data = pickle.load(f)
    return index_data

def build_query_string(query: dict) -> str:
    # Query 조합: UNI, TYPE, KEYWORD 순
    components = []
    if query.get("UNI"):
        components.append(query["UNI"])
    if query.get("TYPE"):
        components.append(query["TYPE"])
    if query.get("KEYWORD"):
        components.append(query["KEYWORD"])
    return " ".join(components)

def search_documents(query: dict, top_k=3):
    index_data = load_index()
    vectorizer = index_data["vectorizer"]
    tfidf_matrix = index_data["tfidf_matrix"]
    metadata = index_data["metadata"]

    query_text = build_query_string(query)
    query_vec = vectorizer.transform([query_text])

    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        meta = metadata[idx]
        results.append({
            "score": float(scores[idx]),
            "university": meta["university"],
            "source": meta["source"],
            "page": meta["page"],
            "text": meta["text"]
        })
    
    return results

if __name__ == "__main__":
    query = {
        "UNI": "yonsei",
        "TYPE": "susi",
        "KEYWORD": "논술"
    }
    results = search_documents(query)

    for r in results:
        print(f"[{r['university']}] p.{r['page']} | score: {r['score']:.4f}")
        print(r['text'][:300], "\n")
