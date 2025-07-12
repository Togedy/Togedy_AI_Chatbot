import os
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

UNIV_ROOT = os.path.join("university")

def read_pages_from_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pages = re.split(r"==== Page \d+ ====", content)
    pages = [page.strip() for page in pages if page.strip()]
    return pages

def collect_all_documents():
    documents = []
    metadata = []

    for root, _, files in os.walk(UNIV_ROOT):
        for file in files:
            if file.endswith("_text.txt"):
                file_path = os.path.join(root, file)
                pages = read_pages_from_text_file(file_path)

                for page_idx, page_text in enumerate(pages, start=1):
                    documents.append(page_text)
                    metadata.append({
                        "university": os.path.basename(root),
                        "source": file,
                        "page": page_idx,
                        "text": page_text
                    })
    return documents, metadata

def build_index():
    print("→ 텍스트 수집 중...")
    documents, metadata = collect_all_documents()

    print(f"→ 총 페이지 수: {len(documents)}")

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    index_data = {
        "tfidf_matrix": tfidf_matrix,
        "vectorizer": vectorizer,
        "metadata": metadata
    }

    with open("document_retrieval/tfidf_index.pkl", "wb") as f:
        pickle.dump(index_data, f)

    print("✅ 인덱싱 완료: document_retrieval/tfidf_index.pkl")

if __name__ == "__main__":
    build_index()
