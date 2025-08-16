# -*- coding: utf-8 -*-
from retriever import TfidfRetriever

if __name__ == "__main__":
    rt = TfidfRetriever()
    docs = rt.load_corpus()
    if not docs:
        raise SystemExit("코퍼스가 비어 있습니다. data/document_chunks.json 또는 university/*/tables_cleaned.csv 를 준비하세요.")
    rt.build(docs)
    rt.save()
    print(f"TF-IDF 인덱스 생성 완료: 문서 {len(docs)}건")
