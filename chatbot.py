
from ner.predict import extract_query_entities
from document_retrieval.retriever import search_documents
from table_searcher import search_table_for_query
from llm.prompt_builder import build_prompt
from llm.inference import ask_gpt


def chatbot_pipeline(question: str) -> str:
    """
    사용자 질문에 대해 전체 파이프라인을 실행하여 답변 생성

    Args:
        question (str): 사용자 자연어 질문

    Returns:
        str: GPT 응답
    """
    print(f"[사용자 질문] {question}")

    # Step 1. NER 추출
    query = extract_query_entities(question)
    if not query or not all(k in query for k in ["UNI", "TYPE", "KEYWORD"]):
        return "[오류] 질문에서 필요한 키워드(대학교/전형/주제)를 찾을 수 없습니다."

    print(f"[추출된 키워드] {query}")

    # Step 2. 문서 검색
    retrieved_docs = search_documents(query)
    if not retrieved_docs:
        return "[오류] 관련 문서를 찾을 수 없습니다."

    # Step 3. 표 검색
    table_data = search_table_for_query(query)

    # Step 4. 프롬프트 생성 (텍스트 + 표)
    prompt = build_prompt(query, retrieved_docs, table_data)

    # Step 5. GPT 호출
    response = ask_gpt(prompt)
    return response


if __name__ == "__main__":
    while True:
        user_input = input("\n질문 입력 ('exit' 입력 시 종료): ")
        if user_input.lower() == "exit":
            break

        answer = chatbot_pipeline(user_input)
        print("\n📌 GPT 응답:")
        print(answer)
