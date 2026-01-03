
---

# 🎓 Togedy AI Chatbot

대학 입시 요강을 기반으로 **자연어 질문에 맞춤형으로 응답하는 AI 챗봇**입니다.
OCR·NER·문서 검색·LLM을 결합하여, 복잡한 입시 요강을 **사람이 묻듯 질문하면 바로 답변**할 수 있도록 설계되었습니다.

---

## 📌 프로젝트 개요

* **프로젝트명**: Togedy AI Chatbot
* **목적**
  대학별 입시 요강(PDF, 표, 텍스트)을 구조화하고
  사용자의 자연어 질문에 대해 정확한 정보를 제공하는 질의응답 시스템 구현
* **대상 질문 예시**

  * “연세대 컴퓨터공학과 수시 전형 방법 알려줘”
  * “고려대 정시 모집 인원은 몇 명이야?”
  * “서강대 논술 전형 일정은 언제야?”

---

## 🧠 핵심 특징

* **KoBERT 기반 NER**

  * 대학명(UNI), 전형(TYPE), 핵심 키워드(KEYWORD) 자동 추출
* **질문 유형 자동 분기**

  * 답변 생성 / 문서 탐색 / 재질문 유도
* **TF-IDF 기반 문서 검색**

  * 대학별 요강 텍스트 및 표 데이터 검색
* **LLM 이중 구조**

  * Gemini: NER 및 탐색 보강
  * GPT-4o: 최종 자연어 응답 생성
* **OCR 기반 데이터 파이프라인**

  * 입시 요강 PDF → 텍스트 / 테이블 구조화

---

## 🔄 전체 시스템 흐름

1. 사용자 질문 입력
2. KoBERT NER로 **UNI / TYPE / KEYWORD** 추출
3. 키워드 조합에 따라 로직 분기

   * UNI 없음 + KEYWORD 있음 → 재질문
   * UNI 있음 + KEYWORD 있음 → 문서 탐색
   * KEYWORD 없음 → 일반 답변 생성
4. TF-IDF 기반 문서 검색
5. 검색 결과 + 질문을 LLM에 전달
6. 최종 자연어 답변 생성

---

## 📁 디렉토리 구조

```text
.
├─ main.py                  # 전체 파이프라인 실행
├─ answer.py                # 답변 생성 로직
├─ extract_all.py           # OCR 및 텍스트 추출
├─ search_and_export.py     # 문서 탐색 및 결과 처리
├─ document_retrieval/      # TF-IDF 기반 검색
│  ├─ build_tfidf.py
│  ├─ retriever.py
│  └─ table_pick.py
├─ KORBERT_NER_UNI/         # 대학명 NER
├─ KORBERT_NER_TYPE/        # 전형 NER
├─ KORBERT_NER_KEYWORD/     # 키워드 NER
├─ llm/                     # LLM 프롬프트 및 추론
├─ ocr/                     # OCR & 테이블 추출
├─ university/              # 대학별 요강 데이터
└─ app/                     # Flask API 서버
```

---

## 🧩 NER 모델 설명

* **모델**: `skt/kobert-base-v1`
* **태깅 방식**: BIO tagging
* **라벨 구성**

  * `UNI` : 대학명
  * `TYPE` : 전형 (수시, 정시, 논술 등)
  * `KEYWORD` : 질문 핵심 개념
* **활용 목적**

  * 질문 의도 파악
  * 문서 탐색 여부 및 대상 결정

---

## 🔍 문서 탐색 방식

* **기법**: TF-IDF + cosine similarity
* **데이터**

  * 대학별 `susi_text.txt`, `jungsi_text.txt`
  * 전형별 CSV 테이블
* **특징**

  * 대학 단위로 탐색 범위 제한
  * 표 데이터와 본문 텍스트 분리 처리

---

## 🤖 LLM 구성

| 역할          | 사용 모델  |
| ----------- | ------ |
| NER / 탐색 보강 | Gemini |
| 최종 답변 생성    | GPT-4o |

* 검색된 문서 내용을 컨텍스트로 활용하여 **환각을 최소화**
* 질문 유형에 따라 프롬프트 구조 분리

---

## ▶ 실행 방법

```bash
pip install -r requirements.txt
python main.py
```

또는 Flask API 실행

```bash
python app/main.py
```

---

## 📌 활용 목적

* 대학 입시 정보 접근성 향상
* 챗봇 기반 입시 상담 자동화
* RAG + NER 결합 구조 실험 및 연구

---

## 📎 참고 사항

* 본 프로젝트는 **깃허브 공개용**으로 정제되었습니다.
* 실제 입시 상담 시에는 반드시 공식 요강을 함께 확인하시기 바랍니다.

