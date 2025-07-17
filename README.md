아래는 현재까지 구현된 \*\*대학교 입시 챗봇 프로젝트(Togedy\_AI\_Chatbot)\*\*의 작업 내용을 기반으로 작성된 `README.md` 초안입니다.

---

```markdown
# Togedy_AI_Chatbot

> 대학교 입시 요강을 기반으로 사용자의 질문에 답변하는 AI 챗봇 프로젝트입니다.  
> OCR → NER → 문서 검색 → GPT 응답의 구조로 구현되며, 사용자 질의에 정확한 정보를 제공합니다.

---

## 🧠 프로젝트 개요

대학교별 수시/정시 모집 요강을 기반으로, 사용자가 자연어로 질문하면  
GPT를 통해 정확한 정보를 답변하는 챗봇 시스템을 구현합니다.

---

## 📁 디렉토리 구조

```

Togedy\_AI\_Chatbot/
├── OCR/                      # OCR 설정
├── app/                      # FastAPI 라우팅 (현재는 미사용)
├── data/                     # NER 라벨, 학습 데이터, TF-IDF 인덱스
├── document\_retrieval/       # 문서 검색 (TF-IDF 기반)
├── llm/                      # 프롬프트 생성 및 GPT 응답 생성
├── ner/                      # NER 모델 학습, 추론, 후처리
├── ocr/                      # 실제 텍스트 및 표 추출기
├── test\_cases/               # 흐름 시뮬레이션 테스트
├── university/               # 대학별 수시/정시 OCR 자료
└── requirements.txt

````

---

## ⚙️ 주요 구성 요소

### 1. OCR
- PDF 기반 모집요강 파일에서 표 및 텍스트 추출
- `ocr/text_extractor.py`, `ocr/table_extractor.py`로 처리
- 결과는 대학별 `*_text.txt`, `*_tables_cleaned.csv` 파일로 저장됨

### 2. NER (개체명 인식)
- 입력 질문에서 대학명(UNI), 전형명(TYPE), 키워드(KEYWORD) 추출
- BIO tagging 기반 `train.tsv`, `test.tsv` 구축
- `trainer.py`로 모델 학습 → 저장 후 `test.py`로 추론 가능

### 3. 문서 검색 (Retrieval)
- `document_chunks.json` + `tfidf_index.pkl` 기반으로 관련 문서 검색
- 추출된 NER 키워드를 기반으로 적절한 문서 스니펫을 반환

### 4. GPT 프롬프트 구성 및 응답
- `llm/prompt_builder.py`: GPT 입력 프롬프트 구성
- `llm/inference.py`: GPT 모델 API를 통해 최종 응답 생성
- 질문이 첫 질문인지, 후속 질문인지 구분하여 흐름 유지

---

## 🧪 테스트 예시

- `서울대 수시에서 모집인원이 궁금해요`
- `연세대 정시 논술전형 제출서류 알려줘`
- `성균관대 교과전형 일정이 어떻게 돼요?`

---

## 🔧 학습 방법

NER 모델 학습을 위해 아래 파일을 사용합니다:

- 학습: `data/train.tsv`
- 테스트: `data/test.tsv`

```bash
python ner/trainer.py
````

모델 저장 후 추론:

```bash
python ner/test.py --sentence "서울대 정시 모집인원 알려줘"
```

---

## 🗂 참고 데이터

* `university/` 내 각 학교별 수시 및 정시 PDF
* OCR 추출 텍스트/표 기반 문서로 변환되어 사용됨

---

## 🧩 향후 과제

* 서버 연동 (FastAPI 또는 Flask)
* 사용자 세션별 대화 흐름 유지
* Retrieval + GPT 답변 정밀도 개선
* 사용자 친화 UI 연동 (예: Streamlit, 웹앱 등)

---

## 👨‍💻 Contributors

* NER 모델, LLM 프롬프트 파이프라인: **\[작성자 이름]**
* OCR 및 문서 전처리: 완료
* 서버 통신/배포: (서버 담당자 예정)

```

---

필요에 따라 아래 항목을 추가하거나 변경하실 수 있습니다:

- `[작성자 이름]` → 님의 이름 또는 GitHub ID
- `학습 모델 파라미터`, `모델 성능`, `예시 응답 캡처`, `API 명세` 등

원하시면 마크다운 파일로도 `.md`로 생성해드릴 수 있습니다.
```
