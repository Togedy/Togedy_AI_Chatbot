# Togedy AI Chatbot

대학 입시요강을 기반으로 질문에 답하는 Flask 기반 RAG 챗봇입니다. 질문에서 대학명, 전형, 핵심어를 추출한 뒤 대학별 모집요강에서 관련 페이지를 검색하고 LLM으로 답변을 생성합니다.

## 처리 흐름

1. KoBERT NER 모델로 `UNI`, `TYPE`, `KEYWORD` 추출
2. Gemini를 이용해 복수 개체와 검색 조건 보정
3. 대학·전형별 추출 문서에서 TF-IDF 기반 관련 페이지 검색
4. 검색 문맥을 바탕으로 OpenAI 모델이 최종 답변 생성
5. 답변과 함께 사용한 문서 및 페이지 정보 반환

## 디렉터리 구성

```text
.
├── main.py                    # Flask API 진입점
├── settings.py                # 환경변수와 프로젝트 경로 설정
├── generate_answers.py        # 단일/후속 질문 답변 파이프라인
├── extract_all.py             # NER 모델 통합
├── search_and_export.py       # 페이지 검색과 결과 구성
├── document_retrieval/        # TF-IDF 검색 모듈
├── KORBERT_NER_UNI/           # 대학명 NER 모델
├── KORBERT_NER_TYPE/          # 전형 NER 모델
├── KORBERT_NER_KEYWORD/       # 핵심어 NER 모델
├── university/                # 대학별 PDF, 텍스트, 표 데이터
├── llm/                       # LLM 클라이언트와 프롬프트
├── ocr/                       # PDF 텍스트·표 추출 도구
└── app/                       # 이전 파이프라인 코드(현재 진입점 아님)
```

## 로컬 실행

Python 3.11 환경을 권장합니다.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python main.py
```

`.env`에 최소 `GOOGLE_API_KEY`와 `OPENAI_API_KEY`를 설정해야 전체 답변 파이프라인이 동작합니다. 포트와 모델 등 나머지 설정은 `.env.example`을 참고하세요.

서버가 시작되면 다음 엔드포인트를 사용할 수 있습니다.

- `GET /health`: 서버 상태와 NER 모델 로딩 여부 확인
- `POST /answer`: 입시 질문 처리

NER 모델은 서버 시작 시가 아니라 첫 `/answer` 요청에서 한 번만 로딩됩니다. `/health`는 무거운 모델을 로딩하지 않습니다.

## API 예시

최초 질문:

```powershell
$body = @{
  first = $true
  question_1 = "연세대학교 수시 모집인원을 알려줘"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:5000/answer `
  -ContentType application/json -Body $body
```

후속 질문에서는 직전 질문과 서버가 반환한 `NER` 값을 함께 전달합니다.

```json
{
  "first": false,
  "question_1": "연세대학교 수시 모집인원을 알려줘",
  "question_2": "컴퓨터과학과만 알려줘",
  "NER": {
    "UNI": ["연세대학교"],
    "TYPE": ["수시"],
    "KEYWORD": ["모집인원"]
  }
}
```

## 개발 확인

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

모델 파일과 대학별 원본 데이터가 크므로 새 환경에서는 저장소 체크아웃과 최초 모델 로딩에 시간이 걸릴 수 있습니다.

## 관리 원칙

- 현재 운영 진입점은 루트의 `main.py`입니다.
- 파일 경로는 실행 위치가 아니라 저장소 루트를 기준으로 해석합니다.
- API 키와 개인 설정은 `.env`에만 두고 커밋하지 않습니다.
- 생성 결과와 캐시는 Git에 추가하지 않습니다.
- 입시 정보는 실제 서비스에 사용하기 전에 반드시 해당 대학의 최신 공식 모집요강과 대조합니다.
