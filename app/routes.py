# app/routes.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 기존 run_pipeline 코드 포함되어 있음
# 아래 클래스는 위 run_pipeline() 아래에 이어 붙이면 됩니다

app = FastAPI()

# CORS 설정 (필요 시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시는 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청 포맷 정의
class QuestionRequest(BaseModel):
    question: str
    first: bool


# POST /ask → 질문 입력받아 응답
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    result = run_pipeline(request.question, request.first)
    return result


# 개발용 실행 코드
if __name__ == "__main__":
    uvicorn.run("app.routes:app", host="0.0.0.0", port=8000, reload=True)
