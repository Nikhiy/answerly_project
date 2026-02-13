from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import create_llm_chain, QAResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ======= Input model =======
class QuestionRequest(BaseModel):
    question: str
    url: str

# ======= Health check =======
@app.get("/")
async def working():
    return {"message": "backend is working"}

# ======= Main route =======
@app.post("/ask", response_model=QAResponse)
async def ask_question(request: QuestionRequest):
    try:
        response = create_llm_chain(
            question=request.question,
            url=request.url
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )
