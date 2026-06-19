from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from services.code_analyzer import analyze, explain, convert

app = FastAPI(title="CodeSage API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str
    language: str


class ConvertRequest(BaseModel):
    code: str
    from_language: str
    to_language: str


@app.get("/")
def root():
    return {"status": "CodeSage API is running"}


@app.post("/analyze")
async def analyze_endpoint(req: CodeRequest):
    return analyze(req.code, req.language)


@app.post("/explain")
async def explain_endpoint(req: CodeRequest):
    return explain(req.code, req.language)


@app.post("/convert")
async def convert_endpoint(req: ConvertRequest):
    return convert(req.code, req.from_language, req.to_language)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
