"""
app.py

FastAPI backend for the Resume/JD Matcher. Mirrors the Express.js
route style you already know from NextStep -- routes, request/response
bodies, CORS -- just in Python.

Endpoints:
  POST /match
    - multipart form: resume_file (PDF/TXT) OR resume_text (string), jd_text (string)
    - returns: { match_score, prediction, missing_keywords }

Run: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import io

from predict import ResumeMatcher

app = FastAPI(title="Resume-JD Matcher API")

# Allow the React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher = ResumeMatcher()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pdfplumber not installed. Run: pip install pdfplumber"
        )
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


class MatchResponse(BaseModel):
    match_score: float
    prediction: str
    missing_keywords: list[str]


@app.post("/match", response_model=MatchResponse)
async def match(
    jd_text: str = Form(...),
    resume_text: Optional[str] = Form(None),
    resume_file: Optional[UploadFile] = File(None),
):
    if not resume_text and not resume_file:
        raise HTTPException(
            status_code=400,
            detail="Provide either resume_text or resume_file."
        )

    if resume_file is not None:
        raw = await resume_file.read()
        if resume_file.filename.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(raw)
        else:
            resume_text = raw.decode("utf-8", errors="ignore")

    if not resume_text or not resume_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the resume."
        )

    result = matcher.score(resume_text, jd_text)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
