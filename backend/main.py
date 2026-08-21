"""
TechNova Multi-Agent AI Analyst — FastAPI backend.
Ishga tushirish (lokal): uvicorn main:app --reload --port 8000
Render'da: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents import run_question, stream_question

app = FastAPI(title="TechNova Multi-Agent AI Analyst")

# Frontend (Vercel) bilan ulanish uchun CORS ochiq (xohlasang, faqat o'z domeningga cheklab qo'y)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    """Oddiy, streaming bo'lmagan endpoint — to'liq javobni bir martada qaytaradi."""
    result = run_question(req.question)
    return {
        "answer": result["answer"],
        "steps": result["steps"],
        "documents": result["documents"],
        "sql_result": result["sql_result"],
        "code_result": result["code_result"],
    }


@app.get("/ask-stream")
def ask_stream(question: str):
    """Server-Sent Events (SSE) orqali har bir agent qadamini jonli oqim sifatida yuboradi."""

    def event_generator():
        try:
            for node_name, node_output in stream_question(question):
                payload = {"node": node_name}
                if node_output.get("plan"):
                    payload["plan"] = node_output["plan"]
                if node_output.get("answer"):
                    payload["answer"] = node_output["answer"]
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            # Safety net: whatever fails, the frontend must still get a response
            # instead of hanging forever waiting for an event that never comes.
            error_payload = {
                "node": "error",
                "answer": f"Xatolik yuz berdi ({type(e).__name__}). Iltimos, savolni qayta yuboring.",
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
