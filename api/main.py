#!/usr/bin/env python3
"""FastAPI wrapper around the House Price Intelligence Assistant agent."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent import MODEL, MODEL_SOURCE, ask  # noqa: E402  (needs sys.path above)


# The Next.js/CRA dev server is reached under either spelling depending on how
# the browser resolves it, and CORS matches origins as exact strings.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app = FastAPI(
    title="House Price Intelligence Assistant",
    description="Ask questions about Sri Lankan house prices and market reports.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="A natural-language question about Sri Lankan house prices.",
        examples=["How have house prices in the Colombo district changed recently?"],
    )


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Does not check that the LLM provider is reachable.

    Reports the resolved model and where it came from. A stale HOUSE_AGENT_MODEL
    exported in the shell outranks .env and is otherwise invisible until the
    provider rejects it, so this is the fastest way to see what is really loaded.
    """
    return {"status": "ok", "model": MODEL, "model_source": MODEL_SOURCE}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    """Answer a question using the agent's prediction and retrieval tools."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank.")

    try:
        answer = ask(question)
    except SystemExit as exc:
        # agent._build_agent() raises SystemExit when the LLM provider package
        # or API key is missing. Uncaught, that would terminate the worker.
        raise HTTPException(
            status_code=503,
            detail=f"Agent is not configured: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed to answer: {type(exc).__name__}: {exc}",
        ) from exc

    return AskResponse(answer=answer)
