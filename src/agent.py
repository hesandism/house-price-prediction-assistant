#!/usr/bin/env python3
"""A tool-calling agent that answers Sri Lankan house price questions."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent

if __package__ in (None, ""):  # allow `python src/agent.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import TOOLS


load_dotenv()

# Override with e.g. HOUSE_AGENT_MODEL="openai:gpt-4o" after installing that
# provider's LangChain package.
DEFAULT_MODEL = "anthropic:claude-opus-5"
MODEL = os.getenv("HOUSE_AGENT_MODEL", DEFAULT_MODEL)

SYSTEM_PROMPT = """\
You are the House Price Intelligence Assistant, an expert on the Sri Lankan \
residential property market. You answer questions using two tools, and you \
choose between them based on what the question actually asks for.

Use retrieve_docs for explanatory and factual questions: market trends, why \
prices moved, policy and interest rates, what happened in a given year or \
quarter, and anything drawn from the Central Bank of Sri Lanka's reports. \
Summarise what you find in your own words and cite the source filename.

Use predict_price for numeric estimation questions about a specific property: \
what a house of a given size, room count, district and age is worth. It needs \
all five inputs, so if the question is missing one, ask a brief follow-up \
rather than inventing a value.

Use both when a question needs an estimate plus reasoning - for example "what \
would a 3-bedroom house in Colombo cost, and is now a good time to buy?". \
Call predict_price for the number and retrieve_docs for the market context, \
then combine them into one answer.

Ground every factual claim in a tool result. If a tool reports that it cannot \
answer, say so plainly and explain why - never substitute a guessed price or \
invent a figure the tools did not return. Report prices in LKR."""


@lru_cache(maxsize=1)
def _build_agent():
    """Construct the tool-calling agent once and reuse it across questions."""
    try:
        return create_agent(model=MODEL, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    except ImportError as exc:
        provider = MODEL.split(":", 1)[0]
        raise SystemExit(
            f"The '{provider}' LangChain provider is not installed.\n"
            f"  pip install langchain-{provider}\n"
            f"Then set its API key (e.g. {provider.upper()}_API_KEY) in your .env, "
            f"or point HOUSE_AGENT_MODEL at a provider you already have.\n"
            f"Original error: {exc}"
        ) from exc


def _as_text(content) -> str:
    """Flatten message content, which is a list of blocks on some providers."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts).strip()
    return str(content)


def ask(question: str) -> str:
    """Ask the assistant a question and return its final answer as text."""
    if not question or not question.strip():
        return "Please ask a question."

    result = _build_agent().invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    messages = result.get("messages", [])
    if not messages:
        return "The agent returned no response."
    return _as_text(messages[-1].content)


EXAMPLE_QUESTIONS = [
    "How have house prices in the Colombo district changed recently?",
    "What would a 1800 sqft, 3 bedroom, 2 bathroom house in Colombo that is 10 years old cost?",
    "Estimate the price of a 5 year old 2200 sqft 4 bed 3 bath house in Kandy, and tell me whether the market is rising.",
]


def main() -> None:
    # Report text carries typographic characters (U+2010 hyphens, en dashes)
    # that a cp1252 Windows console cannot encode. Degrade instead of crashing.
    sys.stdout.reconfigure(errors="replace")

    print(f"House Price Intelligence Assistant  (model: {MODEL})")
    print("=" * 70)

    if "--examples" in sys.argv:
        for question in EXAMPLE_QUESTIONS:
            print(f"\n> {question}\n")
            print(ask(question))
            print("-" * 70)
        return

    print("Example questions you can try:")
    for question in EXAMPLE_QUESTIONS:
        print(f"  - {question}")
    print("\nType a question, or 'quit' to exit. Run with --examples to")
    print("execute all three examples non-interactively.\n")

    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if question.lower() in {"quit", "exit", "q"}:
            return
        if not question:
            continue

        try:
            print(f"\n{ask(question)}\n")
        except SystemExit:
            raise
        except Exception as exc:  # keep the REPL alive on a bad turn
            print(f"\n[error] {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
