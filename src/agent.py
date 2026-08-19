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


# Read this BEFORE load_dotenv(), which does not overwrite variables that are
# already set: a HOUSE_AGENT_MODEL exported in the shell silently outranks the
# one in .env. That precedence is conventional, but it is invisible when it
# bites - you edit .env, nothing changes, and the mismatch only surfaces much
# later as a 404 from the provider. Capturing it here lets MODEL_SOURCE below
# name where the value actually came from.
_SHELL_MODEL = os.environ.get("HOUSE_AGENT_MODEL")

load_dotenv()

# Groq is the provider this project ships with (langchain-groq in
# requirements.txt, GROQ_API_KEY in .env), so the default has to name a model
# Groq currently serves and that supports tool calling. Groq retires model IDs
# without notice - a decommissioned one fails at invoke time with a 404
# "model_not_found", not at startup. Check https://console.groq.com/docs/models
# if that happens. Override with e.g. HOUSE_AGENT_MODEL="openai:gpt-4o" after
# installing that provider's LangChain package.
DEFAULT_MODEL = "groq:openai/gpt-oss-120b"
MODEL = os.getenv("HOUSE_AGENT_MODEL", DEFAULT_MODEL)

if _SHELL_MODEL:
    MODEL_SOURCE = "shell environment (this overrides .env)"
elif os.getenv("HOUSE_AGENT_MODEL"):
    MODEL_SOURCE = ".env"
else:
    MODEL_SOURCE = "built-in default"

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
    # ImportError: the provider package is absent. ValueError: init_chat_model
    # rejected the string - an unrecognised provider prefix, or a bare model
    # name with no "provider:" prefix at all. Both are configuration mistakes
    # with the same fix, so they share one message.
    except (ImportError, ValueError) as exc:
        provider = MODEL.split(":", 1)[0] if ":" in MODEL else "<none>"
        raise SystemExit(
            f"Could not build the agent for HOUSE_AGENT_MODEL={MODEL!r} "
            f"(provider {provider!r}).\n"
            f"  - Install the provider:  pip install langchain-{provider}\n"
            f"  - Set its API key in .env (e.g. {provider.upper()}_API_KEY=...)\n"
            f"  - Or point HOUSE_AGENT_MODEL at a provider you already have, "
            f"in 'provider:model' form.\n"
            f"Original error: {type(exc).__name__}: {exc}"
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

    print("House Price Intelligence Assistant")
    print(f"model:  {MODEL}")
    print(f"source: {MODEL_SOURCE}")
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
