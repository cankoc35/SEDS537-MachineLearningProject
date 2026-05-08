"""Prompt formatting helpers for answer scoring."""

from __future__ import annotations

from typing import Any


def build_scoring_prompt(example: dict[str, Any]) -> str:
    """Build the prompt prefix used before scoring the answer tokens."""

    context = str(example.get("context", "")).strip()
    question = str(example.get("prompt", "")).strip()

    parts = []
    if context:
        parts.append(f"Context:\n{context}")
    parts.append(f"Question:\n{question}")
    parts.append("Answer:\n")

    return "\n\n".join(parts)


def build_preview_text(example: dict[str, Any]) -> str:
    """Build human-readable text showing the exact prompt and candidate answer."""

    prompt = build_scoring_prompt(example)
    answer = str(example.get("answer", "")).strip()
    return f"{prompt}{answer}"
