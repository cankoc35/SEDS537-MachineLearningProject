"""Model loading and answer-token scoring helpers."""

from __future__ import annotations

from typing import Any

from src.generation.prompting import build_scoring_prompt


def load_tokenizer(model_name: str):
    """Load a Hugging Face tokenizer."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Tokenizer loading requires the 'transformers' package. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def select_device(device: str = "auto") -> str:
    """Select the best available torch device."""

    if device != "auto":
        return device

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Model scoring requires PyTorch. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str = "auto"):
    """Load a causal language model for answer-token scoring."""

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "Model scoring requires 'torch' and 'transformers'. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    resolved_device = select_device(device)
    dtype = "auto" if resolved_device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(resolved_device)
    model.eval()
    return model, resolved_device


def tokenize_for_scoring(example: dict[str, Any], tokenizer) -> dict[str, Any]:
    """Tokenize prompt and answer separately so answer tokens can be scored."""

    prompt_text = build_scoring_prompt(example)
    answer_text = str(example.get("answer", "")).strip()

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    return {
        "prompt_text": prompt_text,
        "answer_text": answer_text,
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "answer_tokens": tokenizer.convert_ids_to_tokens(answer_ids),
    }


def score_answer_tokens(
    example: dict[str, Any],
    tokenizer,
    model,
    device: str,
) -> list[dict[str, Any]]:
    """Score each answer token under the causal LM."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError("Model scoring requires PyTorch.") from exc

    tokenized = tokenize_for_scoring(example, tokenizer)
    prompt_ids = tokenized["prompt_ids"]
    answer_ids = tokenized["answer_ids"]

    if not answer_ids:
        return []

    full_ids = prompt_ids + answer_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits[0]
    answer_start = len(prompt_ids)
    token_scores: list[dict[str, Any]] = []

    for answer_index, token_id in enumerate(answer_ids):
        prediction_position = answer_start + answer_index - 1
        token_logprobs = functional.log_softmax(
            logits[prediction_position].float(),
            dim=-1,
        )
        token_probs = torch.exp(token_logprobs)
        token_logprob = token_logprobs[token_id].item()
        token_probability = float(token_probs[token_id].item())
        token_entropy = float(-(token_probs * token_logprobs).sum().item())
        token = tokenized["answer_tokens"][answer_index]

        token_scores.append(
            {
                "token_index": answer_index,
                "token_id": token_id,
                "token": token,
                "token_text": tokenizer.convert_tokens_to_string([token]),
                "probability": token_probability,
                "logprob": token_logprob,
                "entropy": token_entropy,
            }
        )

    return token_scores
