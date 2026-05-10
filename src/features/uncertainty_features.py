"""Feature extraction for token uncertainty metrics."""

from __future__ import annotations

from statistics import mean
from typing import Any


def compute_uncertainty_features(
    token_scores: list[dict[str, Any]],
    low_confidence_threshold: float = 0.1,
    high_entropy_threshold: float = 2.0,
) -> dict[str, float | int]:
    """Summarize token-level uncertainty scores into response-level features."""

    if not token_scores:
        return {
            "answer_length_tokens": 0,
            "mean_token_probability": 0.0,
            "min_token_probability": 0.0,
            "max_token_probability": 0.0,
            "mean_token_logprob": 0.0,
            "min_token_logprob": 0.0,
            "max_token_logprob": 0.0,
            "sum_token_logprob": 0.0,
            "negative_mean_logprob": 0.0,
            "low_confidence_token_ratio": 0.0,
            "mean_token_entropy": 0.0,
            "min_token_entropy": 0.0,
            "max_token_entropy": 0.0,
            "sum_token_entropy": 0.0,
            "high_entropy_token_ratio": 0.0,
        }

    probabilities = [float(score["probability"]) for score in token_scores]
    logprobs = [float(score["logprob"]) for score in token_scores]
    entropies = [float(score.get("entropy", 0.0)) for score in token_scores]
    length = len(token_scores)
    mean_logprob = mean(logprobs)
    low_confidence_count = sum(
        probability < low_confidence_threshold for probability in probabilities
    )
    high_entropy_count = sum(entropy > high_entropy_threshold for entropy in entropies)

    return {
        "answer_length_tokens": length,
        "mean_token_probability": mean(probabilities),
        "min_token_probability": min(probabilities),
        "max_token_probability": max(probabilities),
        "mean_token_logprob": mean_logprob,
        "min_token_logprob": min(logprobs),
        "max_token_logprob": max(logprobs),
        "sum_token_logprob": sum(logprobs),
        "negative_mean_logprob": -mean_logprob,
        "low_confidence_token_ratio": low_confidence_count / length,
        "mean_token_entropy": mean(entropies),
        "min_token_entropy": min(entropies),
        "max_token_entropy": max(entropies),
        "sum_token_entropy": sum(entropies),
        "high_entropy_token_ratio": high_entropy_count / length,
    }


def build_uncertainty_features(token_stats):
    """Compute response-level uncertainty features."""
    return compute_uncertainty_features(token_stats)
