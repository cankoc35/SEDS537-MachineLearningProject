"""Threshold-based hallucination detection baselines."""


def entropy_threshold_baseline(features, threshold: float):
    """Predict hallucination labels from entropy features."""
    raise NotImplementedError


def confidence_threshold_baseline(features, threshold: float):
    """Predict hallucination labels from confidence features."""
    raise NotImplementedError
