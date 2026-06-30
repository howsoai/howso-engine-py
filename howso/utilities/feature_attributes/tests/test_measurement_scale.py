"""Tests for continuous measurement-scale inference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from howso.utilities import infer_feature_attributes


def measurement_scale_benchmark(
    seed: int = 42,
    n: int = 500,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Synthetic benchmark for data-only ratio vs. interval inference."""
    ground_truth = {
        "home_price_usd": "ratio",
        "annual_rainfall_inches": "ratio",
        "engine_displacement_liters": "ratio",
        "income_usd": "ratio",
        "distance_miles": "ratio",
        "weight_kg": "ratio",
        "conversion_rate": "ratio",
        "completion_fraction": "ratio",
        "temperature_c": "interval",
        "calendar_year": "interval",
        "iq_score": "interval",
        "standardized_score": "interval",
        "temperature_f": "interval",
        "elevation_m": "interval",
        "net_profit": "interval",
        "temperature_anomaly": "interval",
        "prediction_residual": "interval",
        "z_score": "interval",
        "centered_income": "interval",
        "embedding_component": "interval",
        "signed_anomaly_score": "interval",
    }

    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "home_price_usd": rng.lognormal(mean=12.7, sigma=0.35, size=n),
            "annual_rainfall_inches": rng.gamma(shape=8, scale=5, size=n),
            "engine_displacement_liters": rng.normal(loc=2.4, scale=0.6, size=n).clip(0.8, None),
            "income_usd": rng.lognormal(mean=10.5, sigma=0.7, size=n),
            "distance_miles": rng.exponential(scale=12, size=n),
            "weight_kg": np.clip(rng.normal(loc=80, scale=15, size=n), 1, None),
            "conversion_rate": rng.beta(a=2, b=8, size=n),
            "completion_fraction": rng.beta(a=5, b=2, size=n),
            "temperature_c": rng.normal(loc=20, scale=10, size=n),
            "calendar_year": rng.integers(1980, 2026, size=n),
            "iq_score": rng.normal(loc=100, scale=15, size=n),
            "standardized_score": rng.normal(loc=0, scale=1, size=n),
            "temperature_f": rng.normal(loc=68, scale=18, size=n),
            "elevation_m": rng.normal(loc=150, scale=300, size=n),
            "net_profit": rng.normal(loc=5000, scale=15000, size=n),
            "temperature_anomaly": rng.normal(loc=0, scale=2, size=n),
            "prediction_residual": rng.normal(loc=0, scale=8, size=n),
            "z_score": rng.normal(loc=0, scale=1, size=n),
            "centered_income": (
                rng.lognormal(mean=10.5, sigma=0.7, size=n)
                - np.exp(10.5)
            ),
            "embedding_component": rng.normal(0, 1, size=n),
            "signed_anomaly_score": rng.normal(0, 4, size=n),
        }
    )

    return df, ground_truth


def test_continuous_type_benchmark() -> None:
    df, ground_truth = measurement_scale_benchmark()
    features = infer_feature_attributes(df, enable_suggestions=False)

    ratio_correct = sum(
        features[feature_name]["continuous_type"] == expected
        for feature_name, expected in ground_truth.items()
        if expected == "ratio"
    )
    interval_correct = sum(
        features[feature_name]["continuous_type"] == expected
        for feature_name, expected in ground_truth.items()
        if expected == "interval"
    )

    assert ratio_correct == 6
    assert interval_correct == 13


def test_types_ratio_interval_override() -> None:
    df, _ = measurement_scale_benchmark()
    features = infer_feature_attributes(
        df,
        types={
            "income_usd": "ratio",
            "temperature_c": "interval",
            "ratio": ["distance_miles"],
        },
        enable_suggestions=False,
    )

    assert features["income_usd"]["type"] == "continuous"
    assert features["income_usd"]["continuous_type"] == "ratio"
    assert features["temperature_c"]["type"] == "continuous"
    assert features["temperature_c"]["continuous_type"] == "interval"
    assert features["distance_miles"]["type"] == "continuous"
    assert features["distance_miles"]["continuous_type"] == "ratio"


def test_non_continuous_features_get_nan_continuous_type() -> None:
    df, _ = measurement_scale_benchmark()
    df = df.assign(category=np.array(["a", "b", "c", "d", "e"] * (len(df) // 5)))
    features = infer_feature_attributes(df, enable_suggestions=False)

    assert features["category"]["type"] == "nominal"
    assert math.isnan(features["category"]["continuous_type"])
