from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

ContinuousType = Literal["ratio", "interval"]
MeasurementScale = Literal["ratio", "interval", "binary"]

INTEGER_ATOL = 1e-8
MODERATE_RIGHT_SKEW = 0.5
STRONG_RIGHT_SKEW = 1.0
SYMMETRIC_SKEW_THRESHOLD = 0.5
SYMMETRIC_PENALTY_MIN_UNIQUE = 20


@dataclass(frozen=True)
class FeatureProfile:
    """
    Cached, data-only summary of one numeric feature.

    Rules use this object instead of repeatedly recomputing
    things such as n_unique, integer-like, contains_zero, etc.
    """

    x: pd.Series
    n: int
    n_unique: int
    min_value: float
    max_value: float
    std: float

    nonnegative: bool
    positive: bool
    contains_zero: bool
    contains_one: bool

    integer_like: bool
    binary_0_1: bool
    in_unit_interval: bool

    year_like: bool
    small_rating_scale: bool

    raw_skew: Optional[float]
    log_skew: Optional[float]
    log_reduces_skew: bool


def clean_numeric(x: pd.Series) -> pd.Series:
    """Convert a series to numeric and discard NaN / +/- infinity."""
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).dropna()


def is_integer_like(x: pd.Series, atol: float = INTEGER_ATOL) -> bool:
    """Return whether all observed values are effectively integers."""
    if x.empty:
        return False

    values = x.to_numpy(dtype=float)
    return bool(np.all(np.isclose(values, np.round(values), atol=atol, rtol=0.0)))


def is_year_like(x: pd.Series, integer_like: bool) -> bool:
    """
    Conservative calendar-year pattern detector.

    It is only a protective heuristic, not proof that a feature is a year.
    """
    if x.empty or not integer_like:
        return False

    min_value = float(x.min())
    max_value = float(x.max())

    return bool(
        1600 <= min_value <= 2500
        and 1600 <= max_value <= 2500
        and (max_value - min_value) <= 300
    )


def is_small_rating_scale(x: pd.Series, integer_like: bool) -> bool:
    """
    Detect likely bounded ordinal scales such as 0-5, 1-5, or 0-10.

    This deliberately treats these as evidence against automatic
    promotion to ratio.
    """
    if x.empty or not integer_like:
        return False

    return bool(
        x.nunique() <= 11
        and x.min() >= 0
        and x.max() <= 10
    )


def build_feature_profile(x: pd.Series) -> FeatureProfile:
    """Build all reusable properties once for one feature."""
    x = clean_numeric(x)

    if x.empty:
        return FeatureProfile(
            x=x,
            n=0,
            n_unique=0,
            min_value=np.nan,
            max_value=np.nan,
            std=np.nan,
            nonnegative=False,
            positive=False,
            contains_zero=False,
            contains_one=False,
            integer_like=False,
            binary_0_1=False,
            in_unit_interval=False,
            year_like=False,
            small_rating_scale=False,
            raw_skew=None,
            log_skew=None,
            log_reduces_skew=False,
        )

    integer_like = is_integer_like(x)

    nonnegative = bool((x >= 0).all())
    positive = bool((x > 0).all())
    contains_zero = bool((x == 0).any())
    contains_one = bool((x == 1).any())
    in_unit_interval = bool(((x >= 0) & (x <= 1)).all())

    unique_values = set(x.unique())
    binary_0_1 = bool(
        unique_values.issubset({0, 1})
        and x.nunique() == 2
    )

    year_like = is_year_like(x=x, integer_like=integer_like)
    small_rating_scale = is_small_rating_scale(x=x, integer_like=integer_like)

    raw_skew: Optional[float] = None
    log_skew: Optional[float] = None
    log_reduces_skew = False

    if positive and x.nunique() >= 3:
        raw_skew = float(x.skew())
        log_skew = float(np.log(x).skew())
        if np.isfinite(raw_skew) and np.isfinite(log_skew):
            log_reduces_skew = abs(log_skew) < abs(raw_skew)
        else:
            raw_skew = None
            log_skew = None

    return FeatureProfile(
        x=x,
        n=len(x),
        n_unique=int(x.nunique()),
        min_value=float(x.min()),
        max_value=float(x.max()),
        std=float(x.std()),
        nonnegative=nonnegative,
        positive=positive,
        contains_zero=contains_zero,
        contains_one=contains_one,
        integer_like=integer_like,
        binary_0_1=binary_0_1,
        in_unit_interval=in_unit_interval,
        year_like=year_like,
        small_rating_scale=small_rating_scale,
        raw_skew=raw_skew,
        log_skew=log_skew,
        log_reduces_skew=log_reduces_skew,
    )


@dataclass(frozen=True)
class Rule:
    """
    One independently configurable test.

    score_group:
        Which aggregate score receives this rule's points.
        Use ``"ratio"``, ``"binary"``, or ``None`` for diagnostics only.
    """

    name: str
    category: str
    score_group: Optional[str]
    score: float
    description: str
    predicate: Callable[[FeatureProfile], bool]


@dataclass(frozen=True)
class RuleResult:
    """Result for one rule evaluated on one feature."""

    name: str
    category: str
    score_group: Optional[str]
    passed: bool
    configured_score: float
    contribution: float
    description: str


def evaluate_rule(rule: Rule, profile: FeatureProfile) -> RuleResult:
    """Evaluate one rule and assign its score contribution."""
    passed = bool(rule.predicate(profile))

    return RuleResult(
        name=rule.name,
        category=rule.category,
        score_group=rule.score_group,
        passed=passed,
        configured_score=rule.score,
        contribution=rule.score if passed else 0.0,
        description=rule.description,
    )


def evaluate_rules(profile: FeatureProfile, rules: list[Rule]) -> list[RuleResult]:
    """Run every registered rule against one feature."""
    return [evaluate_rule(rule, profile) for rule in rules]


def aggregate_scores(rule_results: list[RuleResult]) -> dict[str, float]:
    """Sum contributions by score group."""
    scores: dict[str, float] = {}

    for result in rule_results:
        if result.score_group is None:
            continue

        scores[result.score_group] = (
            scores.get(result.score_group, 0.0)
            + result.contribution
        )

    return scores


def aggregate_category_scores(rule_results: list[RuleResult]) -> dict[str, float]:
    """Sum contributions by human-readable category."""
    category_scores: dict[str, float] = {}

    for result in rule_results:
        category_scores[result.category] = (
            category_scores.get(result.category, 0.0)
            + result.contribution
        )

    return category_scores


DEFAULT_RULES: list[Rule] = [
    Rule(
        name="binary_0_1",
        category="binary",
        score_group="binary",
        score=10.0,
        description="Feature contains exactly the observed values 0 and 1.",
        predicate=lambda p: p.binary_0_1,
    ),
    Rule(
        name="count_nonnegative",
        category="count_evidence",
        score_group="ratio",
        score=1.0,
        description="All values are nonnegative.",
        predicate=lambda p: p.nonnegative,
    ),
    Rule(
        name="count_integer_like",
        category="count_evidence",
        score_group="ratio",
        score=2.0,
        description="All values are integer-like.",
        predicate=lambda p: p.integer_like,
    ),
    Rule(
        name="count_contains_zero",
        category="count_evidence",
        score_group="ratio",
        score=2.0,
        description="An observed zero exists.",
        predicate=lambda p: p.contains_zero,
    ),
    Rule(
        name="count_enough_unique_values",
        category="count_evidence",
        score_group="ratio",
        score=1.0,
        description="Feature has at least four unique values.",
        predicate=lambda p: p.n_unique >= 4,
    ),
    Rule(
        name="proportion_bounded_0_to_1",
        category="proportion_evidence",
        score_group="ratio",
        score=2.0,
        description="All values fall within [0, 1].",
        predicate=lambda p: p.in_unit_interval,
    ),
    Rule(
        name="proportion_continuous_like",
        category="proportion_evidence",
        score_group="ratio",
        score=1.0,
        description="Values are not all integer-like.",
        predicate=lambda p: not p.integer_like,
    ),
    Rule(
        name="proportion_enough_unique_values",
        category="proportion_evidence",
        score_group="ratio",
        score=1.0,
        description="Feature has at least five unique values.",
        predicate=lambda p: p.n_unique >= 5,
    ),
    Rule(
        name="moderate_right_skew",
        category="shape_evidence",
        score_group="ratio",
        score=2.0,
        description="Strictly positive data with moderate right skew.",
        predicate=lambda p: (
            p.positive
            and p.raw_skew is not None
            and p.raw_skew >= MODERATE_RIGHT_SKEW
        ),
    ),
    Rule(
        name="strong_right_skew",
        category="shape_evidence",
        score_group="ratio",
        score=2.0,
        description="Strictly positive, strongly right-skewed data.",
        predicate=lambda p: (
            p.positive
            and p.raw_skew is not None
            and p.raw_skew >= STRONG_RIGHT_SKEW
        ),
    ),
    Rule(
        name="log_normalizes_skew",
        category="shape_evidence",
        score_group="ratio",
        score=2.0,
        description="Log transform materially reduces skew.",
        predicate=lambda p: (
            p.positive
            and p.raw_skew is not None
            and p.log_skew is not None
            and abs(p.log_skew) < abs(p.raw_skew) * 0.5
        ),
    ),
    Rule(
        name="has_negative_values",
        category="ratio_blocker",
        score_group="ratio",
        score=-10.0,
        description="Negative values are incompatible with raw ratio scale.",
        predicate=lambda p: not p.nonnegative,
    ),
    Rule(
        name="year_like_penalty",
        category="ratio_blocker",
        score_group="ratio",
        score=-8.0,
        description="Feature resembles a calendar-year range.",
        predicate=lambda p: p.year_like,
    ),
    Rule(
        name="small_rating_scale_penalty",
        category="ratio_blocker",
        score_group="ratio",
        score=-8.0,
        description="Feature resembles a small bounded ordinal rating scale.",
        predicate=lambda p: p.small_rating_scale,
    ),
    Rule(
        name="symmetric_continuous_penalty",
        category="ratio_blocker",
        score_group="ratio",
        score=-2.0,
        description="Continuous, many unique values, and nearly symmetric.",
        predicate=lambda p: (
            not p.integer_like
            and p.n_unique >= SYMMETRIC_PENALTY_MIN_UNIQUE
            and p.raw_skew is not None
            and abs(p.raw_skew) < SYMMETRIC_SKEW_THRESHOLD
        ),
    ),
    Rule(
        name="log_reduces_skew",
        category="transform",
        score_group=None,
        score=0.0,
        description=(
            "Log transform reduces absolute skewness. "
            "Useful for preprocessing, not measurement scale."
        ),
        predicate=lambda p: p.log_reduces_skew,
    ),
]


@dataclass(frozen=True)
class ScaleClassifierConfig:
    """All classification thresholds in one place."""

    binary_threshold: float = 10.0
    ratio_threshold: float = 5.0

    high_ratio_confidence_threshold: float = 8.0
    medium_ratio_confidence_threshold: float = 5.0


DEFAULT_CONFIG = ScaleClassifierConfig()


def infer_measurement_scale(
    x: pd.Series,
    rules: list[Rule] | None = None,
    config: ScaleClassifierConfig = DEFAULT_CONFIG,
) -> dict:
    """
    Classify a feature as ratio, interval, or binary.

    Default is interval. Binary wins when its evidence reaches
    ``binary_threshold``; ratio wins when aggregate ratio evidence reaches
    ``ratio_threshold``.
    """
    if rules is None:
        rules = DEFAULT_RULES

    profile = build_feature_profile(x)

    if profile.n == 0:
        return {
            "measurement_scale": "interval",
            "inferred_subtype": "unknown",
            "promote_to_ratio": False,
            "confidence": "low",
            "ratio_score": 0.0,
            "binary_score": 0.0,
            "category_scores": {},
            "rule_results": [],
            "reason": "no_valid_numeric_values",
        }

    rule_results = evaluate_rules(profile=profile, rules=rules)

    score_totals = aggregate_scores(rule_results)
    category_scores = aggregate_category_scores(rule_results)

    ratio_score = score_totals.get("ratio", 0.0)
    binary_score = score_totals.get("binary", 0.0)

    passed_names = [
        result.name
        for result in rule_results
        if result.passed
    ]

    if binary_score >= config.binary_threshold:
        return {
            "measurement_scale": "binary",
            "inferred_subtype": "binary_0_1",
            "promote_to_ratio": False,
            "confidence": "high",
            "ratio_score": ratio_score,
            "binary_score": binary_score,
            "category_scores": category_scores,
            "rule_results": rule_results,
            "reason": f"binary threshold met; passed={passed_names}",
        }

    if ratio_score >= config.ratio_threshold:
        count_score = category_scores.get("count_evidence", 0.0)
        proportion_score = category_scores.get("proportion_evidence", 0.0)

        if count_score > proportion_score:
            subtype = "count_like"
        elif proportion_score > count_score:
            subtype = "proportion_like"
        else:
            subtype = "mixed_ratio_evidence"

        if ratio_score >= config.high_ratio_confidence_threshold:
            confidence = "high"
        elif ratio_score >= config.medium_ratio_confidence_threshold:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "measurement_scale": "ratio",
            "inferred_subtype": subtype,
            "promote_to_ratio": True,
            "confidence": confidence,
            "ratio_score": ratio_score,
            "binary_score": binary_score,
            "category_scores": category_scores,
            "rule_results": rule_results,
            "reason": f"ratio threshold met; passed={passed_names}",
        }

    return {
        "measurement_scale": "interval",
        "inferred_subtype": "unknown",
        "promote_to_ratio": False,
        "confidence": "low",
        "ratio_score": ratio_score,
        "binary_score": binary_score,
        "category_scores": category_scores,
        "rule_results": rule_results,
        "reason": f"default interval; passed={passed_names}",
    }


def infer_continuous_type(x: pd.Series) -> ContinuousType:
    """
    Map a continuous numeric feature to ``ratio`` or ``interval``.

    Ambiguous data-only cases map to ``interval``. Binary 0/1 features also
    map to ``interval`` here; IFA uses ``type`` for nominal/continuous distinction.
    """
    result = infer_measurement_scale(x)

    if result["measurement_scale"] == "ratio":
        return "ratio"

    return "interval"


def build_results_table(
    df: pd.DataFrame,
    ground_truth: dict[str, str] | None = None,
    rules: list[Rule] | None = None,
    config: ScaleClassifierConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the rule engine on every column in ``df``.

    Returns a wide summary table and a long rule-results table.
    """
    if rules is None:
        rules = DEFAULT_RULES

    wide_rows = []
    long_rows = []

    for feature_name in df.columns:
        feature = df[feature_name]

        result = infer_measurement_scale(
            x=feature,
            rules=rules,
            config=config,
        )

        profile = build_feature_profile(feature)

        wide_row = {
            "feature": feature_name,
            "actual": (
                ground_truth.get(feature_name)
                if ground_truth is not None
                else None
            ),
            "predicted_scale": result["measurement_scale"],
            "confidence": result["confidence"],
            "predicted_subtype": result["inferred_subtype"],
            "promote_to_ratio": result["promote_to_ratio"],
            "ratio_score": result["ratio_score"],
            "binary_score": result["binary_score"],
            "n": profile.n,
            "n_unique": profile.n_unique,
            "min_value": profile.min_value,
            "max_value": profile.max_value,
            "reason": result["reason"],
        }

        for category, score in result["category_scores"].items():
            wide_row[f"category_score__{category}"] = score

        for rule_result in result["rule_results"]:
            wide_row[f"passed__{rule_result.name}"] = rule_result.passed
            wide_row[f"contribution__{rule_result.name}"] = rule_result.contribution

            long_rows.append(
                {
                    "feature": feature_name,
                    "actual": (
                        ground_truth.get(feature_name)
                        if ground_truth is not None
                        else None
                    ),
                    "predicted_scale": result["measurement_scale"],
                    "confidence": result["confidence"],
                    "rule_name": rule_result.name,
                    "category": rule_result.category,
                    "score_group": rule_result.score_group,
                    "passed": rule_result.passed,
                    "configured_score": rule_result.configured_score,
                    "contribution": rule_result.contribution,
                    "description": rule_result.description,
                }
            )

        if ground_truth is not None:
            wide_row["exact_match"] = (
                result["measurement_scale"]
                == ground_truth.get(feature_name)
            )

        wide_rows.append(wide_row)

    results_wide = pd.DataFrame(wide_rows)
    rule_results_long = pd.DataFrame(long_rows)

    preferred_columns = [
        "feature",
        "actual",
        "predicted_scale",
        "confidence",
        "predicted_subtype",
        "promote_to_ratio",
        "ratio_score",
        "binary_score",
        "exact_match",
        "reason",
    ]

    preferred_columns = [
        column
        for column in preferred_columns
        if column in results_wide.columns
    ]

    remaining_columns = [
        column
        for column in results_wide.columns
        if column not in preferred_columns
    ]

    results_wide = results_wide[preferred_columns + remaining_columns]

    return results_wide, rule_results_long
