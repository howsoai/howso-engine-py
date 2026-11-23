"""Formatting of case data details sent by the engine as a dict of ``features`` and ``values``."""
from __future__ import annotations

from collections.abc import Mapping
from itertools import chain, pairwise
from typing import Any

import numpy as np
import pandas as pd

from howso.utilities import deserialize_cases, TokenizerProtocol


def format_case_table(
    detail: Mapping[str, list[Any]] | None,
    feature_attributes: Mapping[str, Any],
    *,
    tokenizer: TokenizerProtocol | None = None,
) -> pd.DataFrame:
    """Format a single DataFrame of case values, where each case maps to one row in the DataFrame."""
    if detail is None:
        return pd.DataFrame()
    features = detail.get("features")
    values = detail.get("values")
    if not features or not values:
        return pd.DataFrame(columns=features)
    # Treat null rows as a row of nulls so each row still maps to its case
    values = [[None] * len(features) if row is None else row for row in values]
    return deserialize_cases(
        data=values,
        columns=features,
        features=feature_attributes,
        tokenizer=tokenizer,
    )


def format_case_tables(
    detail: Mapping[str, list[Any]] | None,
    feature_attributes: Mapping[str, Any],
    *,
    tokenizer: TokenizerProtocol | None = None,
) -> list[pd.DataFrame]:
    """
    Format a list of per-case case-tables into a list of DataFrames in one formatting pass.

    All cases are concatenated into one DataFrame, formatted once, then sliced back into one
    DataFrame per case. This also guarantees consistent column dtypes across cases, which per-case
    inference does not.
    """
    if detail is None:
        return []
    features = detail.get("features") or []
    values = detail.get("values") or []
    values = [[] if v is None else v for v in values]  # treat nulls as empty table
    flat = list(chain.from_iterable(values))
    if not flat or not features:
        return [pd.DataFrame(columns=features) for _ in values]

    bounds = np.cumsum([0, *(len(v) for v in values)])
    df = deserialize_cases(
        data=flat,
        columns=features,
        features=feature_attributes,
        tokenizer=tokenizer,
    )
    return [
        df.iloc[start:stop].reset_index(drop=True) if stop > start else pd.DataFrame(columns=features)
        for start, stop in pairwise(bounds)
    ]
