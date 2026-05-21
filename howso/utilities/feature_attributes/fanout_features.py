"""
Infer "fanout features" in denormalized DataFrames.

When tabular data is built by joining a tree of relational tables
(e.g. ``Fleet JOIN Car JOIN CarParts``), each parent-table row gets duplicated
across many child rows. Without correction, HowsoEngine treats every duplicate
as an independent observation and over-weights the parent-table attributes.
The fanout-feature configuration tells the engine which columns are constant
when a given key is fixed, so those columns are counted once per distinct key
value rather than once per row.

This module infers that configuration automatically from a (possibly chunked
and streaming) denormalized frame: it discovers the join-fanout hierarchy and
emits a ``{chosen_key: [fanned_out_columns]}`` dict -- innermost-first -- ready
to feed into Howso's ``FeatureAttributes``.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pprint import pprint
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from howso.connectors import AbstractData, DataFrameData
from howso.utilities import infer_feature_attributes

__all__ = ["infer_fanout_feature_config"]


# Recursive type for the public nested fanout-tree return shape.
# Each entry's value is another NestedMapChain (possibly empty for a leaf).
NestedMapChain: TypeAlias = dict[str, "NestedMapChain"]


@dataclass
class _FanoutLevel:
    """One level of the recovered join-fanout hierarchy.

    A level corresponds to one source table in the original relational schema.
    All ``candidate_keys`` at a level are functionally interchangeable join
    keys (typically the source table's PK plus any FK from a child table that
    refers to it).

    Attributes
    ----------
    candidate_keys : Sequence[str]
        Interchangeable key columns at this level -- pinning any one of them
        pins the same set of fanned-out columns.
    fanned_out_columns : Sequence[str]
        Every column functionally determined by this level's key, including
        the key itself and any columns inherited from coarser-grained
        (ancestor) levels of the join.
    distinct_key_values : int
        Number of distinct values the key takes in the denormalized frame --
        equivalently, the row count of the original source table at this
        level of the join.
    """

    candidate_keys: Sequence[str]
    fanned_out_columns: Sequence[str]
    distinct_key_values: int


class _StreamingFanoutInferrer:
    """Incremental inferrer for HowsoEngine fanout-feature configuration.

    Consumes a chunked denormalized data frame (rows joined from a tree of
    relational tables) and recovers the join-fanout hierarchy without holding
    the whole frame in memory upfront. The recovered hierarchy is exactly what
    Howso Engine's fanout-feature configuration needs: for each level, which
    key gates which set of constant-per-key columns.

    Hand DataFrame chunks to :meth:`process_chunk` as they arrive. Every
    ``rebatch_every_num_rows`` new rows the accumulator re-runs batch inference
    on the data seen so far and tracks convergence: the recovered fanout chain
    must be identical across ``stable_runs_required`` consecutive runs **and**
    every chosen non-innermost key must have gone ``saturation_idle_rows`` rows
    without any new distinct value. Stop once :attr:`converged` is ``True`` or
    ``max_rows`` has been exceeded.

    Parameters
    ----------
    features : Mapping[str, Mapping[str, Any]]
        Feature mapping in Howso's ``infer_feature_attributes`` format --
        ``{column_name: {"type": "nominal" | "continuous", ...}, ...}``.
        Columns absent from the mapping are ignored entirely.
    fanout_key_card_floor : float | None, optional
        Minimum cardinality (as a fraction of total rows) a nominal column
        must reach to be considered a fanout-key candidate. ``None`` (default)
        resolves to ``1 / max_rows``, which yields an effective floor of 2
        distinct values for any reasonable ``max_rows`` -- i.e. every
        non-constant nominal column is a candidate. Pass a larger explicit
        value (e.g. ``0.001``) to filter out very-low-cardinality columns
        like Boolean flags on huge tables.
    rebatch_every_num_rows : int, optional
        Re-run batch inference once at least this many new rows have been fed
        since the last run. Row-based (not chunk-based) so behavior is stable
        across variable chunk sizes. Default ``500_000``.
    stable_runs_required : int, optional
        Number of consecutive identical fanout chains required to declare
        convergence. Default ``2``.
    saturation_idle_rows : int, optional
        A non-innermost fanout key is considered saturated once at least this
        many rows have been seen since it last gained a new distinct value.
        Default ``250_000``.
    max_rows : int, optional
        Maximum number of rows to accumulate before forcing a stop, regardless
        of convergence state. Default ``10_000_000``.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        features: Mapping[str, Mapping[str, Any]],
        *,
        fanout_key_card_floor: float | None = None,
        rebatch_every_num_rows: int = 500_000,
        stable_runs_required: int = 2,
        saturation_idle_rows: int = 250_000,
        max_rows: int = 10_000_000,
    ) -> None:
        self.features: Mapping[str, Mapping[str, Any]] = features
        # Default the cardinality floor to 1/max_rows so the effective floor
        # is the minimum meaningful value (>=2 via the max(2, ...) guard
        # inside discover_fanout_hierarchy). Coupled to max_rows so changing
        # one without the other still gives sensible behavior; pass an
        # explicit float to override (e.g. 0.001 to filter low-card columns).
        self.fanout_key_card_floor: float = (
            1.0 / max_rows if fanout_key_card_floor is None else fanout_key_card_floor
        )
        self.rebatch_every_num_rows: int = rebatch_every_num_rows
        self.stable_runs_required: int = stable_runs_required
        self.saturation_idle_rows: int = saturation_idle_rows
        self.max_rows: int = max_rows

        self.cols: list[str] = list(features.keys())
        self._buffers: dict[str, list[np.ndarray]] = {c: [] for c in self.cols}
        # Distinct-value tracking only for nominals -- continuous columns
        # can't be fanout keys.
        self._distinct: dict[str, set[Any]] = {
            f: set() for f, a in features.items() if a["type"] == "nominal"
        }
        # Value is the rows_seen count at which the column's distinct count last grew.
        self._distinct_last_grew: dict[str, int] = {c: 0 for c in self._distinct}

        self.chunks_seen: int = 0
        self.rows_seen: int = 0
        self._rows_at_last_rebatch: int = 0
        self.last_chain: dict[str, list[str]] | None = None
        self.stable_runs: int = 0
        self.converged: bool = False

    @property
    def should_rebatch(self) -> bool:
        """``True`` when at least ``rebatch_every_num_rows`` new rows have been fed since the last run."""
        return (
            self.rows_seen > 0
            and (self.rows_seen - self._rows_at_last_rebatch) >= self.rebatch_every_num_rows
        )

    @property
    def has_pending_rows(self) -> bool:
        """``True`` if rows have been fed since the last :meth:`rebatch` call.

        Useful for the final "drain" check after a chunk loop ends, so the
        caller can run inference one more time on rows that arrived after
        the last scheduled run.
        """
        return self.rows_seen > self._rows_at_last_rebatch

    def process_chunk(self, chunk: pd.DataFrame) -> None:
        """Append a chunk's columns to the accumulator and update key-saturation state.

        Object-dtype columns are scanned for unhashable values (lists, dicts,
        sets, numpy arrays, etc.); any such values are replaced with their
        ``repr()`` -- a deterministic, distinct, hashable surrogate. The
        algorithm only needs a stable unique-ish representation of each value,
        never the value itself, so this preserves correctness.

        Parameters
        ----------
        chunk : pandas.DataFrame
            A chunk of rows from the denormalized frame. Must contain every
            column listed in :attr:`features`; extra columns are tolerated
            and ignored.

        Raises
        ------
        ValueError
            If the chunk is missing any columns listed in :attr:`features`.
        """
        # Bind hot attributes to locals -- per-iteration `self.` lookups add up
        # at ~100 cols x many chunks. Net cost: free; net gain: a few percent.
        cols = self.cols
        buffers = self._buffers
        distinct = self._distinct
        distinct_last_grew = self._distinct_last_grew
        to_hashable = self._to_hashable

        missing: list[str] = [c for c in cols if c not in chunk.columns]
        if missing:
            raise ValueError(f"chunk missing columns: {missing}")
        self.chunks_seen += 1
        self.rows_seen += len(chunk)
        rows_seen = self.rows_seen
        for c in cols:
            arr: np.ndarray = to_hashable(chunk[c].to_numpy())
            buffers[c].append(arr)
            if c in distinct:
                bucket = distinct[c]
                before: int = len(bucket)
                # set.update on tolist() of the raw array is faster than the
                # old np.unique-then-tolist on object arrays: np.unique sorts
                # by Python comparison (slow on strings/mixed dtypes), and the
                # set already dedupes on the way in.
                bucket.update(arr.tolist())
                if len(bucket) > before:
                    distinct_last_grew[c] = rows_seen

    def rebatch(self) -> dict[str, list[str]]:
        """Re-run batch inference on the accumulated data and update convergence state.

        Returns
        -------
        dict[str, list[str]]
            The current fanout chain (innermost key first), of the form
            ``{chosen_key: [columns_fanned_out_at_this_level], ...}``.
        """
        df: pd.DataFrame = pd.DataFrame(
            {c: np.concatenate(self._buffers[c]) for c in self.cols}
        )
        levels: list[_FanoutLevel] = self.discover_fanout_hierarchy(
            df, self.features, fanout_key_card_floor=self.fanout_key_card_floor,
        )
        chain: dict[str, list[str]] = self.to_fanout_chain(levels)
        if chain == self.last_chain and self._keys_saturated(chain):
            self.stable_runs += 1
        else:
            self.stable_runs = 0
        self.last_chain = chain
        if self.stable_runs >= self.stable_runs_required:
            self.converged = True
        self._rows_at_last_rebatch = self.rows_seen
        return chain

    def _keys_saturated(self, chain: Iterable[str]) -> bool:
        """Check whether every non-innermost fanout key has stopped gaining distinct values.

        The innermost key is exempt: in typical row-keyed joins it's
        row-unique by construction (one value per row) and so introduces new
        distinct values with every row forever -- it can never saturate.
        Chain-equality across runs is the only convergence signal we have
        for the innermost level.

        Parameters
        ----------
        chain : Iterable[str]
            Chosen fanout-key column names in innermost-first order. Only
            iterated; order matters (the first element is the innermost and
            is exempted from this check). A fanout-chain dict can be passed
            directly since iterating a mapping yields its keys.

        Returns
        -------
        bool
            ``True`` iff every non-innermost key has gone at least
            ``saturation_idle_rows`` rows without any new distinct value.
        """
        non_innermost_keys: list[str] = list(chain)[1:]  # chain is innermost-first
        return all(
            self.rows_seen - self._distinct_last_grew.get(key, 0)
            >= self.saturation_idle_rows
            for key in non_innermost_keys
        )

    def should_stop(self) -> bool:
        """Indicate whether the caller should stop feeding more chunks.

        Returns
        -------
        bool
            ``True`` once either convergence has been declared or the
            ``max_rows`` budget has been reached.
        """
        return self.converged or self.rows_seen >= self.max_rows

    @staticmethod
    def _fd_exact(
        kc: np.ndarray,
        xc: np.ndarray,
        k_card: int,
        x_card: int,
    ) -> bool:
        """Test whether ``k`` functionally determines ``x`` exactly.

        Implements ``k -> x  iff  #unique(combine(k, x)) == #unique(k)`` using
        a 64-bit code combine when ``k_card * x_card`` fits in 62 bits, and a
        2D-stack fallback otherwise. Used internally by the fanout-hierarchy
        inference to test whether one column's values are constant within
        groups of another's.

        Parameters
        ----------
        kc : numpy.ndarray
            Integer codes for column ``k`` (one entry per row).
        xc : numpy.ndarray
            Integer codes for column ``x`` (one entry per row).
        k_card : int
            Number of distinct values in ``kc``.
        x_card : int
            Number of distinct values in ``xc``.

        Returns
        -------
        bool
            ``True`` iff ``k -> x`` holds exactly across the supplied rows.
        """
        if x_card > k_card:
            return False
        if x_card == 1:
            return True
        if k_card * x_card < (1 << 62):
            combined: np.ndarray = kc * x_card + xc
            return int(np.unique(combined).size) == k_card
        stack: np.ndarray = np.empty((kc.size, 2), dtype=np.int64)
        stack[:, 0] = kc
        stack[:, 1] = xc
        return int(np.unique(stack, axis=0).shape[0]) == k_card

    @staticmethod
    def _to_hashable(arr: np.ndarray) -> np.ndarray:
        """Replace unhashable values in an object array with their ``repr()``.

        The algorithm only needs each distinct value to map to a stable,
        distinct surrogate -- it never recovers the original value. ``repr()``
        gives us that for the typical unhashable cases (lists, dicts, sets,
        numpy arrays) since their reprs are deterministic and distinguish
        distinct contents.

        Pass-through for non-object arrays (numeric dtypes are always
        hashable). For object arrays, walks every element; hashable values
        pass through unchanged, unhashable ones are replaced. The array is
        only copied if at least one unhashable value is found.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array, potentially containing unhashable Python objects.

        Returns
        -------
        numpy.ndarray
            Same shape as ``arr``; every element is hashable. Returns ``arr``
            unchanged when no normalization is needed.
        """
        if arr.dtype != object or len(arr) == 0:
            return arr
        out: np.ndarray | None = None
        for j, v in enumerate(arr):
            if v is None:
                continue
            try:
                hash(v)
            except TypeError:
                if out is None:
                    out = arr.copy()
                out[j] = repr(v)
        return out if out is not None else arr

    @staticmethod
    def discover_fanout_hierarchy(
        df: pd.DataFrame,
        features: Mapping[str, Mapping[str, Any]],
        *,
        sample_size: int = 50_000,
        fanout_key_card_floor: float = 0.0,
        rng_seed: int = 0,
        verbose: bool = False,
    ) -> list[_FanoutLevel]:
        """Recover the join-fanout hierarchy of a denormalized DataFrame.

        Returns ``_FanoutLevel`` records ordered outermost (coarsest grain,
        smallest fanout set) to innermost (finest grain, largest fanout set).
        Each level corresponds to one source table in the original join.

        Parameters
        ----------
        df : pandas.DataFrame
            Wide denormalized frame produced by joining a strict tree of
            relational tables on FK->PK.
        features : Mapping[str, Mapping[str, Any]]
            Feature mapping in Howso's ``infer_feature_attributes`` format --
            ``{column_name: {"type": "nominal" | "continuous", ...}}``.
            Columns absent from the mapping are ignored entirely.
        sample_size : int, optional
            Row sample size used as a cheap pre-filter to reject obvious
            non-FDs before paying the cost of a full-data check. Default
            ``50_000``.
        fanout_key_card_floor : float, optional
            Minimum nominal-column cardinality (as a fraction of ``len(df)``)
            required for a column to be considered a fanout-key candidate.
            Default ``0.0`` (effective floor of 2 via the ``max(2, ...)``
            guard below -- every non-constant nominal is a candidate).
        rng_seed : int, optional
            Seed for the sample-selection RNG (deterministic). Default ``0``.
        verbose : bool, optional
            If ``True``, print a one-line summary of how many candidate keys
            pass the cardinality floor. Default ``False``.

        Returns
        -------
        list[_FanoutLevel]
            One ``_FanoutLevel`` per recovered level, outermost first.

        Raises
        ------
        ValueError
            If any feature has a ``type`` other than ``"nominal"`` or
            ``"continuous"``, or references a column not present in ``df``.
        """
        valid_types: set[str] = {"nominal", "continuous"}
        invalid: dict[str, Mapping[str, Any]] = {
            c: t for c, t in features.items() if t["type"] not in valid_types
        }
        if invalid:
            raise ValueError(
                f"feature 'type' values must be 'nominal' or 'continuous'; "
                f"got invalid entries: {invalid}"
            )
        missing: list[str] = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"features references columns not in df: {missing}")

        nominal_cols: list[str] = [
            c for c, t in features.items() if t["type"] == "nominal"
        ]
        continuous_cols: list[str] = [
            c for c, t in features.items() if t["type"] == "continuous"
        ]
        all_cols: list[str] = nominal_cols + continuous_cols

        n: int = len(df)
        rng: np.random.Generator = np.random.default_rng(rng_seed)

        codes: dict[str, np.ndarray] = {}
        card: dict[str, int] = {}
        for c in all_cols:
            arr, uniq = pd.factorize(df[c], use_na_sentinel=False)
            codes[c] = arr.astype(np.int64, copy=False)
            card[c] = len(uniq)

        sample: np.ndarray = (
            rng.choice(n, sample_size, replace=False)
            if n > sample_size
            else np.arange(n)
        )

        # Precompute per-column sample slices and their cardinalities once.
        # `determines()` is called O(k * m) times per rebatch and previously
        # re-derived these from scratch per call -- a wasted np.unique on the
        # sample for every pair.
        codes_sample: dict[str, np.ndarray] = {c: codes[c][sample] for c in all_cols}
        card_sample: dict[str, int] = {
            c: int(np.unique(codes_sample[c]).size) for c in all_cols
        }

        def determines(key: str, col: str) -> bool:
            """``True`` iff ``key`` functionally determines ``col`` across all rows."""
            if card[col] > card[key]:
                return False
            # Cheap sample reject; uses sample-local cardinalities (precomputed).
            if not _StreamingFanoutInferrer._fd_exact(
                codes_sample[key],
                codes_sample[col],
                card_sample[key],
                card_sample[col],
            ):
                return False
            return _StreamingFanoutInferrer._fd_exact(
                codes[key], codes[col], card[key], card[col],
            )

        floor: int = max(2, int(fanout_key_card_floor * n))
        # Ascending cardinality so smaller-key fanout sets are available for transitive inheritance.
        candidate_keys: list[str] = sorted(
            (c for c in nominal_cols if card[c] >= floor),
            key=lambda c: card[c],
        )
        if verbose:
            print(
                f"[fanout_inference] {len(candidate_keys)} of {len(nominal_cols)} "
                f"nominal cols clear card floor ({floor})"
            )

        fanned_per_key: dict[str, frozenset[str]] = {}
        for key in candidate_keys:
            fanned: set[str] = {key}
            # Transitive shortcut: if key -> prior_key and Det(prior_key) is known, inherit it.
            for prior_key, prior_fanned in fanned_per_key.items():
                if prior_key in fanned or card[prior_key] > card[key]:
                    continue
                if determines(key, prior_key):
                    fanned |= prior_fanned
            for col in all_cols:
                if col in fanned or col == key:
                    continue
                if determines(key, col):
                    fanned.add(col)
            fanned_per_key[key] = frozenset(fanned)

        keys_by_fanout: dict[frozenset[str], list[str]] = defaultdict(list)
        for key, fanout_set in fanned_per_key.items():
            if len(fanout_set) > 1:  # singletons are non-key columns
                keys_by_fanout[fanout_set].append(key)

        levels: list[_FanoutLevel] = [
            _FanoutLevel(
                candidate_keys=sorted(keys),
                fanned_out_columns=sorted(fanout_set),
                distinct_key_values=card[keys[0]],
            )
            for fanout_set, keys in keys_by_fanout.items()
        ]
        levels.sort(key=lambda lvl: len(lvl.fanned_out_columns))

        for i in range(len(levels) - 1):
            if not set(levels[i].fanned_out_columns).issubset(
                levels[i + 1].fanned_out_columns
            ):
                import warnings

                warnings.warn(
                    f"Levels {i} and {i+1} not in a strict subset chain -- "
                    f"strict-tree assumption may be violated"
                )

        return levels

    @staticmethod
    def to_fanout_chain(levels: Sequence[_FanoutLevel]) -> dict[str, list[str]]:
        """Collapse recovered levels into a ``{chosen_key: [fanned_out_cols]}`` chain.

        Picks the first candidate key at each level (any candidate is
        equivalent since they're functionally interchangeable). The value at
        each entry is the set of columns introduced *at* that level -- i.e.
        ``fanned_out_columns(level) - fanned_out_columns(parent level) -
        {chosen_key}``, sorted alphabetically. The dict is ordered innermost
        first so iterating it walks the join chain from row-grain outward.

        This is the shape HowsoEngine's fanout-feature configuration expects:
        each entry says "when ``chosen_key`` is fixed, these columns are
        constant -- treat them as one observation per distinct key value."

        Parameters
        ----------
        levels : Sequence[_FanoutLevel]
            Recovered levels in outermost-first order (as produced by
            :meth:`discover_fanout_hierarchy`). Needs indexed access and a
            length.

        Returns
        -------
        dict[str, list[str]]
            ``{chosen_key: [columns_fanned_out_at_this_level]}``, innermost
            level first.
        """
        chain: dict[str, list[str]] = {}
        for i in reversed(range(len(levels))):
            level: _FanoutLevel = levels[i]
            chosen_key: str = level.candidate_keys[0]
            ancestor_fanout: set[str] = (
                set(levels[i - 1].fanned_out_columns) if i > 0 else set()
            )
            owned_cols: list[str] = sorted(
                set(level.fanned_out_columns) - ancestor_fanout - {chosen_key}
            )
            chain[chosen_key] = owned_cols
        return chain

    @staticmethod
    def to_nested_chain(
        flat_chain: Mapping[str, Sequence[str]],
    ) -> NestedMapChain:
        """Render a flat innermost-first chain as a nested join-tree dict.

        The flat chain has one top-level entry per level; the nested form has
        one top-level entry (the row-grain / innermost key), with outer-level
        chosen keys appearing as nested branches inside their child level.
        Every node is a ``dict``; empty ``{}`` denotes a leaf (an owned
        column with no further fanout below it).

        For a flat chain like::

            {'inner_key': ['col_a', 'col_b'],
             'outer_key': ['outer_attr']}

        this returns::

            {'inner_key': {
                'col_a': {},
                'col_b': {},
                'outer_key': {
                    'outer_attr': {},
                },
            }}

        Parameters
        ----------
        flat_chain : Mapping[str, Sequence[str]]
            ``{chosen_key: [owned_cols]}``, innermost first, as produced by
            :meth:`to_fanout_chain`.

        Returns
        -------
        NestedChain
            Nested join-tree, rooted at the innermost fanout key. Each value
            is itself a (possibly empty) ``NestedChain``; empty dict means
            leaf. Returns ``{}`` if the input is empty.
        """
        keys: list[str] = list(flat_chain)  # innermost first
        if not keys:
            return {}
        nested: NestedMapChain = {}
        for key in reversed(keys):
            node: NestedMapChain = {col: {} for col in flat_chain[key]}
            if nested:
                # Splice the previously-built (outer) subtree in as a branch
                # at this (more-inner) level.
                node.update(nested)
            nested = {key: node}
        return nested


def infer_fanout_feature_config(
    features: Mapping[str, Any],
    data: AbstractData,
    *,
    chunk_size: int | None = None,
    fanout_key_card_floor: float | None = None,
    max_rows: int = 10_000_000,
    rebatch_every_num_rows: int = 500_000,
    saturation_idle_rows: int = 250_000,
    stable_runs_required: int = 2,
    verbose: bool = False,
) -> NestedMapChain:
    """Stream chunks from ``data`` and infer the fanout-feature configuration.

    Drives a :class:`_StreamingFanoutInferrer` over the connector's chunk
    iterator: processes each chunk, periodically re-runs batch inference, and
    stops once convergence is declared or ``max_rows`` is hit. Returns the
    recovered join-fanout structure as a nested dict (the join tree rooted at
    the row-grain key) -- the structure HowsoEngine consumes to avoid
    over-representing parent-table columns in a denormalized frame.

    Parameters
    ----------
    data : AbstractData
        Howso data connector exposing ``yield_chunk()`` -- iterates DataFrames.
    features : Mapping[str, Any]
        Feature mapping in Howso's ``infer_feature_attributes`` format.
    chunk_size : int | None, optional
        Chunk size to pass to ``data.yield_chunk()``. ``None`` (default)
        defers to the connector's own default chunk size.
    fanout_key_card_floor : float | None, optional
        Minimum nominal-column cardinality (fraction of accumulated rows) to
        be considered a fanout-key candidate. ``None`` (default) resolves to
        ``1 / max_rows`` -- effective floor of 2 distinct values, i.e. every
        non-constant nominal is a candidate. Override with an explicit value
        (e.g. ``0.001``) to filter very-low-cardinality columns out.
    max_rows : int, optional
        Stop after this many accumulated rows even if convergence has not
        been declared. Default ``10_000_000``.
    rebatch_every_num_rows : int, optional
        Re-run batch inference once this many new rows have accumulated
        since the last run. Default ``500_000``.
    saturation_idle_rows : int, optional
        A non-innermost fanout key is considered saturated once this many
        rows have been seen without it gaining a new distinct value.
        Default ``250_000``.
    stable_runs_required : int, optional
        Number of consecutive identical chains required to declare
        convergence. Default ``2``.
    verbose : bool, optional
        If ``True``, print per-discovery progress lines and a final summary.
        Default ``True``.

    Returns
    -------
    NestedMapChain
        Nested join-tree rooted at the innermost (row-grain) fanout key.
        Each value is itself a ``NestedChain``; empty ``{}`` denotes a leaf
        (an owned column with no further fanout). Outer-level chosen keys
        appear as nested branches inside their child level. Empty dict if
        no chunks were consumed. See
        :meth:`_StreamingFanoutInferrer.to_nested_chain` for the exact shape.
    """
    n_nom: int = sum(1 for f in features.values() if f["type"] == "nominal")
    n_con: int = sum(1 for f in features.values() if f["type"] == "continuous")
    if verbose:
        print(f"features: {n_nom} nominal, {n_con} continuous ({len(features)} columns)")
        print(
            f"budget: {max_rows:,} rows; re-infer every {rebatch_every_num_rows:,} rows; "
            f"converge after {stable_runs_required} stable runs\n"
        )

    inferrer = _StreamingFanoutInferrer(
        features,
        fanout_key_card_floor=fanout_key_card_floor,
        rebatch_every_num_rows=rebatch_every_num_rows,
        stable_runs_required=stable_runs_required,
        saturation_idle_rows=saturation_idle_rows,
        max_rows=max_rows,
    )

    t0: float = time.perf_counter()
    yield_kwargs: dict[str, Any] = {}
    if chunk_size is not None:
        yield_kwargs["chunk_size"] = chunk_size
    for chunk in data.yield_chunk(**yield_kwargs):
        inferrer.process_chunk(chunk)
        if inferrer.should_rebatch:
            chain: dict[str, list[str]] = inferrer.rebatch()
            if verbose:
                marker: str = "  *converged*" if inferrer.converged else ""
                print(
                    f"[chunk {inferrer.chunks_seen:>4} | {inferrer.rows_seen:>10,} rows]  "
                    f"levels={len(chain)}  stable_runs={inferrer.stable_runs}{marker}"
                )
        if inferrer.should_stop():
            break

    # Run once more if rows arrived after the last scheduled inference.
    if inferrer.has_pending_rows:
        inferrer.rebatch()

    if verbose:
        elapsed_ms: float = (time.perf_counter() - t0) * 1000
        status: str = "converged" if inferrer.converged else "budget exhausted"
        print(
            f"\nDone -- {inferrer.chunks_seen} chunks, {inferrer.rows_seen:,} rows, "
            f"{elapsed_ms:.0f} ms  ({status}).\n"
        )
    return _StreamingFanoutInferrer.to_nested_chain(inferrer.last_chain or {})


if __name__ == "__main__":
    df = pd.read_csv("joined_olist.csv")
    features = infer_feature_attributes(df, default_time_zone="UTC")
    data: AbstractData = DataFrameData(df)

    config = infer_fanout_feature_config(data, features=features, verbose=True)
    # NOTE: I would expect IFA's `fanout_feature_map` to also be defined as
    #       nested dicts `dict[str, dict[str, ...]]` rather than a
    #       weird `dict[str, list[str]]`.
    pprint(config)
