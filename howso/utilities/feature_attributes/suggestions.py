from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import datetime
import json
import textwrap
from typing import Any, Self, TYPE_CHECKING, TypedDict
import warnings

import numpy as np
from rich.console import Console
from rich.table import Table

from howso.utilities.progress import print_status

if TYPE_CHECKING:
    from howso.client.typing import FullPreserveRareValuesConfig, PreserveRareValuesMap

# Fanout features parameters
# --------------------------
FanoutFeaturesMap = dict[tuple[str, ...] | str, list[str]]

_MAX_EXAMPLE_KEYS = 4
"""The number of key features listed as examples when printing a fan-out suggestion."""

_MAX_EXAMPLE_COLUMNS = 3
"""The number of fan-out features listed per key when printing a fan-out suggestion."""


class FanoutFeatureGroup(TypedDict):
    """A JSON-friendly fan-out configuration for compound fanout keys."""

    key_features: list[str]
    """The key features whose values select groups of cases sharing the fanned-out values."""

    fanout_features: list[str]
    """The features whose values are fanned out across the cases of each group."""


FanoutFeaturesInput = FanoutFeaturesMap | Sequence[FanoutFeatureGroup]
"""A ``fanout_feature_map``, as a mapping or as a list of :class:`FanoutFeatureGroup`."""


def normalize_fanout_feature_map(fanout_feature_map: FanoutFeaturesInput) -> FanoutFeaturesMap:
    """
    Normalize either accepted form of ``fanout_feature_map`` into a :data:`FanoutFeaturesMap`.

    Parameters
    ----------
    fanout_feature_map : FanoutFeaturesMap or Sequence of FanoutFeatureGroup
        A mapping of key feature name(s) to fan-out feature names, or a list of groups each
        holding ``key_features`` and ``fanout_features`` lists (the form produced by
        :meth:`IFASuggestionCollector.to_dict`).

    Returns
    -------
    FanoutFeaturesMap
        The equivalent mapping. A group with a single key feature is keyed by that name;
        a group with several key features is keyed by a tuple of names.

    Raises
    ------
    TypeError
        If the value is neither a mapping nor a sequence of groups.
    ValueError
        If a group is missing ``key_features`` or ``fanout_features``, or has no key features.
    """
    if isinstance(fanout_feature_map, Mapping):
        return dict(fanout_feature_map)
    if isinstance(fanout_feature_map, (str, bytes)) or not isinstance(fanout_feature_map, Sequence):
        raise TypeError("`fanout_feature_map` must be a mapping or a list of groups with `key_features` and "
                        f"`fanout_features`; got {type(fanout_feature_map).__name__}.")
    normalized: FanoutFeaturesMap = {}
    for group in fanout_feature_map:
        if not isinstance(group, Mapping) or "key_features" not in group or "fanout_features" not in group:
            raise ValueError("Each group in a list-form `fanout_feature_map` must be a mapping with "
                             "`key_features` and `fanout_features`.")
        key_features = group["key_features"]
        if isinstance(key_features, str):
            key_features = [key_features]
        key_features = list(key_features)
        if not key_features:
            raise ValueError("Each group in a list-form `fanout_feature_map` needs at least one key feature.")
        key: tuple[str, ...] | str = key_features[0] if len(key_features) == 1 else tuple(key_features)
        existing = normalized.setdefault(key, [])
        existing.extend(f for f in group["fanout_features"] if f not in existing)
    return normalized


# Machine-readable output
# -----------------------
SUGGESTIONS_SCHEMA_VERSION = 1
"""Version of the structure returned by :meth:`IFASuggestionCollector.to_dict`."""


class SuggestionCaveat(TypedDict):
    """A condition that limits how far a suggestion can be trusted or applied."""

    code: str
    """A stable identifier for the condition."""

    message: str
    """A human-readable explanation of the condition."""


def _json_default(obj: Any) -> Any:
    """Convert values the standard JSON encoder cannot handle, such as numpy scalars and datetimes."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime.date, datetime.time)):
        return obj.isoformat()
    return str(obj)


def wrap_text(text: str, width: int) -> str:
    """Wrap regular prose on word boundaries, never breaking words."""
    return "\n".join(textwrap.wrap(
        text, width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )) or text


def _count(value: int, singular: str, plural: str | None = None) -> str:
    """Render ``value`` with the correctly pluralized noun, e.g. ``"1 key"`` or ``"3 keys"``."""
    noun = singular if value == 1 else (plural or f"{singular}s")
    return f"{value:,} {noun}"


def wrap_paragraphs(text: str, width: int) -> str:
    """Wrap text on newlines."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            out.append("")
            continue
        # Detect leading whitespace to use as continuation indent
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        # For bullet-style lines, indent continuations past the bullet
        if stripped.startswith("- "):
            cont_indent = indent + "  "
        else:
            cont_indent = indent
        out.append(textwrap.fill(
            line,
            width=width,
            initial_indent="",
            subsequent_indent=cont_indent,
            break_long_words=False,
            break_on_hyphens=False,
        ))
    return "\n".join(out)


class IFASuggestion(ABC):
    """Base class for a warning emitter common to IFA that must list applicable features discovered across shards."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the suggestion, to be used as a key when accessing via the collector."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """A brief description of the suggestion."""
        ...

    @property
    def summary(self) -> str:
        """
        A one-line statement of what was found, for the summary printed when inference completes.

        Should read as a finding rather than an instruction, e.g. ``"Found 3 fan-out features
        across 1 key"``. Defaults to :attr:`description` for suggestions that do not override it.
        """
        return self.description

    @property
    def can_apply(self) -> bool:
        """Whether :meth:`apply` changes the feature attributes, rather than declining with a warning."""
        return True

    @property
    def caveats(self) -> list[SuggestionCaveat]:
        """Conditions that limit how far this suggestion can be trusted or applied."""
        return []

    @property
    @abstractmethod
    def details(self) -> dict[str, Any]:
        """Structured findings behind this suggestion, specific to the suggestion type."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """
        Keyword arguments to ``infer_feature_attributes`` that act on this suggestion.

        Each value can be edited and passed back to ``infer_feature_attributes`` under its key.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Get a machine-readable representation of this suggestion.

        This never prints or warns; caveats that other methods raise as warnings are
        reported under ``caveats``.

        Returns
        -------
        dict
            A dict with the keys ``name``, ``summary``, ``description``, ``can_apply``,
            ``caveats``, ``details`` and ``parameters``.
        """
        return {
            "name": self.name,
            "summary": self.summary,
            "description": self.description,
            "can_apply": self.can_apply,
            "caveats": self.caveats,
            "details": self.details,
            "parameters": self.parameters,
        }

    @abstractmethod
    def apply(self, attributes: dict) -> None:
        """Apply this suggestion to the FeatureAttributesBase object."""
        ...

    @abstractmethod
    def merge(self, other: "IFASuggestion") -> None:
        """Merge this suggestion with another if more than one were computed across separate processes."""
        ...


class FanoutFeaturesSuggestion(IFASuggestion):
    """
    A suggestion to configure fanout features.

    Parameters
    ----------
    fanout_features : FanoutFeaturesMap
        A candidate configuration for fanout features.
    """

    def __init__(self, fanout_features: FanoutFeaturesMap) -> None:
        self._fanout_features = fanout_features

    def __repr__(self) -> str:
        """Print a helpful description of this IFASuggestion."""
        header = "Fan-out Features"

        details = self.details
        num = details["num_key_features"]

        body = (
            f"We have detected {num} key(s) that should be considered as fan-out features. Fan-out "
            "features are columns that have repeated values across multiple rows based on a single "
            "observation. Informing the Howso Engine of fan-out features via your feature attributes "
            "will help it measure uncertainty more accurately. \n\n\tTo read more about fan-out "
            "features, please see: "
            "https://docs.howso.com/en/latest/user_guide/advanced_capabilities/fanout_features.html\n\n"
            "Examples In Your Data:\n"
            "----------------------\n"
        )

        groups = details["groups"]
        for group in groups[:_MAX_EXAMPLE_KEYS]:
            fofs = group["fanout_features"][:_MAX_EXAMPLE_COLUMNS]
            num_not_shown = len(group["fanout_features"]) - len(fofs)
            _start = f"Columns `{'`, `'.join(fofs)}`"
            if num_not_shown > 0:
                _start += f", and {num_not_shown} more"
            key_features = group["key_features"]
            key = key_features[0] if len(key_features) == 1 else tuple(key_features)
            body += f"  - {_start} have repeated values derived from observations in `{key}`\n"
        num_keys_not_shown = len(groups) - _MAX_EXAMPLE_KEYS
        if num_keys_not_shown > 0:
            body += f"  - ...and {_count(num_keys_not_shown, 'more key')}\n"
        body += "\n"

        # Pick a target total width and divvy it up
        total_width = 120
        action_w, details_w, code_w = 24, 40, 50

        options_table = Table(title="Summary of Available Options", show_lines=True, width=total_width)
        options_table.add_column("Action", min_width=action_w, overflow="fold")
        options_table.add_column("Details", min_width=details_w, overflow="fold")
        options_table.add_column("Relevant Code")

        rows = []

        rows.extend([
            (
                "Get a reusable `fanout_feature_map`",
                "You may provide `fanout_feature_map` as a parameter to "
                "`infer_feature_attributes` if you wish to adjust the fan-out feature "
                "configuration. Our detected fan-out feature configuration may be a "
                "good starting point.",
                "From this suggestion object call: "
                "`get_fanout_feature_map()`"
            ),
            (
                "Apply suggestion to this feature attributes object",
                "Save the suggested candidate `fanout_feature_map` "
                "to this feature attributes object.",
                "Call `apply_suggestion()` on the feature attributes object: "
                '`apply_suggestion("fanout_features")`'
            ),
        ])

        for action, details, code in rows:
            options_table.add_row(
                wrap_text(action, action_w),
                wrap_text(details, details_w),
                wrap_text(code, code_w),
            )

        console = Console(width=total_width)
        with console.capture() as capture:
            console.print(options_table)
        return f"{header}\n\n{wrap_paragraphs(body, total_width)}\n\n{capture.get().rstrip()}"

    @property
    def name(self) -> str:
        """The name of this suggestion."""
        return "fanout_features"

    @property
    def description(self) -> str:
        """A brief description of this suggestion."""
        return "Configure fan-out features so that the Howso Engine can more accurately measure uncertainty."

    @property
    def summary(self) -> str:
        """A one-line statement of the fan-out features found."""
        details = self.details
        return (f"Found {_count(details['num_fanout_features'], 'fan-out feature')} across "
                f"{_count(details['num_key_features'], 'column')}")

    @property
    def details(self) -> dict[str, Any]:
        """
        The fan-out features found.

        Contains ``num_key_features`` (the number of key feature groups), ``num_fanout_features``
        and ``groups``, a list of :class:`FanoutFeatureGroup`.
        """
        groups = self._groups()
        return {
            "num_key_features": len(groups),
            "num_fanout_features": sum(len(g["fanout_features"]) for g in groups),
            "groups": groups,
        }

    @property
    def parameters(self) -> dict[str, Any]:
        """The suggested ``fanout_feature_map``, in its list-of-groups form."""
        return {"fanout_feature_map": self._groups()}

    def _groups(self) -> list[FanoutFeatureGroup]:
        """Get the fan-out configuration as a list of groups with list-valued key features."""
        return [
            {
                "key_features": [key] if isinstance(key, str) else list(key),
                "fanout_features": list(cols),
            }
            for key, cols in self._fanout_features.items()
        ]

    def apply(self, attributes: dict) -> None:
        """Apply the computed fanout features config to the FeatureAttributesBase object."""
        for key_features, fanout_features in self._fanout_features.items():
            _key_features = key_features
            if isinstance(_key_features, str):
                _key_features = [key_features]
            for f in fanout_features:
                if f in attributes:
                    attributes[f]["fanout_on"] = list(_key_features)

    def get_fanout_feature_map(self) -> FanoutFeaturesMap:
        """Get the `fanout_feature_map` for use in future calls to `infer_feature_attributes`."""
        return self._fanout_features

    def merge(self, other: IFASuggestion) -> None:
        """Merge another FanoutFeaturesSuggestion into this one."""
        if not isinstance(other, FanoutFeaturesSuggestion):
            raise TypeError(f"Cannot merge {type(other).__name__} into FanoutFeaturesSuggestion.")
        for key, cols in other.get_fanout_feature_map().items():
            if key in self._fanout_features:
                existing = self._fanout_features[key]
                self._fanout_features[key] = existing + [c for c in cols if c not in existing]
            else:
                self._fanout_features[key] = cols


DEFAULT_MAX_DISTILLED_CASES_CAVEAT = "default_max_distilled_cases"
"""Caveat code for rare value multipliers computed from a default ``max_distilled_cases``."""

_DEFAULT_MAX_DISTILLED_CASES_MESSAGE = (
    "The computed case weights for rare value multipliers are likely inaccurate as "
    "`max_distilled_cases` was not provided to `infer_feature_attributes`. Please provide "
    "this parameter or be aware that the case weight multipliers were computed based on a "
    "default `max_distilled_cases` value of 50,000. "
    "An accurate `max_distilled_cases` enables Howso to correctly weight the influence of rare "
    "values in the data, since the weighting is calibrated proportionally to the number of cases "
    "remaining after distillation."
)

_MAX_RANKED_VALUES = 5


class PRVSuggestion(IFASuggestion):
    """A suggestion to configure preservation for rare values."""

    def __init__(self, prvc: FullPreserveRareValuesConfig, values_ranking: Sequence[Mapping[str, Any]],
                 user_set_max_distilled_cases: bool) -> None:
        """
        Instantiate this Preserve Rare Values Suggestion.

        Parameters
        ----------
        prvc : FullPreserveRareValuesConfig
            A full rare values preservation config with protected and unprotected multipliers.
        values_ranking : Sequence of Mapping
            An ordered list of the top five most significant rare values found in the data.
        user_set_max_distilled_cases : bool
            Whether the user specified the max_distilled_cases value, or `prvc` was approximated with a default.
        """
        self._prvc = prvc
        self._ranking = values_ranking
        self._user_set_mdc = user_set_max_distilled_cases

    def __repr__(self) -> str:
        """Print a helpful description of this IFASuggestion."""
        details = self.details
        candidates_explanation = ""
        for candidate in details["top_values"]:
            candidates_explanation += f"\n    - Column name: {candidate['feature']}, value: {candidate['value']}"
        if self.can_apply:
            candidates_explanation += (f"\n\nIn total, we identified {details['num_values']} values that may be "
                                       "lost during data distillation.")
        header = "Rare Value Preservation"
        body = (
            "Here are some values in your data that may be good candidates for Rare Value Preservation:\n"
            f"{candidates_explanation}\n\n"
            "During data distillation workflows, nominal values with weak but detectable signals may "
            "be filtered out. To account for this, you may provide to `infer_feature_attributes` a "
            "`preserve_rare_values_map` detailing rare values to protect automatically, or a full "
            "`preserve_rare_values_config` with fine-grained case weight adjustments. Additionally, "
            "you may apply our suggested configuration for all detected possible rare values to this "
            "feature attributes object. Applying Rare Value Preservation may increase the influence "
            "of rare values on the aggregate signal of the dataset. This is the intended effect to help "
            "preserve the signal of rare values that would otherwise be lost during distillation. "
        )

        # Pick a target total width and divvy it up
        total_width = 120
        action_w, details_w, code_w = 24, 40, 50

        options_table = Table(title="Summary of Available Options", show_lines=True, width=total_width)
        options_table.add_column("Action", min_width=action_w, overflow="fold")
        options_table.add_column("Details", min_width=details_w, overflow="fold")
        options_table.add_column("Relevant Code")

        rows = []

        # Only suggest this option if the user actually set the `max_distilled_cases` value,
        # otherwise the computed multipliers may be very incorrect and should only be used
        # as examples.
        if self.can_apply:
            rows.append((
                "Apply suggestion to this feature attributes object",
                "Save the suggested candidate `preserve_rare_values_config` "
                "to this feature attributes object.",
                "Call `apply_suggestion()` on the feature attributes object: "
                '`apply_suggestion("preserve_rare_values")`'
            ))

        rows.extend([
            (
                "Get a reusable `preserve_rare_values_config`",
                "You may provide a pre-computed `preserve_rare_values_config` as a parameter to "
                "`infer_feature_attributes` if you wish to make adjustments to the case weight "
                "multipliers.",
                "From this suggestion object call: "
                "`get_config()`"
            ),
            (
                "Edit the preserved rare values with a `preserve_rare_values_map`",
                "The rare values to be preserved can be detailed via the `preserve_rare_values_map` "
                'parameter to `infer_feature_attributes`. A good starting point may be the "full" '
                "map of all candidate values. All case weight multipliers will be automatically "
                "configured for the provided values.",
                "From this suggestion object call: "
                "`get_values_map()`"
            ),
        ])

        for action, details, code in rows:
            options_table.add_row(
                wrap_text(action, action_w),
                wrap_text(details, details_w),
                wrap_text(code, code_w),
            )

        console = Console(width=total_width)
        with console.capture() as capture:
            console.print(options_table)
        return f"{header}\n\n{wrap_paragraphs(body, total_width)}\n\n{capture.get().rstrip()}"

    @property
    def name(self) -> str:
        """The name of this suggestion."""
        return "preserve_rare_values"

    @property
    def description(self) -> str:
        """A brief description of this suggestion."""
        return "Configure rare values to avoid losing their signal during data distillation."

    @property
    def summary(self) -> str:
        """A one-line statement of the rare values found."""
        details = self.details
        return (f"Found {_count(details['num_values'], 'rare value')} across "
                f"{_count(details['num_features'], 'column')} "
                "whose signal may be lost during data distillation workflows")

    @property
    def can_apply(self) -> bool:
        """Whether the multipliers can be applied, which requires a user-provided ``max_distilled_cases``."""
        return self._user_set_mdc

    @property
    def caveats(self) -> list[SuggestionCaveat]:
        """A caveat when the multipliers were computed from a default ``max_distilled_cases``."""
        if self._user_set_mdc:
            return []
        return [{"code": DEFAULT_MAX_DISTILLED_CASES_CAVEAT, "message": _DEFAULT_MAX_DISTILLED_CASES_MESSAGE}]

    @property
    def details(self) -> dict[str, Any]:
        """
        The rare values found.

        Contains ``num_values``, ``num_features`` and ``top_values``, the most frequent
        candidates as dicts of ``feature``, ``value`` and ``count``, most frequent first.
        """
        return {
            "num_values": sum(len(cfg["protected_values_multipliers"]) for cfg in self._prvc.values()),
            "num_features": len(self._prvc),
            "top_values": [dict(candidate) for candidate in self._ranking],
        }

    @property
    def parameters(self) -> dict[str, Any]:
        """The suggested ``preserve_rare_values_config`` and the matching ``preserve_rare_values_map``."""
        return {
            "preserve_rare_values_config": self._prvc,
            "preserve_rare_values_map": self._values_map(),
        }

    def _warn_default_max_distilled_cases(self, addendum: str = "", stack_level: int = 4) -> None:
        """
        Warn that the case weight multipliers were computed from a default ``max_distilled_cases``.

        Parameters
        ----------
        addendum : str, default ""
            An additional sentence appended to the warning, describing the consequence for the
            calling method.
        stack_level : int, default 4
            The stack level value to pass into `warn` via `stacklevel`. The default attributes the
            warning to the caller of `apply_suggestion()`; methods a user calls directly pass 3.
        """
        warnings.warn(
            _DEFAULT_MAX_DISTILLED_CASES_MESSAGE + addendum,
            UserWarning,
            stacklevel=stack_level,
        )

    def apply(self, attributes: Mapping[str, Any]) -> None:
        """Apply the computed rare values preservation config to the FeatureAttributesBase object."""
        if not self._user_set_mdc:
            self._warn_default_max_distilled_cases(
                " Since an inaccurate value may result in rare values being under-weighted or "
                "over-weighted, this suggestion was not applied."
            )
            return
        for feature, config in self._prvc.items():
            attributes[feature]["preserve_rare_values"] = config

    def get_config(self, enable_warnings: bool = True) -> FullPreserveRareValuesConfig:
        """Get the `preserve_rare_values_config` for use in future calls to `infer_feature_attributes`."""
        if not self._user_set_mdc and enable_warnings:
            self._warn_default_max_distilled_cases(stack_level=3)
        return self._prvc

    def get_values_map(self) -> PreserveRareValuesMap:
        """Get the `preserve_rare_values_map` for use in future calls to `infer_feature_attributes."""
        if not self._user_set_mdc:
            self._warn_default_max_distilled_cases(stack_level=3)
        return self._values_map()

    def _values_map(self) -> PreserveRareValuesMap:
        """Get the protected values of each feature, without warning."""
        return {
            feature: [value_config["value"] for value_config in config["protected_values_multipliers"]]
            for feature, config in self._prvc.items()
        }

    def merge(self, other: IFASuggestion) -> None:
        """Merge another PRVSuggestion into this one if there are no conflicts."""
        if not isinstance(other, PRVSuggestion):
            raise TypeError(f"Cannot merge {type(other).__name__} into PRVSuggestion.")
        for feature, config in other.get_config(enable_warnings=False).items():
            if feature not in self._prvc:
                self._prvc[feature] = config
            elif self._prvc[feature] != config:
                raise ValueError("Cannot merge `preserve_rare_value_config` objects as they share features with "
                                    "differing configurations.")
        ranking = list(self._ranking)
        for candidate in other._ranking:
            if not any(c["feature"] == candidate["feature"] and c["value"] == candidate["value"] for c in ranking):
                ranking.append(candidate)
        self._ranking = sorted(ranking, key=lambda c: c["count"], reverse=True)[:_MAX_RANKED_VALUES]


class IFASuggestionCollector:
    """Collector of IFASuggestion objects."""

    def __init__(self, suggestions: Sequence[IFASuggestion] | None = None) -> None:
        self._suggestions: dict[str, IFASuggestion] = {}
        suggestions = suggestions or []
        for suggestion in suggestions:
            self.append(suggestion)

    def __getattr__(self, key: str) -> IFASuggestion:
        """Get the suggestion with the provided key."""
        # Avoid an infinite loop with partially constructed objects
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        if key not in self._suggestions:
            raise AttributeError("No suggestion found under the provided key.")
        return self._suggestions[key]

    def __repr__(self) -> str:
        """Print a helpful description of the available suggestions."""
        if not self._suggestions:
            return "You have no suggestions."
        table = Table(title="Suggestions for Potential Data Quality Improvements",
                      caption="To view a more detailed description of a suggestion, access its `name` as a property "
                      "(e.g., `your_attributes_object.suggestions.preserve_rare_values`).\n\nTo apply all suggestions,"
                      ' call `your_attributes_object.apply_suggestion("all"))`.',
                      show_lines=True)
        table.add_column("Name")
        table.add_column("Description")

        for name, suggestion in self._suggestions.items():
            table.add_row(name, suggestion.description)

        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get().rstrip()

    @property
    def suggestions(self) -> dict[str, IFASuggestion]:
        """Get all suggestions that belong to this collector."""
        return self._suggestions

    def summary_lines(self, *, attributes_name: str = "your_attributes_object") -> list[str]:
        """
        Render the summary block as lines of console markup.

        The block is styled to sit beside the progress output from
        :mod:`howso.utilities.progress`: a bold cyan header, the way a progress
        label is styled, one dim-bulleted finding per suggestion, indented the
        way progress lines are, and a dim footer pointing at the suggestions.

        Parameters
        ----------
        attributes_name : str, default "your_attributes_object"
            How to refer to the returned feature attributes object in the footer.

        Returns
        -------
        list of str
            One entry per line, or an empty list when there are no suggestions.
        """
        if not self._suggestions:
            return []
        lines = ["[bold cyan]Feature Attributes Summary[/bold cyan]"]
        lines.extend(f"  [dim]\u00b7[/dim] {suggestion.summary}" for suggestion in self._suggestions.values())
        lines.append(f"  [dim]Inspect `{attributes_name}.suggestions` for details and how to apply them.[/dim]")
        return lines

    def print_summary(self, *, console: Console | None = None) -> None:
        """
        Print the summary of collected suggestions, when there are any.

        Parameters
        ----------
        console : Console, optional
            Console to print to. Defaults to the one :func:`howso.utilities.progress.status_console`
            selects for the current environment.
        """
        lines = self.summary_lines()
        if lines:
            print_status(*lines, console=console)

    def to_dict(self) -> dict[str, Any]:
        """
        Get a machine-readable representation of all collected suggestions.

        This never prints or warns. Values are Python objects, e.g., rare values keep their
        original type; use :meth:`to_json` for a JSON string.

        Returns
        -------
        dict
            A dict with ``schema_version`` (:data:`SUGGESTIONS_SCHEMA_VERSION`) and
            ``suggestions``, a list of :meth:`IFASuggestion.to_dict` results.
        """
        return {
            "schema_version": SUGGESTIONS_SCHEMA_VERSION,
            "suggestions": [suggestion.to_dict() for suggestion in self._suggestions.values()],
        }

    def to_json(self, **kwargs: Any) -> str:
        """
        Get a JSON string of :meth:`to_dict`.

        Numpy scalars and arrays become native JSON values, dates and times become ISO 8601
        strings, and any other value JSON cannot represent becomes its ``str()``.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed through to :func:`json.dumps`, such as ``indent``.

        Returns
        -------
        str
            The JSON representation of the collected suggestions.
        """
        kwargs.setdefault("default", _json_default)
        return json.dumps(self.to_dict(), **kwargs)

    def append(self, suggestion: IFASuggestion) -> None:
        """Append a new IFASuggestion to this collector."""
        if suggestion.name in self._suggestions:
            self._suggestions[suggestion.name].merge(suggestion)
        else:
            self._suggestions[suggestion.name] = suggestion

    def merge(self, other: Self) -> None:
        """Merge all IFASuggestions in another collector object with the IFASuggestions in this object."""
        for name, suggestion in other.suggestions.items():
            if name in self._suggestions:
                self._suggestions[name].merge(suggestion)
            else:
                self._suggestions[name] = suggestion
