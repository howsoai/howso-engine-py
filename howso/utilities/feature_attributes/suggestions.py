from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Mapping, Sequence
import textwrap
from typing import Any
import warnings

from rich.console import Console
from rich.table import Table
from typing_extensions import Self

# Signal preservation parameters
# ------------------------------
# Provided by the user to specify values to protect, multipliers must be computed automatically
PreserveRareValuesMap = dict[str, list[Any]]
# Provided by the user to specify values to protect *with* multipliers, must only compute unprotected multipliers
PreserveRareValuesConfig = dict[str, list[dict[str, Any]]]
# Used internally to represent a "complete" configuration with protected *and* unprotected multipliers
FullPreserveRareValuesConfig = dict[str, dict[str, list[dict[str, Any]] | float]]


def wrap_text(text: str, width: int) -> str:
    """Wrap regular prose on word boundaries, never breaking words."""
    return "\n".join(textwrap.wrap(
        text, width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )) or text


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
    @abstractproperty
    def name(self) -> str:
        """The name of the suggestion, to be used as a key when accessing via the collector."""
        ...

    @property
    @abstractproperty
    def description(self) -> str:
        """A brief description of the suggestion."""
        ...

    @abstractmethod
    def apply(self, attributes: dict) -> None:
        """Apply this suggestion to the FeatureAttributesBase object."""
        ...

    @abstractmethod
    def merge(self, other: Self) -> Self:
        """Merge this suggestion with another if more than one were computed across separate processes."""
        ...


class PRVSuggestion(IFASuggestion):
    """A suggestion to configure preservation for rare values."""

    def __init__(self, prvc: PreserveRareValuesConfig, values_ranking: Sequence[Mapping[str, Any]],
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
        num_candidates = sum(len(cfg["protected_values_multipliers"]) for cfg in self._prvc.values())
        candidates_explanation = ""
        for candidate in self._ranking:
            candidates_explanation += f"\n    - Column name: {candidate['feature']}, value: {candidate['value']}"
        if self._user_set_mdc:
            candidates_explanation += (f"\n\nIn total, we identified {num_candidates} values that may be lost "
                                       "during data distillation.")
        header = "Rare Value Preservation"
        body = (
            "Here are some values in your data that may be good candidates for Rare Value Preservation:\n"
            f"{candidates_explanation}\n\n"
            "During data distillation workflows, nominal values with weak but detectable signals may "
            "be filtered out. To account for this, you may provide to `infer_feature_attributes` a "
            "`preserve_rare_values_map` detailing rare values to protect automatically, or a full "
            "`preserve_rare_values_config` with fine-grained case weight adjustments. Additionally, "
            "you may apply our suggested configuration for all detected possible rare values to this "
            "feature attributes object."
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
        if self._user_set_mdc:
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

    def apply(self, attributes: dict) -> None:
        """Apply the computed rare values preservation config to the FeatureAttributesBase object."""
        if not self._user_set_mdc:
            warnings.warn("Suggested rare values configuration will be applied, but the computed case weight "
                          "multipliers are likely inaccurate as `max_distilled_cases` was not provided to "
                          "`infer_feature_attributes`. Please provide this parameter or be aware that the case "
                          "weight multipliers were computed based on a default `max_distilled_cases` value of 25,000.",
                          UserWarning, stacklevel=3)
        for feature, config in self._prvc.items():
            attributes[feature]["preserve_rare_values"] = config

    def get_config(self) -> PreserveRareValuesConfig:
        """Get the `preserve_rare_values_config` for use in future calls to `infer_feature_attributes`."""
        return self._prvc

    def get_values_map(self) -> PreserveRareValuesMap:
        """Get the `preserve_rare_values_map` for use in future calls to `infer_feature_attributes."""
        values_map = {}
        for feature, config in self._prvc.items():
            values_map[feature] = ([value_config["value"] for value_config
                                    in config["protected_values_multipliers"]])
        return values_map

    def merge(self, other: Self) -> Self:
        """Merge another PreserveRareValuesConfig into this one if there are no conflicts."""
        for feature, config in other.get_config():
            if feature not in self._prvc:
                self._prvc[feature] = config
            elif self._prvc[feature] != config:
                raise ValueError("Cannot merge `preserve_rare_value_config` objects as they share features with "
                                    "differing configurations.")


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
        table = Table(title="Suggestions for Potential Data Quality Improvements",
                      caption="To view a more detailed description of a suggestion, access its `name` as a property "
                      "(e.g., `your_attributes_object.suggestions.preserve_rare_values`).\n\nTo apply all suggestions,"
                      ' call `your_attributes_object.apply_suggestion("all"))`.')
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

    def append(self, suggestion: IFASuggestion) -> None:
        """Append a new IFASuggestion to this collector."""
        if suggestion.name in self._suggestions:
            self._suggestions[suggestion.name].merge(suggestion)
        # Ensure the suggestion has access to the feature attributes
        self._suggestions[suggestion.name] = suggestion

    def merge(self, other: Self) -> Self:
        """Merge all IFASuggestions in another collector object with the IFASuggestions in this object."""
        for name, suggestion in other.suggestions.items():
            if name in self._suggestions:
                self._suggestions[name].merge(suggestion)
