from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Sequence
import textwrap
from typing import Any

from rich.console import Console
from rich.table import Table

# Signal preservation parameters
PreserveRareValuesMap = dict[str, list[Any]]
PreserveRareValuesConfig = dict[str, dict[str, list[dict[str, Any]] | float]]


def wrap_text(text, width):
    """Wrap regular prose on word boundaries, never breaking words."""
    return "\n".join(textwrap.wrap(
        text, width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )) or text


def wrap_code(code, width):
    """Wrap code so long lines break, but preserve line structure."""
    out_lines = []
    for line in code.splitlines():
        if len(line) <= width:
            out_lines.append(line)
        else:
            # Allow breaking inside long code lines (no good word boundaries)
            out_lines.extend(textwrap.wrap(
                line, width=width,
                break_long_words=True,
                break_on_hyphens=False,
                drop_whitespace=False,
            ) or [line])
    return "\n".join(out_lines)


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
    def apply(self):
        """Apply this suggestion to the FeatureAttributesBase object."""
        ...

    @abstractmethod
    def merge(self):
        """Merge this suggestion with another if more than one were computed across separate processes."""
        ...


class PRVSuggestion(IFASuggestion):
    """A suggestion to configure preservation for rare values."""

    # The FeatureAttributesBase object, set by that object once multiprocessing has completed
    _attributes: dict = None

    def __init__(self, prvc: PreserveRareValuesConfig):
        self._prvc = prvc

    def __repr__(self):
        """Print a helpful description of this IFASuggestion."""
        num_candidates = sum(len(cfg["protected_values"]) for cfg in self._prvc.values())
        header = f"We found {num_candidates} candidate value(s) in your data for rare value preservation."
        body = (
            "During data distillation workflows, nominal values with weak but detectable signals may "
            "be filtered out. To account for this, you may provide to `infer_feature_attributes` a "
            "`preserve_rare_values_map` detailing rare values to protect automatically, or a full "
            "`preserve_rare_values_config` with fine-grained case weight adjustments. Additionally, "
            "you may apply our suggested configuration for all detected rare values to this feature "
            "attributes object."
        )

        # Pick a target total width and divvy it up
        total_width = 120
        action_w, details_w, code_w = 24, 40, 50

        table = Table(title="Summary of Available Options", show_lines=True, width=total_width,
                      caption="All listed method calls are accessed through this suggestion, e.g., "
                      "`my_attributes_object.suggestions.preserve_rare_values.apply()`")
        table.add_column("Action", min_width=action_w, overflow="fold")
        table.add_column("Details", min_width=details_w, overflow="fold")
        table.add_column("Applicalbe Method Call")

        rows = [
            (
                "Apply suggestion to this feature attributes object",
                "Save the suggested `preserve_rare_values_config` to this feature attributes "
                "object via `apply()`.",
                "apply()"
            ),
            (
                "Get a reusable `preserve_rare_values_config`",
                "You may provide a pre-computed `preserve_rare_values_config` as a parameter to "
                "`infer_feature_attributes` if you wish to make adjustments to the case weight "
                "multipliers.",
                "get_config()"
            ),
            (
                "Edit the preserved rare values with a `preserve_rare_values_map`",
                "The rare values to be preserved can be detailed via the `preserve_rare_values_map` "
                "parameter to `infer_feature_attributes`. A good starting point may be the \"full\" "
                "map of all candidate values. All case weight multipliers will be automatically "
                "configured for the provided values.",
                "get_values_map()"
            ),
        ]

        for action, details, code in rows:
            table.add_row(
                wrap_text(action, action_w),
                wrap_text(details, details_w),
                wrap_code(code, code_w),
            )

        console = Console(width=total_width)
        with console.capture() as capture:
            console.print(table)
        return f"{header}\n\n{wrap_text(body, total_width)}\n\n{capture.get().rstrip()}"

    @property
    def name(self) -> str:
        """The name of this suggestion."""
        return "preserve_rare_values"

    @property
    def description(self) -> str:
        """A brief description of this suggestion."""
        return "Configure rare values to avoid losing their signal during data distillation."

    def apply(self):
        """Apply the computed signal preservation config to the FeatureAttributesBase object."""
        for feature, config in self._prvc.items():
            self._attributes[feature]["preserve_rare_values"] = config

    def get_config(self) -> PreserveRareValuesConfig:
        """Get the `preserve_rare_values_config` for use in future calls to `infer_feature_attributes`."""
        return self._prvc

    def get_values_map(self) -> PreserveRareValuesMap:
        """Get the `preserve_rare_values_map` for use in future calls to `infer_feature_attributes."""
        values_map = {}
        for feature, config in self._prvc:
            values_map[feature] = [value_config["value"] for value_config in config["protected_values"]]
        return values_map

    def merge(self, other: object):
        """Merge another PreserveRareValuesConfig into this one if there are no conflicts."""
        for feature, config in other.get_config():
            if feature not in self._prvc:
                self._prvc[feature] = config
            else:
                if self._prvc[feature] != config:
                    raise ValueError("Cannot merge `preserve_rare_value_config` objects as they share features with "
                                     "differing configurations.")


class IFASuggestionCollector:
    """Collector of IFASuggestion objects."""

    def __init__(self, suggestions: Sequence[IFASuggestion] = None):
        self._suggestions: dict[str, IFASuggestion] = {}
        suggestions = suggestions or []
        for suggestion in suggestions:
            self.append(suggestion)

    def __getattr__(self, key: str) -> IFASuggestion:
        """Get the suggestion with the provided key."""
        # Avoid an infinite loop with partially constructed objects
        if key.startswith('__') and key.endswith('__'):
            raise AttributeError(key)
        if key not in self._suggestions:
            raise AttributeError("No suggestion found under the provided key.")
        return self._suggestions[key]

    def __repr__(self):
        """Print a helpful description of the available suggestions."""
        table = Table(title="Suggestions for Potential Data Quality Improvements",
                      caption="To view a more detailed description of a suggestion, access its `name` as a property "
                      "(e.g., `your_attributes_object.suggestions.preserve_rare_values`).\n\nTo apply all suggestions,"
                      " call `your_attributes_object.suggestions.apply_all()`.")
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

    def append(self, suggestion: IFASuggestion):
        """Append a new IFASuggestion to this collector."""
        if suggestion.name in self._suggestions:
            self._suggestions[suggestion.name].merge(suggestion)
        # Ensure the suggestion has access to the feature attributes
        self._suggestions[suggestion.name] = suggestion

    def apply_all(self):
        """Apply all suggestions to the FeatureAttributesBase object."""
        for suggestion in self.suggestions_collector.values():
            suggestion.apply()

    def merge(self, other: object):
        """Merge all IFASuggestions in another collector object with the IFASuggestions in this object."""
        for name, suggestion in other.suggestions.items():
            if name in self._suggestions.keys():
                self._suggestions[name].merge(suggestion)
