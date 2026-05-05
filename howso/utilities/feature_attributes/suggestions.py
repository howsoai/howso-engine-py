from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.table import Table

# Signal preservation parameters
PreserveRareValuesMap = dict[str, list[Any]]
PreserveRareValuesConfig = dict[str, dict[str, list[dict[str, Any]] | float]]


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
        """Print a helpful description of this suggestion and its capabilities."""
        num_candidates = sum(len(cfg["protected_values"]) for cfg in self._prvc.values())
        header = f"We found {num_candidates} candidate value(s) in your data for rare value preservation."""
        body = ("During data distillation workflows, nominal values with weak but detectable signals may "
                "be filtered out. To account for this, you may provide to `infer_feature_attributes` a "
                "`preserve_rare_values_map` detailing rare values to protect automatically, or a full "
                "`preserve_rare_values_config` with fine-grained case weight adjustments. Additionally, you may apply "
                "our suggested configuration for all detected rare values to this feature attributes object.")
        table = Table(title="Summary of Available Options", show_lines=True)
        table.add_column("Action")
        table.add_column("Details")
        table.add_column("Code Example")

        # Option 1: apply all
        table.add_row(
            "Apply suggestion to this feature attributes object",
            "Save the suggested `preserve_rare_values_config` to this feature attributes object via `apply()`.",
            "```\nfeature_attributes = infer_feature_attributes(data, chunk_size=100_000)\n"
            "feature_attributes.suggestions.preserve_rare_values.apply()\n```"
        )
        # Option 2: get reusable config
        table.add_row(
            "Get a reusable `preserve_rare_values_config`",
            "You may provide a pre-computed `preserve_rare_values_config` as a parameter to `infer_feature_attributes`"
            "if you wish to make adjustments to the case weight multipliers.",
            "```\nfeature_attributes = infer_feature_attributes(data, chunk_size=100_000)\n"
            "config = feature_attributes.suggestions.preserve_rare_values.get_config()\n"
            "# TODO: make desired changes to the config\n"
            "feature_attributes = infer_feature_attributes(data, preserve_rare_values_config=config)\n```"
        )
        # Option 3: get rare values map
        table.add_row(
            "Edit the preserved rare values with a `preserve_rare_values_map`.",
            "The rare values to be preserved can be detailed via the `preserve_rare_values_map` parameter to "
            "`infer_feature_attributes. A good starting point may be the \"full\" map of all candidate values. "
            "All case weight multipliers will be automatically configured for the provided values.",
            "```\nfeature_attributes = infer_feature_attributes(data, chunk_size=100_000)\n"
            "values_map = feature_attributes.suggestions.preserve_rare_values.get_values_map()\n"
            "# TODO: make desired changes to the values map\n"
            "feature_attributes = infer_feature_attributes(data, chunk_size=100_000, "
            "preserve_rare_values_map=values_map)\n```"
        )

        console = Console()
        with console.capture() as capture:
            console.print(table)
        table_str = capture.get().rstrip()

        return f"{header}\n\n{body}\n\n{table_str}"

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

    def __getattr__(self, name: str) -> IFASuggestion:
        """Get the suggestion with the provided name."""
        if name not in self._suggestions:
            raise AttributeError("No suggestion found under the provided key.")
        return self._suggestions[name]

    def __repr__(self):
        """Print a helpful description of the available suggestions."""
        table = Table(title="Suggestions for Potential Data Quality Improvements",
                      caption="To view a more detailed description of a suggestion, access its `name` as a property "
                      "(e.g., `your_attributes_object.suggestions.preserve_rare_values`)")
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
