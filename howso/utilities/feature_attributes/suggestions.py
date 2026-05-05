from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

#from howso.utilities.feature_attributes.base import SignalPreservationConfig


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


class SPCSuggestion(IFASuggestion):
    """A suggestion to configure and candidates for protected values with signal preservation."""

    # The FeatureAttributesBase object, set by the collector
    _attributes: dict = None

    def __init__(self, spc: dict):
        self._spc = spc

    def __repr__(self):
        """Print a helpful description of this suggestion and its capabilities."""
        num_candidates = sum(len(cfg["protected_values"]) for cfg in self._spc.values())
        header = f"We found {num_candidates} candidate(s) for signal preservation while inferring feature attributes."""
        body = f""

    @property
    def name(self) -> str:
        """The name of this suggestion."""
        return "signal_preservation"

    @property
    def description(self) -> str:
        """A brief description of this suggestion."""
        return "Configure protected values to avoid losing their signal during data distillation."

    def apply(self):
        """Apply the computed signal preservation config to the FeatureAttributesBase object."""
        for feature, config in self._spc:
            self._attributes[feature]["signal_preservation"] = config

    def get_config(self) -> str:
        """Get the `signal_preservation_config` for use in future calls to `infer_feature_attributes`."""
        return self._spc


class IFASuggestionCollector:
    """Collector of IFASuggestion objects."""

    def __init__(self, feature_attributes: dict, suggestions: Sequence[IFASuggestion] = None):
        self._attributes = feature_attributes
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
        table = Table(title=f"{type(self).__name__}", caption="To view a more detailed description of a suggestion, "
                      "access its `name` as a property (e.g., `suggestions.signal_preservation`)")
        table.add_column("Name")
        table.add_column("Description")

        for name, suggestion in self._suggestions.items():
            table.add_row(name, suggestion.description)

        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get().rstrip()

    def append(self, suggestion: IFASuggestion):
        """Append a new IFASuggestion to this collector."""
        if suggestion.name in self._suggestions:
            raise ValueError(f"Suggestion of type `{suggestion.name}` already provided.")
        # Ensure the suggestion has access to the feature attributes
        suggestion._attributes = self._attributes
        self._suggestions[suggestion.name] = suggestion

    def apply_all(self):
        """Apply all suggestions to the FeatureAttributesBase object."""
        for suggestion in self.suggestions.values():
            suggestion.apply(self._attributes)
