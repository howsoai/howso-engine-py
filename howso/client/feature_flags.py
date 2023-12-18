import typing as t
import warnings


class FeatureFlags:
    """
    Client feature flags store.

    Parameters
    ----------
    flags : dict, optional
        A dictionary of flags and their enabled status.
    """

    # Define obsolete flags here to raise a warning when defined
    _obsolete_flags: t.Union[t.Set[str], None] = None

    def __init__(self, flags: t.Optional[t.Dict[str, t.Any]]):
        self._store = dict()
        if flags is not None:
            obsolete = set()
            for key, value in flags.items():
                flag = self.parse_flag(key)
                if self._obsolete_flags and flag in self._obsolete_flags:
                    obsolete.add(flag)
                    continue
                self._store[flag] = bool(value)

            if obsolete:
                lines = (f"- {row}\n" for row in obsolete)
                warnings.warn(
                    "The following Howso feature flags are now obsolete and "
                    "can be removed from your configuration yaml file:\n"
                    f"{''.join(lines)}"
                )

    def is_enabled(self, flag: str) -> bool:
        """
        Get enabled state of a feature flag.

        Parameters
        ----------
        flag : str
            The name of the feature flag.

        Returns
        -------
        bool
            True if the feature flag is enabled.
        """
        return self._store.get(self.parse_flag(flag), False)

    @classmethod
    def parse_flag(cls, flag: str) -> str:
        """Parse the flag name."""
        return flag.replace('-', '_').lower()

    def __iter__(self) -> t.Generator[t.Tuple[str, bool], None, None]:
        """Iterate over flags."""
        return ((key, value) for key, value in self._store.items())

    def __repr__(self) -> str:
        """Implement repr magic method."""
        return str(dict(self._store.items()))
