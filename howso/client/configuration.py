from collections.abc import Iterable, Mapping
from pathlib import Path
import typing as t

import yaml

from howso.client.exceptions import HowsoConfigurationError
from howso.utilities.mapping import CaseInsensitiveMap
from .feature_flags import FeatureFlags


class BaseConfig:
    """Base class for Howso configuration settings."""

    key: str = ''
    """Dot separated key path to configuration object in the yaml."""

    required: set = set()
    """Required sub keys."""

    def __init__(self, config: t.Optional[CaseInsensitiveMap]) -> None:
        if config is None:
            config = CaseInsensitiveMap()
        if not isinstance(config, Mapping):
            raise ValueError('Invalid configuration object.')
        self._config = config
        self.validate()

    def validate(self) -> None:
        """Validate configuration options."""
        self.check_required(self.required)

    def check_required(self, required: Iterable[str]):
        """Check for required keys."""
        for key in required:
            if key not in self._config:
                raise HowsoConfigurationError(f'A value for the configuration option "{self.key}.{key}" is required.')


class ClientConfig(BaseConfig):
    """Representation of the Howso client user configuration options."""

    key = "howso"

    @property
    def client_class(self) -> str | None:
        """The import path to a client class to use."""
        return self._config.get('client')

    @property
    def client_extra_params(self) -> Mapping[str, t.Any]:
        """Additional client init parameters."""
        return self._config.get('client_extra_params') or {}


class HowsoConfiguration:
    """
    Howso client configuration.

    Parameters
    ----------
    config_path : Path or str, optional
        The path to the user's howso.yml file.
    verbose : bool, default False
        Set verbose output.
    """

    feature_flags_class: type[FeatureFlags] = FeatureFlags
    client_config_class: type[ClientConfig] = ClientConfig

    def __init__(self, *, config_path: t.Optional[Path | str] = None, verbose: bool = False):
        """Initialize the configuration object."""
        self.howso_config_path = config_path
        self.verbose = verbose

        if config_path is not None:
            try:
                with open(config_path, 'r') as config:
                    self._config = CaseInsensitiveMap(yaml.safe_load(config))
            except yaml.YAMLError as yaml_exception:
                raise HowsoConfigurationError(
                    'Unable to parse the configuration file located at '
                    f'"{config_path}". Please verify the YAML syntax '
                    'of this file and try again.'
                ) from yaml_exception
            except (IOError, OSError) as exception:
                raise HowsoConfigurationError(
                    'Error reading the configuration file located at '
                    f'"{config_path}". Check the file permissions and '
                    'try again.'
                ) from exception
        else:
            self._config = CaseInsensitiveMap()

        # Initialize configuration classes
        try:
            self.feature_flags = self.feature_flags_class(self._config.get('feature_flags'))
            self.client = self.client_config_class(self._config.get('howso'))
        except HowsoConfigurationError as ex:
            raise HowsoConfigurationError(
                f'See configuration file located at "{config_path}". {ex.message}', code=ex.code) from ex
