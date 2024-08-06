from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import typing as t

from requests.structures import CaseInsensitiveDict
from typing_extensions import TypeVar
import yaml

from howso.client.exceptions import HowsoConfigurationError
from .feature_flags import FeatureFlags

CO = TypeVar("CO", bound="ClientOptions")


class BaseOptions:
    """Base class for Howso configuration options."""

    key: str = ''
    """Dot separated key path to configuration object in the yaml."""

    required: set = set()
    """Required sub keys."""

    def __init__(self, config: t.Optional[Mapping]) -> None:
        if config is None:
            config = {}
        elif not isinstance(config, Mapping):
            raise ValueError('Invalid configuration object.')
        self._config = CaseInsensitiveDict(config)
        self.post_init()

    def post_init(self) -> None:
        """Complete any additional setup or validation."""
        self.check_required(self.required)

    def check_required(self, required: Iterable[str]):
        """Check for required keys."""
        for key in required:
            if key not in self._config:
                raise HowsoConfigurationError(f'A value for the configuration option "{self.key}.{key}" is required.')


class ClientOptions(BaseOptions):
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


class HowsoConfiguration(t.Generic[CO]):
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
    client_config_class: type[ClientOptions] = ClientOptions

    def __init__(self, config_path: t.Optional[Path | str] = None, *, verbose: bool = False):
        """Initialize the configuration object."""
        self.howso_config_path = None
        self.verbose = verbose

        if config_path is not None:
            try:
                self.howso_config_path = Path(config_path).expanduser()
                with open(self.howso_config_path, 'r') as config:
                    self._config = CaseInsensitiveDict(yaml.safe_load(config))
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
            self._config = CaseInsensitiveDict()

        # Initialize configuration classes
        try:
            self.setup()
        except HowsoConfigurationError as ex:
            if config_path is not None:
                raise HowsoConfigurationError(
                    f'See configuration file located at "{config_path}". {ex.message}', code=ex.code) from ex
            else:
                raise HowsoConfigurationError(
                    f'A configuration file is required.". {ex.message}', code=ex.code) from ex

    def setup(self):
        """Setup configuration attributes."""
        self.feature_flags: FeatureFlags = self.feature_flags_class(self._config.get('feature_flags'))
        self.client: CO = t.cast(CO, self.client_config_class(self._config.get('howso')))
