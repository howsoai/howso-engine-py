import yaml

from howso.client.exceptions import HowsoConfigurationError
from howso.client.feature_flags import FeatureFlags


class HowsoConfiguration:
    """
    Howso client configuration.

    Parameters
    ----------
    config_path : str, optional
        The path to the user's howso.yml
    verbose : bool, default False
        Set verbose output.
    """

    feature_flags_class = FeatureFlags

    def __init__(self, *args, config_path=None, verbose=False, **kwargs):
        """Initialize the configuration object."""
        super().__init__(*args, **kwargs)
        self.howso_config_path = config_path
        self.verbose = verbose

        if config_path is not None:
            try:
                with open(config_path, 'r') as config:
                    self.user_config = yaml.safe_load(config)
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
            self.user_config = {}

        # Initialize feature flags
        self.feature_flags = self.feature_flags_class(
            self.user_config.get('feature_flags'))

    def get_user_config_option(self, *args, default=None):
        """
        Retrieve a configuration option from the user's howso.yml settings.

        Parameters
        ----------
        args : str
            The path to the option in the configuration data.
        default : Any, default None
            The value to default to if not found.

        Returns
        -------
        Any
            The value of the option at the given path.
        """
        if len(args) == 0:
            raise AssertionError('At least one configuration option key '
                                 'is required.')
        option = self.user_config
        for arg in args:
            try:
                option = option[arg]
            except (KeyError, TypeError):
                return default
        return option
