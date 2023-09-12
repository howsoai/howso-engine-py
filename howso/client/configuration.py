import os

from howso.client.exceptions import HowsoConfigurationError
import yaml

ENV_CREATE_MAX_WAIT_TIME = 'HOWSO_CLIENT_CREATE_MAX_WAIT_TIME'


class HowsoConfiguration:
    """
    Howso client configuration.

    Parameters
    ----------
    config_path : str
        The path to the user's howso.yml
    verbose : bool, default False
        Set verbose output.
    """

    def __init__(self, *args, config_path=None, verbose=False, **kwargs):
        """Initialize the configuration object."""
        super().__init__(*args, **kwargs)
        self.howso_config_path = config_path
        self.verbose = verbose

        if self.verbose:
            print(f'Using config file: {config_path}')

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

    def get_max_create_time(self, max_wait_time=None, *, default=30):
        """
        Retrieve the maximum time to wait for trainee to become available.

        The value is determined by (in this order):
        1. An explicitly set `max_wait_time` parameter value;
        2. An environment variable `HOWSO_CLIENT_CREATE_MAX_WAIT_TIME;
        3. A setting in the config.yaml file:
           Howso > options > create_max_wait_time; or
        4. the final default of 30 seconds.

        Parameters
        ----------
        max_wait_time : int, optional
            A user provided parameter value to use.

        Returns
        -------
        int or None
            The maximum time to wait in seconds. Or None for no max wait time.
        """
        # 1: Explicitly setting the `max_wait_time` parameter
        if max_wait_time is None:
            sentinel = object()

            # 2: ENV Variable named HOWSO_CLIENT_CREATE_MAX_WAIT_TIME
            env_value = os.environ.get(ENV_CREATE_MAX_WAIT_TIME,
                                       default=sentinel)
            if env_value != sentinel:
                try:
                    max_wait_time = float(env_value)
                except (TypeError, ValueError):
                    # Any value that cannot be converted to a Float will be
                    # interpreted as `None`, which means to wait indefinitely.
                    max_wait_time = None
            else:

                # 3: config.yaml option
                config_value = self.get_user_config_option(
                    'Howso', 'options', 'create_max_wait_time',
                    default=sentinel)
                if config_value != sentinel:
                    try:
                        max_wait_time = float(config_value)
                    except (TypeError, ValueError):
                        # Any value that cannot be converted to a Float will
                        # be interpreted as `None`, which means to
                        # wait indefinitely.
                        max_wait_time = None
                else:
                    # 4. Final default
                    max_wait_time = default
        if max_wait_time == 0:
            # Zero signifies no maximum wait time
            max_wait_time = None
        return max_wait_time
