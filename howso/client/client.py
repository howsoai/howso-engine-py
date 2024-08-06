"""Primary Howso Client Class."""
from __future__ import annotations

from collections.abc import Generator, Sequence
from importlib import import_module
from os import environ
from os.path import expandvars
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import warnings

import yaml

import howso.client.base
from howso.client.exceptions import HowsoConfigurationError
from howso.utilities.utilities import deep_update, UserFriendlyExit

if TYPE_CHECKING:
    from howso.client.base import AbstractHowsoClient


DEFAULT_CONFIG_FILE = "howso.yml"
DEFAULT_CONFIG_FILE_ALT = "howso.yaml"
CONFIG_FILE_ENV_VAR = "HOWSO_CONFIG"
HOME_DIR_CONFIG_PATH = ".howso"
HOWSO_CONFIG_DOCS = "https://docs.howso.com/getting_started/client_configuration.html"  # noqa
LEGACY_CONFIG_FILENAMES = [
    "diveplane.yml", "diveplane.yaml",
    "config.yml", "config.yaml",
]
XDG_DIR_CONFIG_PATH = "howso"
XDG_CONFIG_ENV_VAR = "XDG_CONFIG_HOME"


def _check_isfile(file_paths: Sequence[Path | str]) -> Path | None:
    """
    Check if any of the given paths are files, returning the first one found.

    Parameters
    ----------
    file_paths: Sequence of Path or str
        A sequence of file paths (as Path or str)

    Returns
    -------
    Path or None:
        The first file_path in the given iterable that passes the `isfile`
        check.
    """
    for file_path in file_paths:
        file_path = Path(file_path)
        if file_path.is_file():
            return file_path

    return None


def get_configuration_path(config_path: Optional[Path | str] = None,  # noqa: C901
                           verbose: bool = False) -> Path | None:
    """
    Determine where the configuration is stored, if anywhere.

    If config_path is None, None will be returned.

    If a config_path is that is non-None, it will be processed as a YAML file,
    if the file does not exist at the provided path or there are parse errors,
    an exception will be raised.

    Parameters
    ----------
    config_path : str or None
        The given config_path.
    verbose : bool
        If True provides more verbose messaging. Default is false.

    Returns
    -------
    Path
        The found config_path or None

    Raises
    ------
    HowsoConfigurationError
        Raised if a `config_path` is provided but points to a non-existent
        file or the file is un-parsable as a YAML file.
    """
    if config_path is None:
        user_dir = Path().home()
        xdg_config_home_not_abs_msg = (
            'The path set in the XDG_CONFIG_HOME environment variable'
            'is not absolute: "{0}". The specification for XDG_CONFIG_HOME '
            'variables requires the value to be an absolute path.'.format(
                environ.get(XDG_CONFIG_ENV_VAR)
            ))

        # Check if HOWSO_CONFIG env variable is set
        if environ.get(CONFIG_FILE_ENV_VAR) is not None:
            config_path = environ[CONFIG_FILE_ENV_VAR]
            if not Path(config_path).is_file():
                raise HowsoConfigurationError(
                    'The environment variable "{0}" was found, but it does '
                    'not point to Howso configuration '
                    'file.'.format(CONFIG_FILE_ENV_VAR))
            elif verbose:
                print(CONFIG_FILE_ENV_VAR + ' set to ' + config_path)
        # Check current working directory for howso.yml file
        elif Path(DEFAULT_CONFIG_FILE).is_file():
            config_path = DEFAULT_CONFIG_FILE
        # Falling back to howso.yaml file
        elif Path(DEFAULT_CONFIG_FILE_ALT).is_file():
            config_path = DEFAULT_CONFIG_FILE_ALT
        # Falling back to config.yml file or other legacy names
        elif config_path := _check_isfile(LEGACY_CONFIG_FILENAMES):
            warnings.warn(
                f'Deprecated use of "{config_path}" file. '
                f'Please rename to "{DEFAULT_CONFIG_FILE}".')
        # Check for .yml config file in XDG_CONFIG_HOME directory, if configured
        elif (
            environ.get(XDG_CONFIG_ENV_VAR) is not None and
            Path(environ[XDG_CONFIG_ENV_VAR], XDG_DIR_CONFIG_PATH, DEFAULT_CONFIG_FILE).is_file()
        ):
            # Check if XDG_CONFIG_HOME is an absolute path.
            if not Path(expandvars(environ[XDG_CONFIG_ENV_VAR])).is_absolute():
                raise HowsoConfigurationError(xdg_config_home_not_abs_msg)
            config_path = Path(environ[XDG_CONFIG_ENV_VAR], XDG_DIR_CONFIG_PATH, DEFAULT_CONFIG_FILE)
        # Check for .yaml config file in XDG_CONFIG_HOME directory, if configured
        elif (
            environ.get(XDG_CONFIG_ENV_VAR) is not None and
            Path(environ[XDG_CONFIG_ENV_VAR], XDG_DIR_CONFIG_PATH, DEFAULT_CONFIG_FILE_ALT).is_file()
        ):
            # Check if XDG_CONFIG_HOME is an absolute path.
            if not Path(environ[XDG_CONFIG_ENV_VAR]).expanduser().is_absolute():
                raise HowsoConfigurationError(xdg_config_home_not_abs_msg)
            config_path = Path(environ[XDG_CONFIG_ENV_VAR], XDG_DIR_CONFIG_PATH, DEFAULT_CONFIG_FILE_ALT)
        # Check default home directory for config file
        elif Path(user_dir, HOME_DIR_CONFIG_PATH, DEFAULT_CONFIG_FILE).is_file():  # noqa
            config_path = Path(user_dir, HOME_DIR_CONFIG_PATH, DEFAULT_CONFIG_FILE)
        # falling back to howso.yaml file
        elif Path(user_dir, HOME_DIR_CONFIG_PATH, DEFAULT_CONFIG_FILE_ALT).is_file():  # noqa
            config_path = Path(user_dir, HOME_DIR_CONFIG_PATH, DEFAULT_CONFIG_FILE_ALT)
        # falling back to legacy filenames
        elif config_path := _check_isfile([
            Path(user_dir, HOME_DIR_CONFIG_PATH, file)
            for file in LEGACY_CONFIG_FILENAMES
        ]):
            warnings.warn(
                f'Use of deprecated configuration file name at "{config_path}". '
                f'Please rename to "{DEFAULT_CONFIG_FILE}".')

    # Verify file in config_path parameter exists
    elif config_path and not Path(config_path).is_file():
        raise HowsoConfigurationError(
            "Specified configuration file was not found. Verify that the "
            "location of your configuration file matches the config parameter "
            "used when instantiating the client.")

    if verbose:
        if config_path:
            print(f'Using configuration at path: {config_path}')
        else:
            print('Using configuration-less defaults.')

    if config_path:
        return Path(config_path)
    return None


def _gen_files_in_dir(directory: Path) -> Generator[Path, None, None]:
    """
    Recursively yield file (not directory) paths from the given `directory`.

    Parameters
    ----------
    directory : str or Path
        The directory to look within.

    Yields
    ------
    Path
        Paths to files within the given `directory` at any depth.
    """
    for node in directory.iterdir():
        if node.is_dir():
            yield from _gen_files_in_dir(node)
        else:
            yield node


def get_extras_configs(directory: Optional[Path | str] = None) -> dict:
    """
    Accumulate and return any "extra" config found in resources or other path.

    Parameters
    ----------
    directory : str or Path, default None
        Optional. The directory to look within. This is given as the directory
        containing the `howso.yml` et al file. If not provided will default to
        a 'resources' directory, if found, in the howso namespace.

    Returns
    -------
    dict
        A dictionary containing a merged set of extras configurations found in
        the given `directory` or the default, which is <howso>/resources/.
    """
    if not directory:
        directory = Path(__file__).parent.parent.joinpath('resources')
    if not isinstance(directory, Path):
        directory = Path(directory)

    extras_config = {}

    extras_stems = ['extras', 'Extras', 'EXTRAS']
    extras_exts = ['.yml', '.yaml', '.YML', '.YAML']
    for node in _gen_files_in_dir(directory):
        if node.suffix in extras_exts and node.stem in extras_stems:
            try:
                with open(node, 'r') as config:
                    config_data = yaml.safe_load(config)
            except Exception:  # noqa: Deliberately broad
                raise
            else:
                extras_config = deep_update(extras_config, config_data)

    return extras_config


def get_howso_client_class(**kwargs) -> tuple[type, dict]:  # noqa: C901
    """
    Return the appropriate AbstractHowsoClient subclass based on config.

    This is a "factory function" that, based on the given parameters, will
    decide which AbstractHowsoClient derivative to return.

    In the event that no configuration is found that indicates which client
    class to use, the HowsoDirectClient will be returned.

    Parameters
    ----------
    kwargs : dict
        config_path: str or None
            The path to a valid configuration file, or None.
        verbose : bool
            If True provides more verbose messaging. Default false.
        Any other kwargs. These will be passed to the client constructor along
        with `config_path` and `verbose`.

    Returns
    -------
    AbstractHowsoClient
        A resolved subclass of AbstractHowsoClient.
    dict
        Client extra kwargs.
    """
    config_path = kwargs.get('config_path', None)
    verbose = kwargs.get('verbose', False)

    kind_exit = UserFriendlyExit(verbose=verbose)
    config_data = None

    # Attempt to load and parse config.yaml.
    config_path = get_configuration_path(config_path, verbose) or ""
    if config_path:
        try:
            with open(config_path, 'r') as config:
                config_data = yaml.safe_load(config)
        except TypeError:
            # There is no config_path (None), which is OK.
            config_data = dict()
        except yaml.YAMLError as yaml_exception:
            kind_exit(f'Unable to parse the configuration file located at '
                      f'"{config_path}". Please verify the YAML syntax of '
                      f'this file and try again.', exception=yaml_exception)
        except (IOError, OSError) as exception:
            kind_exit(f'Error reading the configuration file located at '
                      f'"{config_path}". Check the file permissions and try '
                      f'again.', exception=exception)
        else:
            # Lowercase top-level `howso` key.
            if 'Howso' in config_data and 'howso' not in config_data:
                config_data['howso'] = config_data['Howso']
                del config_data['Howso']
    else:
        config_data = {}

    client_class = None

    # Check if the configuration file `config.yaml` contains the item
    # 'client' that is a valid, dotted-path to another sub-class of
    # AbstractHowsoClient. If so, instantiate that one and return it. This
    # provides an opportunity for customer-specific functionality and/or
    # authentication schemes, etc.
    try:
        custom_client = config_data['howso']['client']  # type: ignore
        # Split the dotted-path into "module" and the specific "class". For
        # example. `my_package.my_module.MyClass' would become
        # `custom_module_path` of `my_package.my_module` and
        # `custom_class_name` becomes `MyClass`.
        custom_module_path, custom_class_name = custom_client.rsplit('.', 1)
        # Attempt to load the module itself.
        custom_module = import_module(custom_module_path)
        # Set `client_class` to the actual class provided at the end of the
        # dotted-path.
        client_class = getattr(custom_module, custom_class_name)
        # Ensure that the `client_class` is a subclass of
        # AbstractHowsoClient.
        if not issubclass(client_class, howso.client.base.AbstractHowsoClient):
            raise HowsoConfigurationError(
                'The provided client_class must be a subclass '
                'of AbstractHowsoClient.')
    except KeyError:
        # Looks like no attempt was made to override the default client class.
        # By passing here, we'll determine a default class to return
        pass
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as exception:
        # User attempted to override the default client class, but there was
        # an error.
        kind_exit('The configuration at howso -> client, if provided, '
                  'should contain a valid dotted-path to a '
                  'subclass of AbstractHowsoClient.', exception=exception)
    except HowsoConfigurationError as exception:
        # User provided a dotted-path to a class, but it's not a subclass of
        # the AbstractHowsoClient
        kind_exit('The client configured in Howso -> client is not a '
                  'valid subclass of AbstractHowsoClient.',
                  exception=exception)

    # Determine default client if one is not set by the config
    if client_class is None:
        # Otherwise use the direct client
        from howso.direct import HowsoDirectClient
        client_class = HowsoDirectClient

    # customer-specific functionality and/or authentication schemes, etc.
    try:
        client_extra_params = config_data['howso']['client_extra_params']  # type: ignore
    except KeyError:
        # No extra params set - that is ok - let's move on
        client_extra_params = dict()

    if client_extra_params is None:
        client_extra_params = dict()
    elif not isinstance(client_extra_params, dict):
        kind_exit('The configuration at howso -> client_extra_params '
                  'should be defined as a dictionary.')

    # Add any "Extras"
    try:
        extras_config = get_extras_configs()
        client_extra_params = deep_update(
            client_extra_params,
            extras_config['howso']['client_extra_params']
        )
    except Exception:  # noqa: Deliberately broad
        pass

    if verbose:
        print("Instantiating %r" % client_class)

    if not config_path:
        # Warn when running with out a configuration.
        no_config_msg_verbose_only = client_extra_params.pop(
            "no_config_msg_verbose_only", True)
        if verbose or not no_config_msg_verbose_only:
            no_config_msg = client_extra_params.pop(
                "no_config_msg",
                f"No configuration file was found. Operating with "
                f"HowsoDirectClient and default parameters. To learn more "
                f"about 'howso.yml' configuration files, see documentation "
                f"at: {HOWSO_CONFIG_DOCS}.\n"
            )
            print(no_config_msg)

    return client_class, client_extra_params


def get_howso_client(**kwargs) -> AbstractHowsoClient:
    """
    Return the appropriate AbstractHowsoClient subclass based on config.

    This is a "factory function" that, based on the given parameters, will
    decide which AbstractHowsoClient derivative to instantiate and return.

    Parameters
    ----------
    config_path: str or None, optional
        The path to a valid configuration file, or None
    verbose : bool, optional
        If True provides more verbose messaging. Default is false.
    kwargs : dict
        Additional client arguments. These will be passed to the client
        constructor along with `config_path` and `verbose`.

    Returns
    -------
    AbstractHowsoClient
        An instantiated subclass of AbstractHowsoClient.
    """
    client_class, client_params = get_howso_client_class(**kwargs)
    client_params.update(kwargs)
    return client_class(**client_params)


# For backwards compatibility, let this factory function assume the default
# client class name.
HowsoClient = get_howso_client
