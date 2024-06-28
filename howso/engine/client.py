from howso.client import AbstractHowsoClient, HowsoPandasClient
from howso.client.pandas import HowsoPandasClientMixin

__client_instance = None

__all__ = [
    'get_client',
    'use_client',
]


def get_client() -> AbstractHowsoClient:
    """
    Get the active Howso client instance.

    Returns
    -------
    HowsoPandasClient
        The active client.
    """
    global __client_instance
    if __client_instance is None:
        __client_instance = HowsoPandasClient()
    return __client_instance


def use_client(client: AbstractHowsoClient) -> None:
    """
    Set the active Howso client instance to use for the API.

    Parameters
    ----------
    client : AbstractHowsoClient
        The client instance.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        When the client is not an instance of AbstractHowsoClient.
    """
    global __client_instance
    if not isinstance(client, AbstractHowsoClient):
        raise ValueError("`client` must be a subclass of "
                         "AbstractHowsoClient")
    if not isinstance(client, HowsoPandasClientMixin):
        raise ValueError("`client` must be a HowsoPandasClient")
    __client_instance = client
