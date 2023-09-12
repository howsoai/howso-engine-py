import json

from howso.openapi.exceptions import UnauthorizedException


class HowsoError(Exception):
    """
    An exception raised when Howso encounters an error.

    Parameters
    ----------
    message : str
        The error message.
    code : str, optional
        The error code.
    url : str, optional
        An absolute URI that identifies the problem type.
    """

    message = None
    code = None
    url = None

    def __init__(self, message, code=None, url=None):
        """Initialize a HowsoError."""
        if code is None:
            code = "0"
        if url is None:
            url = "about:blank"
        self.message = message
        self.code = code
        self.url = url
        super().__init__((message, code, url))


class HowsoConfigurationError(HowsoError):
    """An error raised when the howso.yml options are misconfigured."""


class HowsoApiError(HowsoError):
    """
    An error raised by the Howso rest API.

    Parameters
    ----------
    message : str
        The error message.
    code : str, optional
        The problem type code.
    status : int, optional
        The HTTP status code.
    url : str, optional
        An absolute URI that identifies the problem type.
    """

    status = None

    def __init__(self, message, code=None, status=None, url=None):
        """Initialize a HowsoApiError."""
        if status is None:
            status = -1
        self.status = status
        super().__init__(message, code, url)

    @classmethod
    def from_openapi(cls, obj):
        """
        Build a HowsoApiError from OpenAPI error object.

        Parameters
        ----------
        obj : ApiException
            The OpenAPI error.

        Returns
        -------
        HowsoApiError
            The constructed error instance.
        """
        return cls.from_json(obj.body)

    @classmethod
    def from_json(cls, obj):
        """
        Build a HowsoApiError from API response json.

        Parameters
        ----------
        obj : str
            A json string.

        Returns
        -------
        HowsoApiError
            The constructed error instance.
        """
        return cls.from_dict(json.loads(obj))

    @classmethod
    def from_dict(cls, obj):
        """
        Build a HowsoApiError from API response.

        Parameters
        ----------
        obj : dict
            The error information.

        Returns
        -------
        HowsoApiError
            The constructed error instance.
        """
        if obj is None:
            obj = {}

        title = obj.get('title', '')
        detail = obj.get('detail', '')
        status = obj.get('status')
        code = obj.get('code')
        url = obj.get('type')

        # Build message string
        if not title and not detail:
            title = 'An unknown error occurred.'
        message = f"{title}".strip()
        if title and detail:
            if message[-1] not in [':', '.', ',', ';']:
                message += ":"
            message += " "
        message += f"{detail}".strip()

        return cls(message, code, status, url)


class HowsoAuthenticationError(HowsoApiError):
    """An error raised when the authentication API request fails."""

    @classmethod
    def from_openapi(cls, obj):
        """
        Build a HowsoApiError from OpenAPI error object.

        Parameters
        ----------
        obj : ApiException
            The OpenAPI error.

        Returns
        -------
        HowsoApiError
            The constructed error instance.
        """
        if isinstance(obj, UnauthorizedException):
            try:
                body = json.loads(obj.body)
                error = body.get('error', obj.status)
            except Exception:
                error = obj.status
            return cls(f'{obj.reason}: {error}', status=obj.status)
        else:
            return super().from_openapi(obj)


class HowsoNotUniqueError(HowsoError):
    """An error raised when an attempt to rename a trainee is unsuccessful."""


class HowsoTimeoutError(HowsoError):
    """An error raised when an operation times out."""


class HowsoWarning(UserWarning):
    """A warning raised from core output."""
