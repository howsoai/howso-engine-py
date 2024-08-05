from __future__ import annotations

from collections.abc import Generator, Mapping
import typing as t

from requests import JSONDecodeError as RequestsJSONDecodeError, Response
from typing_extensions import NotRequired, TypeAliasType, TypedDict


class ValidationErrorDetail(TypedDict):
    """Representation of a single validation error object."""

    message: str
    field: NotRequired[list[str]]
    code: NotRequired[str | None]


ValidationErrorCollection = TypeAliasType(
    "ValidationErrorCollection",
    t.Union[list[ValidationErrorDetail], dict[str, "ValidationErrorCollection"]]
)
"""A collection of validation error objects."""


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

    def __init__(self, message: str, code: t.Optional[str] = None, url: t.Optional[str] = None):
        """Initialize a HowsoError."""
        if code is None:
            code = "0"
        if url is None:
            url = "about:blank"
        self.message = message
        self.code = code
        self.url = url
        super().__init__((message, code))


class HowsoValidationError(HowsoError):
    """
    An error raised when parameters are invalid.

    Parameters
    ----------
    message : str
        The error message.
    code : str, optional
        The error code.
    errors : Mapping of ValidationErrorCollection, optional
        Map of parameters to error messages.
    url : str, optional
        An absolute URI that identifies the problem type.
    """

    def __init__(
        self,
        message: str,
        *,
        code: t.Optional[str] = None,
        errors: t.Optional[Mapping[str, ValidationErrorCollection]] = None,
        url: t.Optional[str] = None,
    ):
        self.errors = errors
        super().__init__(message, code=code, url=url)

    def messages(self) -> list[str]:
        """Get validation error messages for each field."""
        messages = []
        for error in self.iter_errors():
            msg = error['message']
            if field := error.get('field'):
                msg = f"{'.'.join(field)}: {msg}"
            messages.append(msg)
        return messages

    def iter_errors(self) -> Generator[ValidationErrorDetail]:
        """Iterate over field error messages."""
        def _traverse(path: list[str], collection: Mapping | list):
            if isinstance(collection, Mapping):
                for key, item in collection.items():
                    yield from _traverse([*path, key], item)
            elif isinstance(collection, list):
                for item in collection:
                    error = ValidationErrorDetail(
                        message=item.get('message') or 'An unknown error occurred.',
                        field=path
                    )
                    if code := item.get('code'):
                        error['code'] = code
                    yield error

        if self.errors is not None:
            yield from _traverse([], self.errors)


class HowsoApiError(HowsoError):
    """
    An error raised by the Howso rest API.

    Parameters
    ----------
    message : str
        The error message.
    code : str, optional
        The problem type code.
    errors : Mapping of ValidationErrorCollection, optional
        Map of parameters to error messages.
    status : int, optional
        The HTTP status code.
    url : str, optional
        An absolute URI that identifies the problem type.
    """

    status = None

    def __init__(
        self,
        message: str,
        *,
        code: t.Optional[str] = None,
        errors: t.Optional[Mapping[str, ValidationErrorCollection]] = None,
        status: t.Optional[int] = None,
        url: t.Optional[str] = None,
    ):
        """Initialize a HowsoApiError."""
        if status is None:
            status = -1
        self.status = status
        self.errors = errors
        super().__init__(message, code=code, url=url)

    @classmethod
    def from_dict(cls, obj: Mapping | None):
        """
        Build a HowsoApiError from API response.

        Parameters
        ----------
        obj : Mapping
            The error information.

        Returns
        -------
        HowsoApiError
            The constructed error instance.
        """
        if obj is None:
            obj = {}

        default_msg = 'An unknown error occurred.'
        detail = obj.get('detail') or default_msg
        status = obj.get('status')
        code = obj.get('code')
        url = obj.get('type')
        errors = obj.get('errors')

        if isinstance(detail, list):
            # This helper can only process a single message at a time, use the first value
            detail = detail[0] if len(detail) > 0 else default_msg

        return cls(detail, code=code, status=status, url=url, errors=errors)

    @classmethod
    def from_response(cls, obj: Response):
        """Build HowsoApiError from Response object."""
        status = obj.status_code
        default_msg = 'An unknown error occurred.'
        try:
            data = obj.json()
            message = data.get('detail') or default_msg
            code = data.get('code')
            url = data.get('type')
            errors = data.get('errors')
        except (RequestsJSONDecodeError, TypeError, AttributeError):
            message = default_msg
            code = None
            url = None
            errors = None

        if isinstance(message, list):
            # This helper can only process a single message at a time, use the first value
            message = message[0] if len(message) > 0 else default_msg

        return cls(message, code=code, status=status, url=url, errors=errors)


class HowsoApiValidationError(HowsoValidationError, HowsoApiError):
    """An error raised when parameters are invalid."""

    def __init__(
        self,
        message: str,
        *,
        code: t.Optional[str] = None,
        errors: t.Optional[Mapping[str, ValidationErrorCollection]] = None,
        status: t.Optional[int] = 400,
        url: t.Optional[str] = None,
    ):
        super(HowsoValidationError, self).__init__(message=message, code=code, status=status, errors=errors, url=url)


class HowsoAuthenticationError(HowsoApiError):
    """An error raised due to an authentication failure."""


class HowsoConfigurationError(HowsoError):
    """An error raised when the howso.yml options are misconfigured."""


class HowsoNotUniqueError(HowsoError):
    """An error raised when an attempt to rename a trainee is unsuccessful."""


class HowsoTimeoutError(HowsoError):
    """An error raised when an operation times out."""


class HowsoWarning(UserWarning):
    """A warning raised from core output."""


class UnsupportedArgumentWarning(HowsoWarning):
    """Warning for when unsupported arguments are supplied to an endpoint."""
