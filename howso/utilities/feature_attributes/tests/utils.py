from contextlib import suppress
from tempfile import TemporaryDirectory


class TemporaryDirectoryIgnoreErrors(TemporaryDirectory):
    """
    Override to fix a known issue with TemporaryDirectory that can cause cleanup errors.

    There are fixes in Python's >= 3.12, but until then, just use this.

    Once 3.12 is the minimum version, this can be removed if the parameter `ignore_cleanup_errors`
    is set to `True` in the TemporaryDirectory constructor.
    """

    def cleanup(self):
        """
        Override cleanup to suppress any exceptions that may occur during cleanup.

        This is a workaround for known issues with TemporaryDirectory cleanup.
        """
        with suppress(Exception):
            super().cleanup()
