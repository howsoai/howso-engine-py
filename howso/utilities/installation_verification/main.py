from __future__ import annotations

from functools import partial
import sys
import warnings

try:
    from howso import engine
except ImportError:
    engine = None

try:
    from howso.validator import Validator  # noqa: might not be available # type: ignore[reportMissingImports]
except OSError as e:
    Validator = e
except ImportError:
    Validator = None

try:
    from howso.synthesizer import Synthesizer  # noqa: might not be available # type: ignore[reportMissingImports]
except ImportError:
    Synthesizer = None

from ._output import is_databricks, iv_print
from .checks_cpu import check_visible_cores
from .checks_engine import (
    check_engine_operation,
    check_generate_dataframe,
    check_latency,
    check_performance,
    check_save,
    overridable_resources,
)
from .checks_environment import check_locales_available, check_tzdata_installed
from .checks_optional import check_basic_synthesis, check_synthesizer_create_delete, check_validator_operation
from .registry import InstallationCheckRegistry


def configure(registry: InstallationCheckRegistry) -> None:
    """
    Register the correct checks for the install environment.

    Parameters
    ----------
    registry : InstallationCheckRegistry
        The InstallationCheckRegistry instance.
    """
    registry.add_check(
        name="Howso Client: Visible CPUs",
        fn=check_visible_cores,
        client_required="AbstractHowsoClient",
    )

    registry.add_check(
        name="Howso Local: Timezone support",
        fn=check_tzdata_installed,
        client_required="HowsoDirectClient",
    )

    registry.add_check(
        name="Howso Client: Basic react",
        fn=partial(check_generate_dataframe, threshold=10.0),
        client_required="HowsoPlatformClient",
    )

    registry.add_check(
        name="Howso Local: Basic react",
        fn=partial(check_generate_dataframe, threshold=1.0),
        client_required="HowsoDirectClient",
    )

    registry.add_check(
        name="Howso Client: Network latency",
        fn=partial(check_latency, notice_threshold=25,
                   warning_threshold=20),
        client_required="HowsoPlatformClient",
    )

    registry.add_check(
        name="Howso Client: System performance",
        fn=partial(check_performance, num_samples=2_000,
                   notice_threshold=15.0, warning_threshold=20.0),
        client_required="HowsoPlatformClient",
    )

    registry.add_check(
        name="Howso Client: Overridable resources",
        fn=partial(overridable_resources),
        client_required="HowsoPlatformClient",
    )

    registry.add_check(
        name="Howso Local: System performance",
        fn=partial(check_performance, num_samples=5_000,
                   notice_threshold=10.0, warning_threshold=20.0),
        client_required="HowsoDirectClient",
    )

    registry.add_check(
        name="Howso Local: Save Trainee",
        fn=check_save,
        client_required="HowsoDirectClient",
    )

    registry.add_check(
        name="Howso Engine™: Basic operations",
        fn=check_engine_operation,
        client_required="AbstractHowsoClient",
        other_requirements=[engine],
    )

    registry.add_check(
        name="Howso Synthesizer™: Supported system locales",
        fn=check_locales_available,
        client_required="AbstractHowsoClient",
        other_requirements=[Synthesizer],
    )

    registry.add_check(
        name="Howso Synthesizer: Basic operations",
        fn=partial(check_synthesizer_create_delete),
        client_required="AbstractHowsoClient",
        other_requirements=[Synthesizer],
    )

    registry.add_check(
        name="Howso Synthesizer: Basic synthesis",
        fn=check_basic_synthesis,
        client_required="AbstractHowsoClient",
        other_requirements=[Synthesizer],
    )

    registry.add_check(
        name="Howso Validator: Basic operations",
        fn=check_validator_operation,
        client_required="AbstractHowsoClient",
        other_requirements=[Validator],
    )


def main() -> None:
    """Primary entry point."""
    iv_print("[bold]Validating Howso® Installation")
    registry = InstallationCheckRegistry()

    with warnings.catch_warnings():
        configure(registry)
        warnings.simplefilter("ignore")
        result = registry.run_checks()
        if is_databricks():
            return
        sys.exit(result)
