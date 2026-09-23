from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

import pandas as pd

from howso.utilities import infer_feature_attributes

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

from ._output import iv_print
from ._types import Status
from .checks_engine import generate_dataframe

if TYPE_CHECKING:
    from .registry import InstallationCheckRegistry


def check_basic_synthesis(*, registry: InstallationCheckRegistry,
                          source_df: pd.DataFrame | None = None) -> tuple[Status, str]:
    """
    Validate that Synthesizer can perform a basic synthesis.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    source_df : pd.DataFrame or None, default None
        Optional. If not provided a new dataframe will be synthesized.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    try:
        # Using a made-up dataframe, synthesize a new one from it.
        if source_df is None:
            source_df, _ = generate_dataframe(client=registry.client)

        features = infer_feature_attributes(source_df)
        if not Synthesizer:
            raise AssertionError("Howso Synthesizer™ is not installed.")  # noqa: TRY301
        with Synthesizer(client=registry.client, privacy_override=True) as s:
            s.train(source_df, features)
            synthesized_df = s.synthesize_cases(n_samples=100)
        if synthesized_df.shape != (100, 4):
            return (Status.CRITICAL, "Synthetic dataframe is the wrong shape.")
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL,
                "Could not complete check. Check installation.")
    else:
        return (Status.OK, "")


def check_synthesizer_create_delete(*, registry: InstallationCheckRegistry,
                                    source_df: pd.DataFrame | None = None
                                    ) -> tuple[Status, str]:
    """
    Ensure that a Trainee can can be created and deleted.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    source_df : pd.DataFrame or None, default None
        Optional. If not provided a new dataframe will be synthesized.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
    """
    s = None
    try:
        if source_df is None:
            source_df, _ = generate_dataframe(client=registry.client)

        features = infer_feature_attributes(source_df)
        if not Synthesizer:
            raise AssertionError("Howso Synthesizer™ is not installed.")  # noqa: TRY301
        s = Synthesizer(client=registry.client, privacy_override=True)

        s.train(source_df[:50], features)
        n = s.cl.get_num_training_cases(s.trainee.id)
        if n != 50:
            return (
                Status.ERROR,
                (
                    f"Training did not produce the correct number of "
                    f"training cases ({n}). Howso Synthesizer might not be "
                    "installed correctly. "
                    "Try: `pip install --upgrade howso-synthesizer`."
                )
            )

        s.cl.delete_trainee(s.trainee.id)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (
            Status.CRITICAL,
            (
                "Could not complete check. Check installation. "
                "Try: `pip install --upgrade howso-synthesizer`."
            )
        )
    else:
        return (Status.OK, "")
    finally:
        try:
            if s:
                s.cl.delete_trainee(s.trainee.id)
        except Exception:  # noqa: BLE001, S110
            pass


def check_validator_operation(
    *, registry: InstallationCheckRegistry,
    source_df: pd.DataFrame | None = None,
) -> tuple[Status, str]:
    """
    Ensure that Validator-Enterprise operates as it should.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    source_df : pd.DataFrame or None, default None
        Optional. If not provided a new dataframe will be synthesized.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    if isinstance(Validator, Exception):
        iv_print(Validator, file=registry.logger)
        return (
            Status.CRITICAL,
            (
                "Howso Validator™ was not installed correctly. "
                "Please check installation."
            )
        )
    try:
        if source_df is None:
            source_df, _ = generate_dataframe(client=registry.client, num_samples=150)

        orig_df = source_df.sample(frac=0.5)
        gen_df = source_df[~source_df.index.isin(orig_df.index)]
        features = infer_feature_attributes(orig_df)
        if not Validator:
            raise AssertionError("Howso Validator™ is not installed.")  # noqa: TRY301

        with Validator(orig_df, gen_df, features=features, verbose=-1) as v:
            result = v.run_metric("DescriptiveStatistics")

        if result.desirability == 0 or len(result.errors):
            return (Status.CRITICAL, "Validator encountered one or more errors.")

    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL,
                "Could not complete operation. Check installation.")
    else:
        return (Status.OK, "")
