from __future__ import annotations

import math
import time
import traceback
from typing import Any, TYPE_CHECKING

import pandas as pd

from howso.client import AbstractHowsoClient
from howso.client.exceptions import HowsoError
from howso.client.schemas import Trainee
from howso.utilities import infer_feature_attributes, Timer

try:
    from howso import engine
except ImportError:
    engine = None

from ._helpers import _is_platform_client, get_nonce
from ._types import Status

if TYPE_CHECKING:
    from .registry import InstallationCheckRegistry


def generate_dataframe(*, client: AbstractHowsoClient,
                       num_samples: int = 150,
                       timeout: int | None = None
                       ) -> tuple[pd.DataFrame, float | int]:
    """
    Use HowsoClient to create a dataframe of random data.

    Parameters
    ----------
    client : AbstractHowsoClient
        The Howso client instance to use.
    num_samples : int, default 150
        The number of samples to synthesize.
    timeout : int or None, default None
        Optional. If provided, `num_samples` is ignored and synthesis happens
        1 record at a time until `timeout` seconds have elapsed.

    Returns
    -------
    pd.DataFrame
        A dataframe of the synthesized records.

    """
    continuous_feature = {
        "type": "continuous",
        "decimal_places": 2,
        "bounds": {
            "min": 0.0,
            "max": 100.0,
            "allow_null": False,
        }
    }
    features = {
        "alpha": continuous_feature,
        "beta": continuous_feature,
        "gamma": continuous_feature,
        "class": {
            "type": "nominal",
            "bounds": {
                "allowed": ["apple", "banana", "cherry"],
                "allow_null": False},
        }
    }
    feature_names = list(features.keys())

    trainee = client.create_trainee(
        name=f"installation_verification generated dataframe ({get_nonce()})",
        features=features,
        persistence="allow" if _is_platform_client(client) else "never"
    )
    if not isinstance(trainee, Trainee):
        raise HowsoError("Unable to create trainee.")
    try:
        client.set_feature_attributes(trainee.id, features)
        client.acquire_trainee_resources(trainee.id, max_wait_time=0)
        action: pd.DataFrame | list[list[Any]]
        if timeout:
            # Generate 1 case at a time until `timeout` has passed.
            deadline = time.monotonic() + timeout
            action = []
            while time.monotonic() < deadline:
                if reaction := client.react(
                    trainee.id, action_features=feature_names,
                    num_cases_to_generate=1, desired_conviction=1.0,
                    generate_new_cases="no", suppress_warning=True
                ):
                    action.append(reaction["action"].iloc[0].tolist())
            elapsed_time = timeout
        else:
            with Timer() as timer:
                reaction = client.react(
                    trainee.id, action_features=feature_names,
                    num_cases_to_generate=num_samples, desired_conviction=1.0,
                    generate_new_cases="no", suppress_warning=True
                )
            action = reaction["action"] if reaction else []
            elapsed_time = timer.seconds or math.nan
    finally:
        client.delete_trainee(trainee.id)
    df = pd.DataFrame(action, columns=feature_names)
    return df, elapsed_time


def check_generate_dataframe(*, registry: InstallationCheckRegistry,
                             threshold: float | None = None) -> tuple[Status, str]:
    """
    Rate the speed in which a dataframe was able to be generated.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    threshold : float or None, default None
        Optional. If provided determines how long the process can run before
        considering it to return a status of WARNING.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    try:
        _, duration = generate_dataframe(client=registry.client,
                                         num_samples=150)
    except ValueError:
        traceback.print_exc(file=registry.logger)
        return (
            Status.CRITICAL,
            (
                "The client was unable to find Howso core binaries. "
                "Please see the Howso Client installation documentation "
                "for further details."
            )
        )
    if threshold is not None and duration > threshold:
        return (
            Status.WARNING,
            (
                f"The client required a duration of {duration:,.1f} to "
                f"synthesize a DataFrame, this should require no more than "
                f"{threshold:,.1f} seconds. This warning may be expected in "
                f"auto-scaling installations."
            )
        )
    return (Status.OK, "")


def check_save(*, registry: InstallationCheckRegistry,
               source_df: pd.DataFrame | None = None) -> tuple[Status, str]:
    """
    Ensure that a Trainee can can be saved.

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
    client = trainee = None
    try:
        client = registry.client
        if source_df is None:
            source_df, _ = generate_dataframe(client=client)
        features = infer_feature_attributes(source_df)
        feature_names = list(features.keys())
        if trainee := client.create_trainee(
            name=f"installation_verification check save ({get_nonce()})",
            features=features
        ):
            client.train(trainee.id, source_df, features=feature_names)
            client.persist_trainee(trainee.id)
        else:
            raise HowsoError("Could not create a trainee.")  # noqa: TRY301
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL,
                "Could not save Trainee. Please check file permissions.")
    else:
        return (Status.OK, "")
    finally:
        try:
            if client and trainee:
                client.delete_trainee(trainee.id)
        except Exception:  # noqa: BLE001, S110
            pass


def check_latency(*, registry: InstallationCheckRegistry,
                  source_df: pd.DataFrame | None = None,
                  notice_threshold: int = 25, warning_threshold: int = 20,
                  timeout: int = 10) -> tuple[Status, str]:
    """
    Ensure creation of `sample_threshold` requests within `timeout` seconds.

    # This test uses a deliberately inefficient manner to synthesize records
    # and is # done this way to expose network latency issues.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    source_df : pd.DataFrame or None, default None
        Optional. If not provided a new dataframe will be synthesized.
    notice_threshold : int, default 25
        The number of samples that should be generated within `timeout`
        seconds. If it cannot generate this number within the timeout, then the
        resulting Status will be NOTICE.
    warning_threshold : int, default 20
        The number of samples that should be generated within `timeout`
        seconds. If it cannot generate this number within the timeout, then the
        resulting Status will be WARNING.
    timeout : int, default 10
        The number of seconds to run synthesis, one sample at a time.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    try:
        if source_df is None:
            source_df, _ = generate_dataframe(client=registry.client,
                                              timeout=timeout)
        num_rows = source_df.shape[0]
        if num_rows < warning_threshold:
            return (
                Status.WARNING,
                (
                    f"{num_rows} records synthesized in {timeout:,d} seconds. "
                    f"A minimum of {warning_threshold:,d} samples expected. "
                    "Ensure a good network connection and that Howso "
                    "Platform is installed on sufficient cluster hardware. "
                    "In auto-scaling installations this may be due to slow node "
                    "start-ups."
                )
            )
        if num_rows < notice_threshold:
            return (
                Status.NOTICE,
                (
                    f"{num_rows} records synthesized in {timeout:,d} seconds. "
                    f"Less than {notice_threshold:,d} may indicate "
                    "a poor network connection or slow/oversubscribed "
                    "cluster hardware. This notice is expected in auto-scaling "
                    "installations."
                )
            )
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL,
                "Could not complete check. Check installation.")
    else:
        return (Status.OK, "")


def check_performance(*, registry: InstallationCheckRegistry,
                      num_samples: int = 5_000, notice_threshold: float = 10.0,
                      warning_threshold: float = 20.0) -> tuple[Status, str]:
    """
    Ensure can generate `num_samples` records with `time_threshold` seconds.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    num_samples : int, default 5,000
        The number of samples to generate.
    notice_threshold : float, default 10.0
        The notice time-threshold in seconds. If the generation of `num_samples`
        requires more than `threshold` seconds, the returned Status will be
        NOTICE.
    warning_threshold : float, default 20.0
        The warning time-threshold in seconds. If the generation of `num_samples`
        requires more than `threshold` seconds, the returned Status will be
        WARNING.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    try:
        _, num_seconds = generate_dataframe(client=registry.client,
                                            num_samples=num_samples)
        msg = (
            f"{num_samples:,d} records were synthesized in "
            f"{num_seconds:,.1f} seconds. "
        )
        if num_seconds > warning_threshold:
            return (
                Status.WARNING,
                (
                    msg + f" This should require fewer than {warning_threshold:,.1f} "
                    "seconds. Ensure the installation is on equipment that meets "
                    "Howso's recommended hardware specifications. "
                    "In auto-scaling installations this may be due to slow node "
                    "start-ups."
                )
            )
        if num_seconds > notice_threshold:
            return (
                Status.NOTICE,
                (
                    msg + f" Greater than {notice_threshold:,.1f} seconds may indicate "
                    "slow or underpowered hardware. Ensure the installation is on "
                    "equipment that meets Howso's recommended specifications. "
                    "This notice is expected in auto-scaling installations."
                )
            )
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (
            Status.CRITICAL,
            "Could not complete operation. Check installation."
        )
    else:
        return (Status.OK, "")


def overridable_resources(*, registry: InstallationCheckRegistry) -> tuple[Status, str]:
    """
    Ensure that resources are overridable for Platform workers.

    registry : The InstallationCheckRegistry
        The registry used to run this check.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    # Create a trainee and acquire its resources.
    try:
        if engine:
            with engine.Trainee(
                name=f"installation_verification overridable resources ({get_nonce()})",
                persistence="never",
                runtime={"scaling": {"resources": {"cpu": {"minimum": 1_500}}}},
            ) as trainee:
                runtime = trainee.get_runtime()
                try:
                    cpu_limits = runtime["scaling"]["resources"]["cpu"]  # type: ignore[reportOptionalSubscript]
                    if cpu_limits["minimum"] != 1_500:  # type: ignore[reportOptionalSubscript]
                        raise AssertionError("Incorrect value returned")  # noqa: TRY301
                except (KeyError, TypeError, ValueError):
                    traceback.print_exc(file=registry.logger)
                    return (Status.CRITICAL, "get_runtime() returned an unexpected response.")
                except AssertionError:
                    traceback.print_exc(file=registry.logger)
                    return (Status.CRITICAL, "get_runtime() returned an unexpected value for minimum CPU cores.")
                else:
                    return (Status.OK, "")
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL, "Could not create a trainee.")

    # Reached only if `engine` could not be imported.
    return (Status.CRITICAL, "Howso Engine™ is not installed.")


def check_engine_operation(
    *,
    registry: InstallationCheckRegistry,
    source_df: pd.DataFrame | None = None
) -> tuple[Status, str]:
    """
    Ensure that Howso Engine operates as it should.

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
    trainee = None
    try:
        if source_df is None:
            source_df, _ = generate_dataframe(client=registry.client,
                                              num_samples=150)

        features = infer_feature_attributes(source_df)

        train_idx = source_df.sample(frac=0.8).index
        df_train = source_df.loc[source_df.index.isin(train_idx)]
        df_test = source_df.loc[~source_df.index.isin(train_idx)]
        x_train = df_train.drop("class", axis=1)
        y_train = df_train["class"]
        x_test = df_test.drop("class", axis=1)

        action_features = ["class"]
        context_features = x_train.columns.tolist()
        if not engine:
            raise AssertionError("Howso Engine™ is not installed.")  # noqa: TRY301
        trainee = engine.Trainee(
            name=(f"installation_verification "
                  f"check engine operations ({get_nonce()})"),
            features=features, overwrite_existing=True
        )
        trainee.train(x_train.join(y_train))
        trainee.analyze()
        response = trainee.react(x_test, context_features=context_features,
                                 action_features=action_features)
        results = response["action"][action_features]
        if results.shape[0] != x_test.shape[0]:
            return (
                Status.ERROR,
                (
                    "Results do not have the same number of samples as the "
                    "input data."
                )
            )
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL,
                "Could not complete operation. Check installation.")
    else:
        return (Status.OK, "")
    finally:
        try:
            if engine and trainee:
                engine.delete_trainee(trainee.id)
        except Exception:  # noqa: BLE001, S110
            pass
