from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import pandas as pd
from pandas import DataFrame, Index

from howso.client.client import get_howso_client_class
from howso.client.schemas.reaction import Reaction
from howso.utilities import deserialize_cases, format_dataframe
from howso.utilities.internals import deserialize_to_dataframe


class HowsoPandasClientMixin:
    """
    Overrides Howso client methods to return data as pandas data types.

    Base: :class:`howso.client.AbstractHowsoClient`
    """

    def get_session_indices(self, *args, **kwargs) -> Index:
        """
        Base: :func:`howso.client.AbstractHowsoClient.get_session_indices`.

        Returns
        -------
        Index
            An index of the session indices for the requested session.
        """
        indices = super().get_session_indices(*args, **kwargs)
        if isinstance(indices, list):
            return pd.Index(indices, dtype='int64')
        return pd.Index([], dtype='int64')

    def get_session_training_indices(self, *args, **kwargs) -> Index:
        """
        Base: :func:`howso.client.AbstractHowsoClient.get_session_training_indices`.

        Returns
        -------
        Index
            An index of the session training indices for the requested session.
        """
        indices = super().get_session_training_indices(*args, **kwargs)
        if isinstance(indices, list):
            return pd.Index(indices, dtype='int64')
        return pd.Index([], dtype='int64')

    def react_group(self, *args, **kwargs) -> DataFrame:
        """
        Base: :func:`howso.client.AbstractHowsoClient.react_group`.

        Returns
        -------
        DataFrame
            A DataFrame of feature name columns to the conviction of grouped
            cases rows.
        """
        cases = super().react_group(*args, **kwargs)
        return deserialize_to_dataframe(cases)

    def get_cases(self, trainee_id: str, *args, **kwargs) -> DataFrame:
        """
        Base: :func:`howso.client.AbstractHowsoClient.get_cases`.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to get cases from.

        Returns
        -------
        DataFrame
            A DataFrame of feature name columns and case rows.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        feature_attributes = self.resolve_feature_attributes(trainee_id)
        response = super().get_cases(trainee_id, *args, **kwargs)
        return deserialize_cases(
            response['cases'],
            response['features'],
            feature_attributes
        )

    def get_extreme_cases(
        self,
        trainee_id: str,
        num: int,
        sort_feature: str,
        features: Optional[Iterable[str]] = None
    ) -> DataFrame:
        """
        Base: :func:`howso.client.AbstractHowsoClient.get_extreme_cases`.

        Parameters
        ----------
        trainee_id : str
            The ID of the Trainee to retrieve extreme cases from.
        num : int
            The number of cases to get.
        sort_feature : str
            The feature name by which extreme cases are sorted by.
        features: iterable of str, optional
            An iterable of feature names to use when getting extreme cases.

        Returns
        -------
        DataFrame
            A DataFrame of feature name columns and extreme case rows.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        feature_attributes = self.resolve_feature_attributes(trainee_id)
        response = super().get_extreme_cases(trainee_id, num, sort_feature, features)
        return deserialize_cases(
            response['cases'],
            response['features'],
            feature_attributes
        )

    def get_feature_conviction(self, *args, **kwargs) -> DataFrame:
        """
        Base: :func:`howso.client.AbstractHowsoClient.get_feature_conviction`.

        Returns
        -------
        DataFrame
            A DataFrame containing the familiarity conviction rows to feature
            columns.
        """
        response = super().get_feature_conviction(*args, **kwargs)
        index = []
        rows = []
        if response.get('familiarity_conviction_addition'):
            index.append('familiarity_conviction_addition')
            rows.append(response['familiarity_conviction_addition'])
        if response.get('familiarity_conviction_removal'):
            index.append('familiarity_conviction_removal')
            rows.append(response['familiarity_conviction_removal'])
        return deserialize_to_dataframe(rows, index=index)

    def react_aggregate(self, *args, **kwargs) -> DataFrame:
        """
        Base: :func:`howso.client.AbstractHowsoClient.react_aggregate`.

        Returns
        -------
        DataFrame
            A DataFrame of feature name columns to statistic value rows.
        """
        response = super().react_aggregate(*args, **kwargs)
        return pd.DataFrame(response)

    def get_marginal_stats(self, *args, **kwargs) -> DataFrame:
        """
        Base: :func:`howso.client.AbstractHowsoClient.get_marginal_stats`.

        Returns
        -------
        DataFrame
            A DataFrame of feature name columns to statistic value rows.
        """
        response = super().get_marginal_stats(*args, **kwargs)
        return pd.DataFrame(response)

    def react_series(
        self,
        trainee_id: str,
        *args,
        series_index: str = '.series',
        **kwargs
    ) -> Reaction:
        """
        Base: :func:`howso.client.AbstractHowsoClient.react_series`.

        Parameters
        ----------
        trainee_id : str
            The trainee id.
        series_index : str, default ".series"
            When set to a string, will include the series index as a
            column in the returned DataFrame using the column name given.
            If set to None, no column will be added.

        Returns
        -------
        Reaction:
            A MutableMapping (dict-like) with these keys -> values:
                action -> pandas.DataFrame
                    A data frame of action values.

                details -> dict or list
                    An aggregated list of any requested details.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        feature_attributes = self.resolve_feature_attributes(trainee_id)
        response = super().react_series(trainee_id, *args, series_index=series_index, **kwargs)
        response['action'] = format_dataframe(response.get("action"), feature_attributes)
        return response

    def react(self, trainee_id, *args, **kwargs) -> Reaction:
        """
        Base: :func:`howso.client.AbstractHowsoClient.react`.

        Returns
        -------
        Reaction:
            A MutableMapping (dict-like) with these keys -> values:
                action -> pandas.DataFrame
                    A data frame of action values.

                details -> dict or list
                    An aggregated list of any requested details.
        """
        trainee_id = self._resolve_trainee(trainee_id).id
        feature_attributes = self.resolve_feature_attributes(trainee_id)
        response = super().react(trainee_id, *args, **kwargs)
        columns = response['details'].get('action_features')
        if 'prediction_stats' in response['details']:
            response['details']['prediction_stats'] = pd.DataFrame(response['details']['prediction_stats'][0]).T
        response['action'] = deserialize_cases(response['action'], columns, feature_attributes)
        return response


def get_howso_pandas_client(**kwargs):
    """
    Return the appropriate AbstractHowsoClient subclass based on config.

    This is a "factory function" that, based on the given parameters, will
    decide which AbstractHowsoClient derivative to instantiate and return
    using the Pandas client mixin.

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
        An instantiated subclass of AbstractHowsoClient constructed with
        the HowsoPandasClientMixin.
    """
    client_class, client_params = get_howso_client_class(**kwargs)

    # Construct the client class from the requested base client class and
    # the pandas mixin
    class HowsoPandasClient(HowsoPandasClientMixin, client_class):  # noqa
        pass

    client_params.update(kwargs)
    return HowsoPandasClient(**client_params)  # noqa


HowsoPandasClient = get_howso_pandas_client
