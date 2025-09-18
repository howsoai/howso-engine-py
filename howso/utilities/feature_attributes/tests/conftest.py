from __future__ import annotations

from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mongomock
import pytest

try:
    from howso.connectors.abstract_data import (
        convert_data,
        DaskDataFrameData,
        DataFrameData,
        make_data_source,
        MongoDBData,
        ParquetDataFile,
        ParquetDataset,
        SQLTableData,
        TabularFile,
    )
except (ModuleNotFoundError, ImportError):
    pass


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


def mongodb_data(df) -> Iterator[MongoDBData]:
    """Yield a MongoDBData instance populated with data from the given DataFrame."""
    with patch("howso.connectors.abstract_data.mongodb_data.MongoClient", new=mongomock.MongoClient):
        adc = MongoDBData("mongodb://localhost/test_db#test_collection")
        # Populate the mocked MongoDB with the provided DataFrame
        if len(df) != 0:
            adc._collection.insert_many(df.to_dict(orient="records"))
        yield adc


def sqltable_data(df) -> Iterator[SQLTableData]:
    """Yield a SQLTableData instance populated with the data from the given DataFrame."""
    with TemporaryDirectoryIgnoreErrors() as temp_dir:
        destination = make_data_source(f"sqlite:///{temp_dir}/db.sqlite#main.data")
        convert_data(DataFrameData(df), destination)
        yield destination


def dask_dataframe_data(df) -> Iterator[DaskDataFrameData]:
    """Yield a DaskDataFrameData instance populated with the data from the given DataFrame."""
    import dask.dataframe as dd
    yield make_data_source(dd.from_pandas(df, npartitions=3))


def pd_dataframe_data(df) -> Iterator[DataFrameData]:
    """Yield a DataFrameData instance populated with the data from the given DataFrame."""
    yield make_data_source(df)


def parquet_datafile(df) -> Iterator[ParquetDataFile]:
    """Yield a ParquetDataFile instance populated with the data from the given DataFrame."""
    with TemporaryDirectoryIgnoreErrors() as temp_dir:
        destination = make_data_source(Path(f"{temp_dir}/data.parquet"))
        convert_data(make_data_source(df), destination)
        yield destination


def parquet_dataset(df) -> Iterator[ParquetDataset]:
    """Yield a ParquetDataFile instance populated with the data from the given DataFrame."""
    with TemporaryDirectoryIgnoreErrors() as temp_dir:
        destination = make_data_source(Path(f"{temp_dir}/"))
        convert_data(make_data_source(df), destination)
        yield destination


def tabular_file(df) -> Iterator[TabularFile]:
    """Yield a TabularFile instance populated with the data from the given DataFrame."""
    with TemporaryDirectoryIgnoreErrors() as temp_dir:
        destination = make_data_source(Path(f"{temp_dir}/data.tsv"))
        convert_data(make_data_source(df), destination)
        yield destination


@pytest.fixture
def adc(request):
    """Produce an ADC populated with arbitrary data given a parametrized tuple."""
    adc_type, df = request.param
    if adc_type == "MongoDBData":
        adc_gen = mongodb_data(df)
    elif adc_type == "SQLTableData":
        adc_gen = sqltable_data(df)
    elif adc_type == "ParquetDataFile":
        adc_gen = parquet_datafile(df)
    elif adc_type == "TabularFile":
        adc_gen = tabular_file(df)
    elif adc_type == "DaskDataFrameData":
        adc_gen = dask_dataframe_data(df)
    else:  # Pandas DataFrame
        adc_gen = pd_dataframe_data(df)
    yield from adc_gen
