import pandas as pd

from factors import get_registered_factors
from leak_detection.data import DEFAULT_DATA_PATH, describe_dataset, load_sample_factors


def test_sample_dataset_exists():
    assert DEFAULT_DATA_PATH.exists(), "Expected simulated dataset to be present"


def test_sample_dataset_columns_and_order():
    df = load_sample_factors()
    expected = ["date", "symbol", "open", "high", "low", "close", "volume", "ret_1d"]
    assert list(df.columns) == expected
    sorted_df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df, sorted_df)


def test_descriptor_summary():
    df = load_sample_factors()
    descriptor = describe_dataset(df)
    assert descriptor.rows == len(df)
    assert descriptor.symbols, "should list symbols"
    assert descriptor.start_date <= descriptor.end_date


def test_registered_factors_align_with_dataset():
    df = load_sample_factors()
    factors = get_registered_factors()
    for name, definition in factors.items():
        series = definition.function(df.copy())
        assert len(series) == len(df), f"{name} output length mismatch"
        pd.testing.assert_index_equal(series.index, df.index)
