import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from wakatools import utils


@pytest.mark.unittest
def test_time_to_depth(seismic_data):
    depth = utils.time_to_depth(seismic_data)
    assert_array_equal(
        depth, pd.Series([np.nan, np.nan, np.nan, np.nan, 0.8, 0.8, 0.8])
    )
    assert depth.dtype == float
    assert len(depth) == len(seismic_data)
    assert depth.isnull().sum() == 4


@pytest.mark.unittest
def test_add_depth_column(seismic_data):
    # Test on single lines
    df_with_depth = utils.add_depth_column(
        seismic_data[seismic_data["ID"] == "line1"].copy()
    )
    assert df_with_depth["ID"].nunique() == 1
    expected_depth = pd.Series(
        [
            np.np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.8,
            0.8,
            0.8,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.31999999999999945,
            0.3200000000000002,
        ]
    )
    assert_array_equal(df_with_depth["depth"], expected_depth)
    # Test on multiple lines
    df_with_depth = utils.add_depth_column(seismic_data)
    assert "depth" in df_with_depth.columns  # onnodig
    expected_depth = pd.Series([np.nan, np.nan, np.nan, np.nan, 0.8, 0.8, 0.8])
    assert_array_equal(df_with_depth["depth"], expected_depth)
    assert len(df_with_depth) == len(seismic_data)
