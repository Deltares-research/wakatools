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
    df_with_depth = utils.add_depth_column(seismic_data)
    assert "depth" in df_with_depth.columns
    expected_depth = pd.Series([np.nan, np.nan, np.nan, np.nan, 0.8, 0.8, 0.8])
    assert_array_equal(df_with_depth["depth"], expected_depth)
    assert df_with_depth["depth"].dtype == float
    assert len(df_with_depth) == len(seismic_data)
