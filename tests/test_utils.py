import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from wakatools.utils import conversion, scaling


@pytest.mark.parametrize(
    "min_, max_, expected",
    [
        (None, None, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        (-1, 7, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]),
        (1.5, 2.5, [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]),
    ],
    ids=["default", "wider_range", "inner_range"],
)
def test_scale(min_, max_, expected):
    array = np.array([0, 1, 2, 3, 4, 5])
    scaled = scaling.scale(array, min_=min_, max_=max_)
    assert_array_almost_equal(scaled, expected)


@pytest.mark.unittest
def test_add_depth_column(seismic_data):
    depth = conversion.calculate_depth(seismic_data)
    assert isinstance(depth, pd.Series)
    assert_array_almost_equal(
        depth,
        [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.32, 0.32],
    )

    # Test on single lines
    depth = conversion.calculate_depth(seismic_data[seismic_data["ID"] == "line1"])
    assert_array_almost_equal(
        depth,
        [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8],
    )
