import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from wakatools.utils import conversion
from wakatools.utils.spatial import target_grid_from


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


# @pytest.mark.parametrize(
#         "value, resolution",
#         ([1.5, 0.5 ],
#          [1.6, 0.5],
#          [1.4, 0.5],
#          )
# )
# def test_round_to_upper(value, resolution)
# # def test_target_grid_from(xyz_dataframe, resolution):
# #     grid = target_grid_from(xyz_dataframe, resolution=resolution)
