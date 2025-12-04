import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

import wakatools  # Make sure "waka" accessor is available


@pytest.fixture
def invalid_dataarray():
    return xr.DataArray([1, 2], coords={"invalid": [0, 1]}, dims=["invalid"])


class TestDataArrayAccessor:
    @pytest.mark.unittest
    def test_accessor_exists(self, bathymetry_grid):
        assert hasattr(bathymetry_grid, "waka")

    @pytest.mark.unittest
    def test_accessor_invalid_dataarray(self, invalid_dataarray):
        with pytest.raises(
            ValueError, match="DataArray must have 'x' and 'y' dimensions."
        ):
            invalid_dataarray.waka

    @pytest.mark.unittest
    def test_grid_coordinates(self, bathymetry_grid):
        coords = bathymetry_grid.waka.grid_coordinates()
        assert isinstance(coords, np.ndarray)
        assert_array_almost_equal(
            coords,
            [
                [0.5, 4.5],
                [1.5, 4.5],
                [2.5, 4.5],
                [3.5, 4.5],
                [4.5, 4.5],
                [0.5, 3.5],
                [1.5, 3.5],
                [2.5, 3.5],
                [3.5, 3.5],
                [4.5, 3.5],
                [0.5, 2.5],
                [1.5, 2.5],
                [2.5, 2.5],
                [3.5, 2.5],
                [4.5, 2.5],
                [0.5, 1.5],
                [1.5, 1.5],
                [2.5, 1.5],
                [3.5, 1.5],
                [4.5, 1.5],
                [0.5, 0.5],
                [1.5, 0.5],
                [2.5, 0.5],
                [3.5, 0.5],
                [4.5, 0.5],
            ],
        )
