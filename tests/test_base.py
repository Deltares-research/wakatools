import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

import wakatools  # Make sure "waka" accessor is available


@pytest.fixture
def invalid_dataarray():
    return xr.DataArray([1, 2], coords={"invalid": [0, 1]}, dims=["invalid"])


@pytest.fixture
def invalid_dataframe():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


class TestDataFrameAccessor:
    @pytest.mark.unittest
    def test_accessor_exists(self, xyz_dataframe):
        assert hasattr(xyz_dataframe, "waka")

    @pytest.mark.unittest
    def test_accessor_invalid_dataframe(self, invalid_dataframe):
        with pytest.raises(
            ValueError, match="DataFrame must have 'x' and 'y' columns."
        ):
            invalid_dataframe.waka

    @pytest.mark.unittest
    def test_bounds(self, xyz_dataframe):
        bounds = xyz_dataframe.waka.bounds()
        assert bounds == (0.3, 0.2, 4.9, 4.8)

    @pytest.mark.unittest
    def test_coordinates(self, xyz_dataframe):
        coords = xyz_dataframe.waka.coordinates()
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (10, 2)
        assert_array_almost_equal(
            coords,
            [
                [0.3, 3.6],
                [1.8, 2.1],
                [2.7, 1.7],
                [4.9, 4.8],
                [0.6, 0.2],
                [3.1, 3.4],
                [4.4, 2.9],
                [2.0, 1.3],
                [1.2, 4.1],
                [3.8, 0.7],
            ],
        )

    @pytest.mark.unittest
    def test_coordinates_scaled(self, xyz_dataframe):
        coords = xyz_dataframe.waka.coordinates_scaled()
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (10, 2)
        assert_array_almost_equal(
            coords,
            [
                [0.0, 0.73913043],
                [0.32608696, 0.41304348],
                [0.52173913, 0.32608696],
                [1.0, 1.0],
                [0.06521739, 0.0],
                [0.60869565, 0.69565217],
                [0.89130435, 0.58695652],
                [0.36956522, 0.23913043],
                [0.19565217, 0.84782609],
                [0.76086957, 0.10869565],
            ],
        )

        coords = xyz_dataframe.waka.coordinates_scaled(bbox=(0, 0, 5, 5))
        assert_array_almost_equal(
            coords,
            [
                [0.06, 0.72],
                [0.36, 0.42],
                [0.54, 0.34],
                [0.98, 0.96],
                [0.12, 0.04],
                [0.62, 0.68],
                [0.88, 0.58],
                [0.4, 0.26],
                [0.24, 0.82],
                [0.76, 0.14],
            ],
        )

    @pytest.mark.unittest
    def test_get_raster_values(self, xyz_dataframe, bathymetry_grid):
        values = xyz_dataframe.waka.get_raster_values(bathymetry_grid)
        assert isinstance(values, np.ndarray)
        assert values.shape == (10,)
        assert_array_almost_equal(
            values,
            [0.1, 0.3, 0.5, 0.4, 0.4, 0.4, 0.6, 0.5, 0.1, 0.7],
        )


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

    def test_grid_coordinates_scaled(self, bathymetry_grid):
        coords = bathymetry_grid.waka.grid_coordinates_scaled()
        assert isinstance(coords, np.ndarray)
        assert_array_almost_equal(
            coords,
            [
                [0.1, 0.9],
                [0.3, 0.9],
                [0.5, 0.9],
                [0.7, 0.9],
                [0.9, 0.9],
                [0.1, 0.7],
                [0.3, 0.7],
                [0.5, 0.7],
                [0.7, 0.7],
                [0.9, 0.7],
                [0.1, 0.5],
                [0.3, 0.5],
                [0.5, 0.5],
                [0.7, 0.5],
                [0.9, 0.5],
                [0.1, 0.3],
                [0.3, 0.3],
                [0.5, 0.3],
                [0.7, 0.3],
                [0.9, 0.3],
                [0.1, 0.1],
                [0.3, 0.1],
                [0.5, 0.1],
                [0.7, 0.1],
                [0.9, 0.1],
            ],
        )

        coords = bathymetry_grid.waka.grid_coordinates_scaled(bbox=(2, 2, 4, 4))
        assert_array_almost_equal(
            coords,
            [
                [-0.75, 1.25],
                [-0.25, 1.25],
                [0.25, 1.25],
                [0.75, 1.25],
                [1.25, 1.25],
                [-0.75, 0.75],
                [-0.25, 0.75],
                [0.25, 0.75],
                [0.75, 0.75],
                [1.25, 0.75],
                [-0.75, 0.25],
                [-0.25, 0.25],
                [0.25, 0.25],
                [0.75, 0.25],
                [1.25, 0.25],
                [-0.75, -0.25],
                [-0.25, -0.25],
                [0.25, -0.25],
                [0.75, -0.25],
                [1.25, -0.25],
                [-0.75, -0.75],
                [-0.25, -0.75],
                [0.25, -0.75],
                [0.75, -0.75],
                [1.25, -0.75],
            ],
        )
