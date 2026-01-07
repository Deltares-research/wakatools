import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

import wakatools as waka


@pytest.fixture
def extra_xyz_points():
    """
    Extra xyz points as GeoDataFrame for testing. Also tests interpolation works when
    input is a GeoDataFrame instead of a Pandas DataFrame.

    """
    x = [0.75, 2.6, 4.6]
    y = [1.9, 4.38, 1.5]
    return gpd.GeoDataFrame(
        {"x": x, "y": y, "z": [2.0, 4.0, 1.0]}, geometry=gpd.points_from_xy(x, y)
    )


@pytest.mark.unittest
def test_tin_surface(xyz_dataframe, bathymetry_grid):
    result = waka.interpolation.tin_surface(
        xyz_dataframe, value="z", target_grid=bathymetry_grid
    )
    assert isinstance(result, xr.DataArray)
    assert result.dims == bathymetry_grid.dims
    assert_array_almost_equal(result["x"], bathymetry_grid["x"])
    assert_array_almost_equal(result["y"], bathymetry_grid["y"])
    assert_array_almost_equal(
        result,
        [
            [np.nan, np.nan, np.nan, 4.27831633, 3.54617347],
            [1.86190476, 4.22426036, 2.25976331, 1.78345588, 3.48566176],
            [1.99462366, 3.52857143, 2.07810651, 1.84813278, np.nan],
            [2.28494624, 3.21268657, 0.6, 1.72119205, np.nan],
            [np.nan, 2.20425532, 2.3177305, np.nan, np.nan],
        ],
    )


@pytest.mark.unittest
def test_tin_surface_multiple_inputs(xyz_dataframe, extra_xyz_points, bathymetry_grid):
    result = waka.interpolation.tin_surface(
        xyz_dataframe, extra_xyz_points, value="z", target_grid=bathymetry_grid
    )
    assert isinstance(result, xr.DataArray)
    assert result.dims == bathymetry_grid.dims
    assert_array_almost_equal(result["x"], bathymetry_grid["x"])
    assert_array_almost_equal(result["y"], bathymetry_grid["y"])
    assert_array_almost_equal(
        result,
        [
            [np.nan, np.nan, np.nan, 3.95641234, 3.54163961],
            [1.86190476, 4.22426036, 2.25976331, 1.78345588, 3.48566176],
            [1.81862745, 3.52857143, 2.07810651, 1.84813278, 2.51944444],
            [2.25980392, 1.0537246, 0.6, 0.92142857, 0.99285714],
            [np.nan, 2.20425532, 2.3177305, np.nan, np.nan],
        ],
    )


@pytest.mark.unittest
def test_griddata(xyz_dataframe, bathymetry_grid):
    result = waka.interpolation.griddata(
        xyz_dataframe, value="z", target_grid=bathymetry_grid, method="linear"
    )
    assert isinstance(result, xr.DataArray)
    assert result.dims == bathymetry_grid.dims
    assert_array_almost_equal(result["x"], bathymetry_grid["x"])
    assert_array_almost_equal(result["y"], bathymetry_grid["y"])
    assert_array_almost_equal(
        result,
        [
            [np.nan, np.nan, np.nan, 4.27831633, 3.54617347],
            [1.86190476, 4.22426036, 2.25976331, 1.78345588, 3.48566176],
            [1.99462366, 3.52857143, 2.07810651, 1.84813278, np.nan],
            [2.28494624, 3.21268657, 0.6, 1.72119205, np.nan],
            [np.nan, 2.20425532, 2.3177305, np.nan, np.nan],
        ],
    )


@pytest.mark.unittest
def test_griddata_multiple_inputs(xyz_dataframe, extra_xyz_points, bathymetry_grid):
    result = waka.interpolation.griddata(
        xyz_dataframe,
        extra_xyz_points,
        value="z",
        target_grid=bathymetry_grid,
        method="linear",
    )
    assert isinstance(result, xr.DataArray)
    assert result.dims == bathymetry_grid.dims
    assert_array_almost_equal(result["x"], bathymetry_grid["x"])
    assert_array_almost_equal(result["y"], bathymetry_grid["y"])
    assert_array_almost_equal(
        result,
        [
            [np.nan, np.nan, np.nan, 3.95641234, 3.54163961],
            [1.86190476, 4.22426036, 2.25976331, 1.78345588, 3.48566176],
            [1.81862745, 3.52857143, 2.07810651, 1.84813278, 2.51944444],
            [2.25980392, 1.0537246, 0.6, 0.92142857, 0.99285714],
            [np.nan, 2.20425532, 2.3177305, np.nan, np.nan],
        ],
    )


@pytest.mark.unittest
def test_rbf(xyz_dataframe, bathymetry_grid):
    result = waka.interpolation.rbf(
        xyz_dataframe,
        value="z",
        target_grid=bathymetry_grid,
        kernel="thin_plate_spline",
    )
    assert isinstance(result, xr.DataArray)
    assert result.dims == bathymetry_grid.dims
    assert_array_almost_equal(result["x"], bathymetry_grid["x"])
    assert_array_almost_equal(result["y"], bathymetry_grid["y"])
    assert_array_almost_equal(
        result,
        [
            [2.94248917, 4.74939923, 3.45931706, 2.6264665, 3.54428058],
            [2.05023821, 4.34627561, 2.46904129, 1.38290294, 3.34335636],
            [2.6933896, 4.34772865, 2.33047468, 1.44382587, 3.40129971],
            [2.70351887, 2.17604895, 0.37018807, 1.16146539, 3.1474354],
            [2.74546424, 1.2038024, 0.22702653, 1.51034794, 3.28929487],
        ],
    )


@pytest.mark.unittest
def test_rbf_multiple_inputs(xyz_dataframe, extra_xyz_points, bathymetry_grid):
    result = waka.interpolation.rbf(
        xyz_dataframe,
        extra_xyz_points,
        value="z",
        target_grid=bathymetry_grid,
        kernel="thin_plate_spline",
    )
    assert isinstance(result, xr.DataArray)
    assert result.dims == bathymetry_grid.dims
    assert_array_almost_equal(result["x"], bathymetry_grid["x"])
    assert_array_almost_equal(result["y"], bathymetry_grid["y"])
    assert_array_almost_equal(
        result,
        [
            [3.24991721, 5.10403698, 4.33130934, 3.35601807, 3.82001932],
            [1.9442857, 4.24262351, 2.58360578, 1.5681149, 3.54221398],
            [1.84571354, 4.101627, 2.31777791, 1.22968938, 2.67595636],
            [1.66651923, 1.845768, 0.44394348, 0.81182356, 0.98649738],
            [2.48076876, 1.31605174, 0.72302727, 1.90346097, 2.05052316],
        ],
    )
