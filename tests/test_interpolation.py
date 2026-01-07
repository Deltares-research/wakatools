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
        {"x": x, "y": y, "z": [-0.8, -1.1, -0.1]}, geometry=gpd.points_from_xy(x, y)
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
            [np.nan, np.nan, np.nan, -0.92397959, -0.74540816],
            [-0.92857143, -1.16153846, -1.00769231, -0.80588235, -0.51176471],
            [-0.78387097, -0.92857143, -0.99230769, -0.84522822, np.nan],
            [-0.65483871, -0.72537313, -0.87368421, -0.65364238, np.nan],
            [np.nan, -0.40319149, -0.32163121, np.nan, np.nan],
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
            [np.nan, np.nan, np.nan, -0.9387987, -0.74561688],
            [-0.92857143, -1.16153846, -1.00769231, -0.80588235, -0.51176471],
            [-0.80490196, -0.92857143, -0.99230769, -0.84522822, -0.30833333],
            [-0.65784314, -0.49187359, -0.87368421, -0.74166667, -0.15833333],
            [np.nan, -0.40319149, -0.32163121, np.nan, np.nan],
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
            [np.nan, np.nan, np.nan, -0.92397959, -0.74540816],
            [-0.92857143, -1.16153846, -1.00769231, -0.80588235, -0.51176471],
            [-0.78387097, -0.92857143, -0.99230769, -0.84522822, np.nan],
            [-0.65483871, -0.72537313, -0.87368421, -0.65364238, np.nan],
            [np.nan, -0.40319149, -0.32163121, np.nan, np.nan],
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
            [np.nan, np.nan, np.nan, -0.9387987, -0.74561688],
            [-0.92857143, -1.16153846, -1.00769231, -0.80588235, -0.51176471],
            [-0.80490196, -0.92857143, -0.99230769, -0.84522822, -0.30833333],
            [-0.65784314, -0.49187359, -0.87368421, -0.74166667, -0.15833333],
            [np.nan, -0.40319149, -0.32163121, np.nan, np.nan],
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
            [-1.24679101, -1.37123147, -1.19358764, -0.92044931, -0.69413756],
            [-0.92302158, -1.17919863, -1.10170068, -0.76607969, -0.43201774],
            [-0.71444697, -0.94733604, -1.27297183, -0.98268999, -0.36354366],
            [-0.51283852, -0.36865029, -0.97112513, -0.88971181, -0.24710307],
            [-0.51152566, -0.14064553, -0.18279145, -0.17774019, 0.15233631],
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
            [-1.21043934, -1.34758147, -1.14429245, -0.89672649, -0.69497869],
            [-0.93829858, -1.19224435, -1.08839383, -0.77108689, -0.44019728],
            [-0.88698985, -0.99354556, -1.25900461, -0.98143547, -0.33635813],
            [-0.72350365, -0.43125777, -0.96721311, -0.8842504, -0.16400419],
            [-0.56029485, -0.1429627, -0.17260795, -0.18272512, 0.20799584],
        ],
    )
