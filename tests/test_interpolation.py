import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

import wakatools as waka


@pytest.mark.unittest
def test_tin_surface(xyz_dataframe, bathymetry_grid):
    tin = waka.interpolation.tin_surface(xyz_dataframe, "z", bathymetry_grid)
    assert isinstance(tin, xr.DataArray)
    assert tin.dims == bathymetry_grid.dims
    assert_array_almost_equal(tin["x"], bathymetry_grid["x"])
    assert_array_almost_equal(tin["y"], bathymetry_grid["y"])
    assert_array_almost_equal(
        tin,
        [
            [np.nan, np.nan, np.nan, 4.27831633, 3.54617347],
            [1.86190476, 4.22426036, 2.25976331, 1.78345588, 3.48566176],
            [1.99462366, 3.52857143, 2.07810651, 1.84813278, np.nan],
            [2.28494624, 3.21268657, 0.6, 1.72119205, np.nan],
            [np.nan, 2.20425532, 2.3177305, np.nan, np.nan],
        ],
    )


@pytest.mark.unittest
def test_griddata(xyz_dataframe, bathymetry_grid):
    griddata = waka.interpolation.griddata(
        xyz_dataframe, "z", bathymetry_grid, method="linear"
    )
    assert isinstance(griddata, xr.DataArray)
    assert griddata.dims == bathymetry_grid.dims
    assert_array_almost_equal(griddata["x"], bathymetry_grid["x"])
    assert_array_almost_equal(griddata["y"], bathymetry_grid["y"])
    assert_array_almost_equal(
        griddata,
        [
            [np.nan, np.nan, np.nan, 4.27831633, 3.54617347],
            [1.86190476, 4.22426036, 2.25976331, 1.78345588, 3.48566176],
            [1.99462366, 3.52857143, 2.07810651, 1.84813278, np.nan],
            [2.28494624, 3.21268657, 0.6, 1.72119205, np.nan],
            [np.nan, 2.20425532, 2.3177305, np.nan, np.nan],
        ],
    )


@pytest.mark.unittest
def test_rbf(xyz_dataframe, bathymetry_grid):
    rbf = waka.interpolation.rbf(xyz_dataframe, "z", bathymetry_grid, method="linear")
    assert isinstance(rbf, xr.DataArray)
    assert rbf.dims == bathymetry_grid.dims
    assert_array_almost_equal(rbf["x"], bathymetry_grid["x"])
    assert_array_almost_equal(rbf["y"], bathymetry_grid["y"])
    assert_array_almost_equal(
        rbf,
        [
            [3.2605582, 4.267619, 3.34530481, 3.07298889, 3.74846097],
            [1.99674917, 3.60153486, 2.34114118, 1.85475464, 3.30451605],
            [2.50224382, 3.47892979, 2.123843, 1.76075331, 3.00202682],
            [2.51108057, 2.0606107, 0.66357299, 1.43890494, 2.46777589],
            [2.65463457, 1.64966538, 1.03867453, 1.7032501, 2.35897279],
        ],
    )
