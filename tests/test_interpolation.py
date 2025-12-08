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
    rbf = waka.interpolation.rbf(xyz_dataframe, "z", bathymetry_grid)
    assert isinstance(rbf, xr.DataArray)
    assert rbf.dims == bathymetry_grid.dims
    assert_array_almost_equal(rbf["x"], bathymetry_grid["x"])
    assert_array_almost_equal(rbf["y"], bathymetry_grid["y"])
    assert_array_almost_equal(
        rbf,
        [
            [2.94248917, 4.74939923, 3.45931706, 2.6264665, 3.54428058],
            [2.05023821, 4.34627561, 2.46904129, 1.38290294, 3.34335636],
            [2.6933896, 4.34772865, 2.33047468, 1.44382587, 3.40129971],
            [2.70351887, 2.17604895, 0.37018807, 1.16146539, 3.1474354],
            [2.74546424, 1.2038024, 0.22702653, 1.51034794, 3.28929487],
        ],
    )
