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
