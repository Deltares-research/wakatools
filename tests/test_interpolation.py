import numpy as np
import pandas as pd
import pytest

import wakatools as waka


@pytest.mark.unittest
def test_tin_surface(bathymetry_grid):
    # TODO: Replace sample data below with proper fixtures
    rnd = np.random.RandomState(0)
    n = 50
    points = rnd.rand(n, 2)  # (x, y)
    values = np.sin(points[:, 0] * 5) + np.cos(points[:, 1] * 4)
    data = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "z": values})

    waka.interpolation.tin_surface(data, bathymetry_grid)
