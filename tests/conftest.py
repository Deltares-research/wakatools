from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def seismic_data():
    return pd.DataFrame(
        {
            "ID": np.repeat(["line1", "line2"], [7, 6]),
            "x": [
                0.5,
                1.5,
                2.5,
                3.5,
                0.5,
                1.5,
                2.5,
                0.5,
                1.5,
                2.5,
                3.5,
                0.5,
                1.5,
            ],
            "y": np.repeat([0.5, 1.5], [7, 6]),
            "time": [
                0.0041,
                0.0042,
                0.0043,
                0.0041,
                0.0051,
                0.0052,
                0.0053,
                0.0054,
                0.0055,
                0.0056,
                0.0057,
                0.0058,
                0.0059,
            ],
            "reflector": np.repeat(["bathy", "bk", "bathy", "ok"], [4, 3, 4, 2]),
        }
    )


@pytest.fixture
def seismic_file(tmp_path: Path, seismic_data: pd.DataFrame):
    filepath = tmp_path / "seismic.dat"
    seismic_data.to_csv(filepath, sep=" ", index=False, header=False)
    return filepath


@pytest.fixture
def bathymetry_grid():
    xcoords = np.arange(5) + 0.5
    ycoords = np.arange(5, 0, -1) - 0.5
    data = np.array(
        [
            [0, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6, 0.7, 0.8],
        ]
    )
    return xr.DataArray(data, coords={"y": ycoords, "x": xcoords}, dims=("y", "x"))


@pytest.fixture
def xyz_dataframe():
    return pd.DataFrame(
        {
            "x": np.random.uniform(0, 5, 10),
            "y": np.linspace(-3, 4, 10),
            "z": np.random.randn(10),
        }
    )


def borehole_a():
    nlayers = 5
    top = [0, 0.8, 1.5, 2.5, 3.7]
    bottom = top[1:] + [4.2]
    mv = 0.2
    end = mv - bottom[-1]
    return pd.DataFrame(
        {
            "nr": np.full(nlayers, "A"),
            "x": 4,
            "y": 2,
            "surface": np.full(nlayers, mv),
            "end": np.full(nlayers, end),
            "top": top,
            "bottom": bottom,
            "lith": ["K", "K", "Z", "Z", "K"],
        }
    )


def borehole_b():
    nlayers = 5
    top = [0, 0.6, 1.2, 2.5, 3.1]
    bottom = top[1:] + [3.9]
    mv = 0.3
    end = mv - bottom[-1]
    return pd.DataFrame(
        {
            "nr": np.full(nlayers, "B"),
            "x": 2,
            "y": 1,
            "surface": np.full(nlayers, mv),
            "end": np.full(nlayers, end),
            "top": top,
            "bottom": bottom,
            "lith": ["K", "K", "V", "V", "K"],
        }
    )


@pytest.fixture
def boreholes():
    return pd.concat([borehole_a(), borehole_b()], ignore_index=True)


@pytest.fixture
def sample_grid():
    # Xarray DataArray (bbox: 0, 0, 4, 4; xmin, ymin, xmax, ymax)
    return
