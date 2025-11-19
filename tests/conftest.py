import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def seismic_data():
    # Pandas DataFrame Sample seismic data fixture
    return pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "time": [10, 20, 30]})


@pytest.fixture
def seismic_file(tmp_path, seismic_data):
    filepath = tmp_path / "seismic.dat"
    seismic_data.to_csv(filepath, sep=" ", index=False, header=False)
    return filepath


def borehole_a():
    nlayers = 5
    top = [0, 0.8, 1.5, 2.5, 3.7]
    bottom = top[1:] + [4.2]
    mv = 0.2
    end = mv - bottom[-1]
    return pd.DataFrame(
        {
            "nr": np.full(nlayers, "A"),
            "x": np.full(nlayers, 2),
            "y": np.full(nlayers, 3),
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
            "x": np.full(nlayers, 1),
            "y": np.full(nlayers, 4),
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
