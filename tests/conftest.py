import pytest


@pytest.fixture
def seismic_data():
    # Pandas DataFrame Sample seismic data fixture
    return {"x": [0, 1, 2], "y": [0, 1, 2], "time": [10, 20, 30]}


@pytest.fixture
def borehole_data():
    # GeoST BoreholeCollection fixture
    return


@pytest.fixture
def sample_grid():
    # Xarray DataArray (bbox: 0, 0, 4, 4; xmin, ymin, xmax, ymax)
    return
