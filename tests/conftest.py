from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from geost import BoreholeCollection, spatial


@pytest.fixture
def testdatadir():
    return Path(__file__).parent / "testdata"


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
def xyz_dataframe(bathymetry_grid):
    xcoords = np.array([0.3, 1.8, 2.7, 4.9, 0.6, 3.1, 4.4, 2.0, 1.2, 3.8])
    ycoords = np.array([3.6, 2.1, 1.7, 4.8, 0.2, 3.4, 2.9, 1.3, 4.1, 0.7])

    # Get the nearest bathymetry values at these coordinates and subtract some values
    # forconsistency in elevation.
    bathy = spatial.get_raster_values(xcoords, ycoords, bathymetry_grid)
    diff = np.array([1.0, 1.2, 1.8, 1.1, 0.9, 1.3, 1.0, 0.8, 1.4, 0.9])

    return pd.DataFrame(
        {
            "x": xcoords,
            "y": ycoords,
            "z": bathy - diff,
        }
    )


@pytest.fixture
def boreholes():
    nrs = ["A", "B"]
    xcoords = [4, 2]
    ycoords = [2, 1]
    surfaces = [0.2, 0.3]
    ends = [-2.1, -1.2]
    tops = [0.0, 0.1, 0.2, 0.7, 1.0, 1.3, 0.0, 0.5, 0.8, 1.2]
    bottoms = [0.1, 0.2, 0.7, 1.0, 1.3, 2.3, 0.5, 0.8, 1.2, 1.5]
    lith = [
        "silt",
        "zwakZandigGrind",
        "zwakZandigSilt",
        "zwakZandigGrind",
        "klei",
        "siltigZand",
        "silt",
        "sterkZandigGrind",
        "zwakZandigeKlei",
        "zand",
    ]

    header = gpd.GeoDataFrame(
        {
            "nr": nrs,
            "x": xcoords,
            "y": ycoords,
            "surface": surfaces,
            "end": ends,
        },
        geometry=gpd.points_from_xy(xcoords, ycoords),
        crs=28992,
    )
    data = pd.DataFrame(
        {
            "nr": np.repeat(nrs, [6, 4]),
            "x": np.repeat(xcoords, [6, 4]),
            "y": np.repeat(ycoords, [6, 4]),
            "surface": np.repeat(surfaces, [6, 4]),
            "end": np.repeat(ends, [6, 4]),
            "top": tops,
            "bottom": bottoms,
            "geotechnicalSoilName": lith,
        }
    )
    return BoreholeCollection(header, data)
