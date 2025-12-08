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
            "x": [0.3, 1.8, 2.7, 4.9, 0.6, 3.1, 4.4, 2.0, 1.2, 3.8],
            "y": [3.6, 2.1, 1.7, 4.8, 0.2, 3.4, 2.9, 1.3, 4.1, 0.7],
            "z": [1.4, 3.9, 0.6, 4.2, 2.8, 1.1, 3.3, 0.4, 4.7, 2.0],
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


@pytest.fixture
def single_horizon_seismic_file(tmp_path: Path):
    content = (
        "182043.95333,334581.60333,BwxTrajec_10_20240709_125934_P61_T1_RAW_LF,3629.00, 0.0064,88.8421\n"
        "182043.89000,334581.39000,BwxTrajec_10_20240709_125934_P61_T1_RAW_LF,3630.00, 0.0064,1168.08\n"
        "182043.84000,334581.22000,BwxTrajec_10_20240709_125934_P61_T1_RAW_LF,3631.00, 0.0064,-643.247\n"
        "182043.77667,334581.00667,BwxTrajec_10_20240709_125934_P61_T1_RAW_LF,3632.00, 0.0064,-823.187\n"
        "182043.71333,334580.79333,BwxTrajec_10_20240709_125934_P61_T1_RAW_LF,3633.00, 0.0064,-643.717\n"
        "182043.65000,334580.58000,BwxTrajec_10_20240709_125934_P61_T1_RAW_LF,3634.00, 0.0064,-1149.45\n"
        "182043.60000,334580.40000,BwxTrajec_10_20240709_125934_P61_T1_RAW_LF,3635.00, 0.0064,-1357.83\n"
        "182387.29000,335337.84000,BwxTrajec_10_P_-1_20240709_101637_P61_T1_RAW_LF,2.00, 0.0046,-1009.27\n"
        "182387.20000,335337.60000,BwxTrajec_10_P_-1_20240709_101637_P61_T1_RAW_LF,3.00, 0.0046,-804.448\n"
        "182387.11000,335337.36000,BwxTrajec_10_P_-1_20240709_101637_P61_T1_RAW_LF,4.00, 0.0046,-1521\n"
        "182387.02500,335337.14500,BwxTrajec_10_P_-1_20240709_101637_P61_T1_RAW_LF,5.00, 0.0046,-3237.2\n"
        "182386.94000,335336.93000,BwxTrajec_10_P_-1_20240709_101637_P61_T1_RAW_LF,6.00, 0.0046,-4206.23\n"
        "182386.87000,335336.76000,BwxTrajec_10_P_-1_20240709_101637_P61_T1_RAW_LF,7.00, 0.0046,-4236.83\n"
    )
    filepath = tmp_path / "seismic_single_horizon.dat"
    filepath.write_text(content)
    return filepath


def multi_horizon_seismic_file(tmp_path: Path):
    content = (
        "PROFILE 1st reflector (t TYPE 1 44\n"
        "SNAPPING PARAMETERS 2    20 2\n"
        "  1.82366320E+05  3.35292340E+05           5.08      1.00              1     -1108.811 2 BwxTrajec_10_20240709_125934_P61_T1_RAW_LF\n"
        "  1.82366227E+05  3.35292190E+05           5.09      2.00              2     -316.7626 2 BwxTrajec_10_20240709_125934_P61_T1_RAW_LF \n"
        "  1.82366133E+05  3.35292040E+05           5.09      3.00              3      293.0978 2 BwxTrajec_10_20240709_125934_P61_T1_RAW_LF \n"
        "  1.82387290E+05  3.35337840E+05           4.60      2.00              2     -1009.268 2 BwxTrajec_10_P_-1_20240709_101637_P61_T1_RAW_LF \n"
        "  1.82387200E+05  3.35337600E+05           4.60      3.00              3     -804.4478 2 BwxTrajec_10_P_-1_20240709_101637_P61_T1_RAW_LF \n"
        "EOD 1st reflector (top)\n"
        "PROFILE 2nd reflector (b TYPE 1 6 \n"
        "SNAPPING PARAMETERS 3    20 2\n"
        "  1.82366320E+05  3.35292340E+05           5.63      1.00              1     -26.51012 2 BwxTrajec_10_20240709_125934_P61_T1_RAW_LF \n"
        "  1.82366227E+05  3.35292190E+05           5.63      2.00              2     -726.5775 2 BwxTrajec_10_20240709_125934_P61_T1_RAW_LF \n"
        "  1.82963330E+05  3.36372350E+05           4.07   1473.00           1473     -516.7563 2 BwxTrajec_9_P_-20_20240710_105753_P61_T1_RAW_LF \n"
        "  1.82963450E+05  3.36372650E+05           4.07   1475.00           1475      2390.504 2 BwxTrajec_9_P_-20_20240710_105753_P61_T1_RAW_LF \n"
        "EOD 2nd reflector (bottom)\n"
    )
    filepath = tmp_path / "seismic_multi_horizon.dat"
    filepath.write_text(content)
    return filepath
