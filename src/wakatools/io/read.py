import re
from io import StringIO

import geost
import pandas as pd

from wakatools.io import kingdom_exports

BOREHOLE_READERS = {
    "geotechnical": geost.read_bhrgt,
    "geological": geost.read_bhrg,
    "pedological": geost.read_bhrg,
}


def read_seismics(filename):
    data = pd.read_csv(filename, header=None)
    header = ["x", "y", "ID", "count", "time", "amplitude"]
    data.columns = header
    return data


def read_boreholes(files, type_="geotechnical", **kwargs):
    reader = BOREHOLE_READERS.get(type_)
    if reader is None:
        raise ValueError(f"Unsupported borehole type: {type_}")

    return reader(files, **kwargs)


def read_multi_horizon_seismic(filename: str) -> dict[str, pd.DataFrame]:
    """
    Read multi-horizon seismic data from a file. Input file is a Geocard Image 7 format
    from The Kingdom Software.

    Parameters
    ----------
    filename : str
        File path to the multi-horizon seismic data file as .dat file, export from Kingdom as
        Geocard Image 7 format.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with seismic data per horizon.

    """
    with open(filename) as f:
        text = f.read()

    pattern = re.compile(r"(PROFILE.*?)(?=PROFILE|EOD|$)", re.DOTALL)
    matches = pattern.findall(text)

    horizons = {}
    for horizon in matches:
        lines = horizon.strip().splitlines()
        data = lines[2:]
        title = re.search(r"PROFILE\s+(.*?)\s*\(", lines[0])
        if title:
            horizon_name = title.group(1)
        else:
            horizon_name = "unknown"

        textdata = "\n".join(data)

        data = pd.DataFrame([d.split() for d in data])
        data["line_id"] = horizon_name
        data["t"] = textdata  # TODO: remove this line later

    return horizons
