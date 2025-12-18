import geost
import pandas as pd

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
