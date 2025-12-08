# import geost
import pandas as pd


def read_seismics(filename):
    data = pd.read_csv(filename, header=None)
    header = ["x", "y", "ID", "count", "time", "amplitude"]
    data.columns = header
    return data


def read_boreholes(filepath):
    # boreholes = geost.read_boreholes(filepath)
    pass
