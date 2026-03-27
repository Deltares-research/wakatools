import csv
import os
import re
from pathlib import Path

import pandas as pd


def load_ipf_metadata(path: str | Path) -> pd.DataFrame:
    """
    load_ipf_metadata reads the metadata file in IPF format of RWS and returns a
    DataFrame with the metadata.

    Parameters
    ----------
    path : str or Path
        Path to the metadata file

    Returns
    -------
    pd.DataFrame
        DataFrame with the metadata, with columns as specified in the file.
        The first two lines of the file are used to determine the number of columns
        and their names, and the rest of the lines are read as
        data rows.
    """

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    n_cols = int(lines[1].strip())
    column_names = lines[2 : 2 + n_cols]

    data_start = 2 + n_cols + 1
    data_lines = lines[data_start:]
    rows = list(csv.reader(data_lines, delimiter=","))

    df = pd.DataFrame(rows, columns=column_names)
    df = df.apply(pd.to_numeric, errors="ignore")

    return df


def load_measurement_file(path: str | Path) -> pd.DataFrame:
    """
    Reads data file in format as described in metadata and returns DataFrame with
    time, heas_measured and head_modelled

    Parameters
    ----------
    path : str or Path
        Path to the measurement file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, head_measured, head_modelled

    """

    with open(path, "r", encoding="utf-8") as f:
        raw = f.readlines()

    for i, line in enumerate(raw):
        if re.match(r"\d{14},", line.strip()):
            data_start = i
            break

    df = pd.read_csv(
        path, skiprows=data_start, names=["time", "head_measured", "head_modelled"]
    )

    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S")

    return df


class DataWrapper:
    def __init__(self, head, data):
        self.head = head
        self.data = data


def link_metadata_and_measurements(
    df_ipf: pd.DataFrame, measurement_folder: str | Path = "."
) -> dict[str, DataWrapper]:
    """
    _summary_

    Parameters
    ----------
    df_ipf : pd.DataFrame
        Dataframe containing metadata
    measurement_folder : str | Path, optional
        folder containing measurement files (.txt), by default "."

    Returns
    -------
    dict[str, DataWrapper]
        Dictionary where keys are buis_id (e.g. "B28F0210001") and values are
        DataWrapper objects containing the metadata row and the measurement dataframe.

    """

    result = {}

    for _, row in df_ipf.iterrows():
        series_path = row["MEETREEKS"].strip('"')
        buis_id = os.path.basename(series_path)

        txt_filename = f"{buis_id}.txt"
        txt_path = os.path.join(measurement_folder, txt_filename)

        if not os.path.exists(txt_path):
            print(f"⚠ Missing: {txt_filename} (skipped)")
            continue

        df_data = load_measurement_file(txt_path)

        result[buis_id] = DataWrapper(head=row, data=df_data)

    return result
