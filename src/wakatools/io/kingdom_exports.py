import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

COLUMN_DTYPE_SCHEMA = {
    "x": "float64",
    "y": "float64",
    "time": "float64",
    "ID": "String",
    "reflector": "string",
}


def _apply_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for column, dtype in COLUMN_DTYPE_SCHEMA.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)
    return df


def geocard7(filename: str | Path) -> pd.DataFrame:
    """
    Parse a Kingdom Geocard7 seismic export file and return a DataFrame
    containing the seismic data.

    Parameters
    ----------
    filename : str
        Path to the Geocard7 seismic export file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the seismic data.

    """
    with open(filename) as f:
        text = f.read()

    # Match sections starting with PROFILE and capturing up to EOD
    pattern = re.compile(r"(PROFILE.*?)(?=PROFILE|EOD|$)", re.DOTALL)
    matches = pattern.findall(text)

    all_horizons = []
    for horizon in matches:
        lines = horizon.strip().splitlines()
        data = lines[2:]
        title = re.search(r"PROFILE\s+(.*?)\s*\(", lines[0])
        if title:
            horizon_name = title.group(1)
        else:
            horizon_name = "unknown"

        data = pd.DataFrame([d.split() for d in data])
        data.columns = [
            "x",
            "y",
            "time",
            "pointcount",
            "pointcountint",
            "amplitude",
            "noclue",
            "ID",
        ]

        data["reflector"] = horizon_name
        all_horizons.append(data)

    return pd.concat(all_horizons, ignore_index=True)


def single_horizon(
    filename: str | Path,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Reads a Kingdom export file containing data from a single seismic horizon into a
    DataFrame.
    The export is expected to be of type “X Y Line Trace Time Amplitude”.
    Custom column names can be provided for alternative export formats.

    Parameters
    ----------
    filename : str | Path
            Path to the Kingdom export file
    columns : Sequence[str] | None, optional 'X Y Line Trace Time Amplitude'
        Optional input for column names if export is different, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing the seismic data.

    Raises
    ------
    ValueError
        If length of the columns do not match the dataframe length, raise ValueError.

    """
    data = pd.read_csv(filename)

    if columns is None:
        columns = ["x", "y", "ID", "trace", "time", "amplitude"]

    if len(columns) != data.shape[1]:
        raise ValueError(f"Expected {len(columns)} columns, got {data.shape[1]}")

    data.columns = list(columns)
    return data
