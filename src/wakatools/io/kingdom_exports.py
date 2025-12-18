import re

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


def geocard7(filename: str) -> pd.DataFrame:
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
