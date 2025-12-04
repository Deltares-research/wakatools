import numpy as np
import pandas as pd
import shapely

from wakatools.constants import SeismicVelocity


def calculate_absolute_time(
    line1: shapely.LineString, line2: shapely.LineString
) -> np.ndarray:
    """
    Function to calculate the absolute time difference between two seismic lines.

    Parameters
    ----------
    line1 : shapely.LineString
        First seismic line as a shapely LineString object. This is the reference line,
        typically the bathymetry.
    line2 : shapely.LineString
        Second seismic line as a shapely LineString object.

    Returns
    -------
    np.ndarray
        Absolute time difference between the two seismic lines.
    """
    return np.array(
        [
            p.z - line1.interpolate(line1.project(p)).z
            for p in shapely.points(line2.coords)
        ]
    )


def _time_to_depth(df: pd.DataFrame) -> pd.Series:
    """
    Function to convert seismic two-way travel time to depth using a constant
    seismic velocity model.

    Parameters
    ----------
    df : pd.DataFrame
        Seismic dataframe with columns 'x', 'y', 'time', and 'reflector'.

    Returns
    -------
    pd.Series
        Depth values corresponding to the input seismic data.
    """
    bathy_line = shapely.linestrings(
        df.loc[df["reflector"] == "bathy", ["x", "y", "time"]].values
    )

    depth = pd.Series(index=df.index)
    for ref in df["reflector"].unique():
        if ref == "bathy":
            continue

        ref_line = shapely.linestrings(
            df.loc[df["reflector"] == ref, ["x", "y", "time"]].values
        )

        time = calculate_absolute_time(bathy_line, ref_line)

        depth.loc[df["reflector"] == ref] = time * (SeismicVelocity.SEDIMENT / 2.0)

    return depth


def calculate_depth(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the depth with respect to the bathymetry reflector for deeper reflectors in
    a Pandas DataFrame containing seismic data. This converts seismic two-way travel time
    to depth using a constant seismic velocity model.

    Parameters
    ----------
    df : pd.DataFrame
        Seismic dataframe with columns 'x', 'y', 'time', and 'reflector'.

    Returns
    -------
    pd.Series
        Depth values for bathymetry and reflectors corresponding to the input seismic
        data.

    """
    if df["ID"].nunique() == 1:
        depth = _time_to_depth(df)
    else:
        depth = df.groupby("ID", group_keys=False).apply(
            lambda x: _time_to_depth(x), include_groups=False
        )

    return depth.fillna(0.0)
