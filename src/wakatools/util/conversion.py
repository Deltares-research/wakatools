import numpy as np
import pandas as pd
import shapely


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


def time_to_depth(df: pd.DataFrame) -> pd.Series:
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

        depth.loc[df["reflector"] == ref] = time * 800

    return depth


def add_depth_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to add a depth column to a seismic dataframe by converting
    two-way travel time to depth using a constant seismic velocity model.

    Parameters
    ----------
    df : pd.DataFrame
        Seismic dataframe with columns 'x', 'y', 'time', and 'reflector'.

    Returns
    -------
    pd.DataFrame
        Seismic dataframe with an added 'depth' column.
    """
    # df["depth"] = (
    #     df.groupby("ID", group_keys=False).apply(time_to_depth).reset_index(drop=True)
    # )
    df["depth"] = df.groupby("ID", group_keys=False).apply(
        lambda x: time_to_depth(x), include_groups=False
    )

    # test if bovenste wel werkt voo rmeer id;s, zo ja doe een if else for nunique
    df["depth"] = time_to_depth(df)
    return df
