import numpy as np
import pandas as pd
import xarray as xr


def round_to_lower(value, base):
    return np.floor(value / base) * base


def round_to_upper(value, base):
    return np.ceil(value / base) * base


def target_grid_from(xyz: pd.DataFrame, resolution: int | float) -> xr.DataArray:
    """
    Create an interpolation target grid as an xarray.DataArray based on x and y
    locations from a DataFrame and a specified grid cell resolution.

    Parameters
    ----------
    xyz : pd.DataFrame
        DataFrame containing at least 'x' and 'y' columns representing data points
        that define the spatial boundaries of the target grid.
    resolution : int | float
        Resolution of the grid in meters

    Returns
    -------
    xr.DataArray
        A 2D DataArray with dimensions ("y", "x"), filled with zeros.
        The coordinates correspond to the grid cells covering the extent defined by the
        input points, aligned according to the specified resolution.

    """
    xmin, ymin = [
        round_to_lower(value, resolution) for value in (xyz["x"].min(), xyz["y"].min())
    ]
    xmax, ymax = [
        round_to_upper(value, resolution) for value in (xyz["x"].max(), xyz["y"].max())
    ]

    xgrid = np.arange(xmin - (0.5 * resolution), xmax + resolution, resolution)
    ygrid = np.arange(ymax + (0.5 * resolution), ymin - resolution, -resolution)

    X, _ = np.meshgrid(xgrid, ygrid)  # noqa: N806

    return xr.DataArray(np.zeros_like(X), coords=(ygrid, xgrid), dims=("y", "x"))
