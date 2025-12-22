from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine

BBox = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
xres = yres = int | float


@dataclass
class GridSpecs:
    """
    Placeholder for method to derive a grid from various information (see Issue #1:
    https://github.com/Deltares-research/wakatools/issues/1). Currently not used but
    planned for future implementation to make "target_grid" optional input in interpolation
    routines.
    """

    bbox: BBox = None
    resolution: int | float | tuple["xres", "yres"] = None
    crs: int | str = None
    transform: Affine = None
    shape: tuple[int, int] = None
    align: Literal["center", "corner"] = "center"


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
    xmin, ymin, xmax, ymax = xyz.waka.bounds()
    xmin = round_to_lower(xmin, resolution)
    ymin = round_to_lower(ymin, resolution)
    xmax = round_to_upper(xmax, resolution)
    ymax = round_to_upper(ymax, resolution)

    # Shift xmin and ymax by 0.5 resolution to create cellcenters
    xgrid = np.arange(xmin + (0.5 * resolution), xmax, resolution)
    ygrid = np.arange(ymax - (0.5 * resolution), ymin, -resolution)

    X, _ = np.meshgrid(xgrid, ygrid)  # noqa: N806

    return xr.DataArray(np.zeros_like(X), coords=(ygrid, xgrid), dims=("y", "x"))
