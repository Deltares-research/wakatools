"""
Interpolation techniques:

- griddata
- RBF
- TIN

"""

import numpy as np
import pandas as pd
import xarray as xr


def _calculate_barycentric_coordinates(tri, simplex, points):
    x = tri.transform[simplex, :2]
    y = points - tri.transform[simplex, 2]
    barycentric = np.einsum("ijk,ik->ij", x, y)
    coordinates = np.c_[barycentric, 1 - barycentric.sum(axis=1)]
    return coordinates


def tin_surface(data: pd.DataFrame, target_grid: xr.DataArray) -> xr.DataArray:
    """
    Interpolate a TIN (Triangulated Irregular Network) surface from a Pandas DataFrame
    containing x,y,value for a set of points using a target grid. The interpolation is
    done by taking a weighted average of the values at the vertices of the enclosing
    triangle, using the barycentric coordinates. These are a weight measure for the distance
    of the point from each vertex. The resulting grid only contains values for each cell
    that falls within the convex hull of the input points because only those cells have
    valid barycentric coordinates.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'x', 'y', and 'z' columns representing the points to
        interpolate from.
    target_grid : xr.DataArray
        Target grid as an xarray DataArray on which to interpolate the values.

    Returns
    -------
    xr.DataArray
        Interpolated values on the target grid as an xarray DataArray.

    """
    from scipy.spatial import Delaunay

    grid_points = target_grid.waka.grid_coordinates()
    values = data["z"].values

    tri = Delaunay(data[["x", "y"]])
    simplices = tri.find_simplex(grid_points)
    bary_coords = _calculate_barycentric_coordinates(tri, simplices, grid_points)

    corner_values = values[tri.simplices[simplices]]

    interpolated = np.sum(corner_values * bary_coords, axis=1)
    interpolated[simplices < 0] = np.nan  # Outside the convex hull of points

    return xr.DataArray(
        interpolated.reshape(target_grid.shape),
        coords=target_grid.coords,
        dims=target_grid.dims,
    )
