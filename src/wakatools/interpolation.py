import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from wakatools.validation import validate_input


@validate_input
def tin_surface(
    *data: pd.DataFrame | gpd.GeoDataFrame, value: str, target_grid: xr.DataArray
) -> xr.DataArray:
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
    data : pd.DataFrame | gpd.GeoDataFrame
        One or more DataFrame or GeoDataFrame instances containing 'x', 'y', and 'value'
        columns representing the points to interpolate from.
    value : str
        The name of the column in `data` that contains the values to interpolate.
    target_grid : xr.DataArray
        Target grid as an xarray DataArray on which to interpolate the values.

    Returns
    -------
    xr.DataArray
        Interpolated values on the target grid as an xarray DataArray.

    """
    data = pd.concat(data, ignore_index=True)

    interpolated = _tin(
        data.waka.coordinates(), data[value].values, target_grid.waka.grid_coordinates()
    )

    return xr.DataArray(
        interpolated.reshape(target_grid.shape),
        coords=target_grid.coords,
        dims=target_grid.dims,
    )


def _tin(
    points: np.ndarray, values: np.ndarray, query_points: np.ndarray
) -> np.ndarray:
    """
    Interpolate a TIN (Triangulated Irregular Network) surface for a set of query points
    based on input points and their associated values. The interpolation is done by
    taking a weighted average of the values at the vertices of the enclosing triangle,
    using the barycentric coordinates.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (N, 2) containing the x,y coordinates of the input points.
    values : np.ndarray
        An array of shape (N,) containing the values associated with each input point.
    query_points : np.ndarray
        An array of shape (M, 2) containing the x,y coordinates of the query points to
        interpolate.

    Returns
    -------
    np.ndarray
        An array of shape (M,) containing the interpolated values at the query points.

    """
    from scipy.spatial import Delaunay

    def _calculate_barycentric_coordinates(tri, simplex, points):
        x = tri.transform[simplex, :2]
        y = points - tri.transform[simplex, 2]
        barycentric = np.einsum("ijk,ik->ij", x, y)
        coordinates = np.c_[barycentric, 1 - barycentric.sum(axis=1)]
        return coordinates

    tri = Delaunay(points)
    simplices = tri.find_simplex(query_points)
    bary_coords = _calculate_barycentric_coordinates(tri, simplices, query_points)

    corner_values = values[tri.simplices[simplices]]

    interpolated = np.nansum(corner_values * bary_coords, axis=1)
    interpolated[simplices < 0] = np.nan  # Outside the convex hull of points

    return interpolated


@validate_input
def griddata(
    *data: pd.DataFrame | gpd.GeoDataFrame,
    value: str,
    target_grid: xr.DataArray,
    **kwargs,
) -> xr.DataArray:
    """
    Interpolate values from a Pandas DataFrame containing x,y,value for a set of points
    onto a target grid using SciPy's griddata function.

    Parameters
    ----------
    data : pd.DataFrame | gpd.GeoDataFrame
        One or more DataFrame or GeoDataFrame instances containing 'x', 'y', and 'value'
        columns representing the points to interpolate from.
    value : str
        The name of the column in `data` that contains the values to interpolate.
    target_grid : xr.DataArray
        Target grid as an xarray DataArray on which to interpolate the values.
    **kwargs
        Additional keyword arguments to pass to `scipy.interpolate.griddata`, such as
        `method` which can be 'linear', 'nearest', or 'cubic'. See SciPy documentation
        for more details.

    Returns
    -------
    xr.DataArray
        Interpolated values on the target grid as an xarray DataArray.

    """
    from scipy.interpolate import griddata as scipy_griddata

    data = pd.concat(data, ignore_index=True)

    grid_points = target_grid.waka.grid_coordinates()
    interpolated = scipy_griddata(
        points=data.waka.coordinates(),
        values=data[value].values,
        xi=grid_points,
        **kwargs,
    )

    return xr.DataArray(
        interpolated.reshape(target_grid.shape),
        coords=target_grid.coords,
        dims=target_grid.dims,
    )


@validate_input
def rbf(
    *data: pd.DataFrame | gpd.GeoDataFrame,
    value: str,
    target_grid: xr.DataArray,
    **kwargs,
) -> xr.DataArray:
    """
    Interpolate values from a Pandas DataFrame containing x,y,value for a set of points
    onto a target grid using Radial Basis Function (RBF) interpolation.

    Parameters
    ----------
    data : pd.DataFrame | gpd.GeoDataFrame
        One or more DataFrame or GeoDataFrame instances containing 'x', 'y', and 'value'
        columns representing the points to interpolate from.
    value : str
        The name of the column in `data` that contains the values to interpolate.
    target_grid : xr.DataArray
        Target grid as an xarray DataArray on which to interpolate the values.
    **kwargs
        Additional keyword arguments to pass to `scipy.interpolate.RBFInterpolator`,
        such as `kernel`, `epsilon`, etc. See SciPy documentation for more details.

    Returns
    -------
    xr.DataArray
        Interpolated values on the target grid as an xarray DataArray.

    """
    from scipy.interpolate import RBFInterpolator

    data = pd.concat(data, ignore_index=True)

    # Use scaled coordinates for better numerical stability
    scaled_coords = data.waka.coordinates_scaled(bbox=target_grid.rio.bounds())

    rbf = RBFInterpolator(
        scaled_coords,
        data[value].values,
        **kwargs,
    )

    grid_points = target_grid.waka.grid_coordinates_scaled()
    interpolated = rbf(grid_points)

    return xr.DataArray(
        interpolated.reshape(target_grid.shape),
        coords=target_grid.coords,
        dims=target_grid.dims,
    )
