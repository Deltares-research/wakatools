import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 (get `rio` accessor registered and ignore "unused import" warning)
import xarray as xr

from .utils import scaling


@pd.api.extensions.register_dataframe_accessor("waka")
class DataFrameAccessor:
    """
    Waka DataFrame accessor `.waka` for `pandas.DataFrame` instances holding wakatools
    functionalities for DataFrames.
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._df = pandas_obj

    @staticmethod
    def _validate(pandas_obj: pd.DataFrame):
        if not {"x", "y"}.issubset(set(pandas_obj.columns)):
            raise ValueError("DataFrame must have 'x' and 'y' columns.")

    def bounds(self) -> tuple:
        """
        Get the bounding box of the DataFrame as (xmin, ymin, xmax, ymax).

        Returns
        -------
        tuple
            A tuple containing the bounding box (xmin, ymin, xmax, ymax).

        """
        xmin = self._df["x"].min()
        ymin = self._df["y"].min()
        xmax = self._df["x"].max()
        ymax = self._df["y"].max()
        return (xmin, ymin, xmax, ymax)

    def coordinates(self) -> np.ndarray:
        """
        Get an array of all coordinates in the DataFrame in the shape (N, 2).

        Returns
        -------
        np.ndarray
            An array of shape (N, 2) containing all coordinates.

        """
        return self._df[["x", "y"]].to_numpy()

    def coordinates_scaled(self, bbox: tuple = None) -> np.ndarray:
        """
        Get an array of all coordinates in the DataFrame in the shape (N, 2), scaled to
        between 0 and 1 based on the bounding box of the DataFrame (xmin, ymin, xmax, ymax)
        or on a specified bounding box.

        Parameters
        ----------
        bbox : tuple, optional
            A tuple specifying the bounding box (xmin, ymin, xmax, ymax) to use for scaling.
            If None, the bounding box of the DataFrame is used. The default is None.

        Returns
        -------
        np.ndarray
            An array of shape (N, 2) containing all scaled coordinates.

        """
        if bbox is None:
            xmin, ymin, xmax, ymax = self.bounds()
        else:
            xmin, ymin, xmax, ymax = bbox

        xs = scaling.scale(self._df["x"].values, min_=xmin, max_=xmax)
        ys = scaling.scale(self._df["y"].values, min_=ymin, max_=ymax)
        return np.c_[xs, ys]


@xr.register_dataarray_accessor("waka")
class DataArrayAccessor:
    """
    Waka DataArray accessor `.waka` for `xarray.DataArray` instances holding wakatools
    functionalities for DataArrays.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._validate(xarray_obj)
        self._da = xarray_obj

    @staticmethod
    def _validate(xarray_obj: xr.DataArray):
        if not {"x", "y"}.issubset(set(xarray_obj.dims)):
            raise ValueError("DataArray must have 'x' and 'y' dimensions.")

    def grid_coordinates(self) -> np.ndarray:
        """
        Get an array of all grid coordinates in the DataArray in the shape (N, 2).

        Returns
        -------
        np.ndarray
            An array of shape (N, 2) containing all grid coordinates.

        """
        xgrid, ygrid = np.meshgrid(
            self._da.coords["x"].values, self._da.coords["y"].values
        )
        return np.c_[xgrid.ravel(), ygrid.ravel()]

    def grid_coordinates_scaled(self, bbox: tuple = None) -> np.ndarray:
        """
        Get an array of all grid coordinates in the DataArray in the shape (N, 2), scaled
        to between 0 and 1 based on the bounding box of the grid (xmin, ymin, xmax, ymax)
        or on a specified bounding box.

        Parameters
        ----------
        bbox : tuple, optional
            A tuple specifying the bounding box (xmin, ymin, xmax, ymax) to use for scaling.
            If None, the bounding box of the DataArray is used. The default is None.

        Returns
        -------
        np.ndarray
            An array of shape (N, 2) containing all scaled grid coordinates.

        """
        if bbox is None:
            xmin, ymin, xmax, ymax = self._da.rio.bounds()
        else:
            xmin, ymin, xmax, ymax = bbox

        xs = scaling.scale(self._da.coords["x"].values, min_=xmin, max_=xmax)
        ys = scaling.scale(self._da.coords["y"].values, min_=ymin, max_=ymax)
        xgrid, ygrid = np.meshgrid(xs, ys)
        return np.c_[xgrid.ravel(), ygrid.ravel()]
