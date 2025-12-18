import numpy as np
import rioxarray  # noqa: F401 (get `rio` accessor registered and ignore "unused import" warning)
import xarray as xr

from .utils import scaling


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
