import numpy as np
import xarray as xr


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
