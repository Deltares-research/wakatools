from functools import wraps

import geopandas as gpd
import pandas as pd
import xarray as xr


class MissingColumnsError(Exception):
    """
    Custom Exception for missing columns when trying to interpolate data.
    """

    pass


def validate_input(func):
    """
    Validate input Pandas DataFrame instance or instances before interpolation occurs.
    This checks the presence of the required columns in order for all data to be properly
    concatenated to produce the interpolation input.

    Raises
    ------
    MissingColumnsError
        If any of the input DataFrames are missing required columns.
    """

    @wraps(func)
    def wrapper(
        *xyz: pd.DataFrame, value: str, target_grid: xr.DataArray, **kwargs
    ) -> xr.DataArray:
        required_cols = ["x", "y", value]
        for df in xyz:
            if not isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
                raise TypeError(
                    "All input data must be Pandas DataFrame or Geopandas GeoDataFrame instances."
                )

            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise MissingColumnsError(
                    f"Interpolation data DataFrame is missing required columns: {missing}. "
                    f"Please ensure that all input DataFrames have 'x', 'y', and '{value}' "
                    "columns."
                )
        return func(*xyz, value=value, target_grid=target_grid, **kwargs)

    return wrapper
