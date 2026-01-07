import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

from wakatools.validation import MissingColumnsError, validate_input


@validate_input
def interpolation_validation_passes(*xyz, value, target_grid, **kwargs) -> bool:
    """
    Dummy interpolation function for testing purposes to test the validation before
    interpolation only. Returns True if the validation passes.

    """
    return True


@pytest.fixture
def dummy_target():
    return xr.DataArray([1, 2, 3])


@pytest.mark.unittest
def test_validation_passes(dummy_target):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    assert interpolation_validation_passes(df, value="z", target_grid=dummy_target)

    gdf = gpd.GeoDataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    assert interpolation_validation_passes(gdf, value="z", target_grid=dummy_target)


@pytest.mark.unittest
def test_validation_passes_multiple(dummy_target):
    df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    gdf2 = gpd.GeoDataFrame({"x": [7, 8], "y": [9, 10], "z": [11, 12]})
    assert interpolation_validation_passes(
        df1, gdf2, value="z", target_grid=dummy_target
    )


@pytest.mark.unittest
def test_validation_fails_missing_x(dummy_target):
    df = pd.DataFrame({"y": [3, 4], "z": [5, 6]})
    with pytest.raises(
        MissingColumnsError,
        match=r"Interpolation data DataFrame is missing required columns: \['x'\]",
    ):
        interpolation_validation_passes(df, value="z", target_grid=dummy_target)


@pytest.mark.unittest
def test_validation_fails_missing_y(dummy_target):
    df = pd.DataFrame({"x": [3, 4], "z": [5, 6]})
    with pytest.raises(
        MissingColumnsError,
        match=r"Interpolation data DataFrame is missing required columns: \['y'\]",
    ):
        interpolation_validation_passes(df, value="z", target_grid=dummy_target)


@pytest.mark.unittest
def test_validation_fails_missing_value(dummy_target):
    df = pd.DataFrame({"x": [3, 4], "y": [5, 6]})
    with pytest.raises(
        MissingColumnsError,
        match=r"Interpolation data DataFrame is missing required columns: \['z'\]",
    ):
        interpolation_validation_passes(df, value="z", target_grid=dummy_target)


@pytest.mark.unittest
def test_validation_fails_dtype(dummy_target):
    invalid_input = []
    with pytest.raises(
        TypeError,
        match=r"All input data must be Pandas DataFrame or Geopandas GeoDataFrame instances.",
    ):
        interpolation_validation_passes(
            invalid_input, value="z", target_grid=dummy_target
        )


@pytest.mark.unittest
def test_validation_fails_multiple(dummy_target):
    df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    df2 = pd.DataFrame({"z": [11, 12]})
    with pytest.raises(
        MissingColumnsError,
        match=r"Interpolation data DataFrame is missing required columns: \['x', 'y'\]",
    ):
        interpolation_validation_passes(df1, df2, value="z", target_grid=dummy_target)
