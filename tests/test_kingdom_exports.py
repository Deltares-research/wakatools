import re

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from wakatools.io import kingdom_exports


@pytest.mark.unittest
def test_geocard7(seismic_geocard7_file):
    df = kingdom_exports.geocard7(seismic_geocard7_file)
    assert isinstance(df, pd.DataFrame)
    assert_array_equal(df["reflector"].unique(), ["1st reflector", "2nd reflector"])
    assert {"x", "y", "time", "ID"}.issubset(df.columns)
    assert len(df) == 8864  # Assert if all data is read


@pytest.mark.unittest
def test_single_horizon(seismic_xyltta_file):
    df = kingdom_exports.single_horizon(seismic_xyltta_file)
    assert isinstance(df, pd.DataFrame)
    assert {"x", "y", "time", "ID"}.issubset(df.columns)
    assert len(df) == 4842  # Assert if all data is read

    df = kingdom_exports.single_horizon(
        seismic_xyltta_file, columns=["x", "y", "ID", "trace", "time", "amplitude"]
    )
    assert isinstance(df, pd.DataFrame)
    assert {"x", "y", "ID", "trace", "time", "amplitude"}.issubset(df.columns)
    assert len(df) == 4842

    with pytest.raises(ValueError, match="Expected 5 columns, got 6"):
        kingdom_exports.single_horizon(
            seismic_xyltta_file, columns=["x", "y", "ID", "trace", "time"]
        )
