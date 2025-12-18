import re

import pandas as pd
import pytest

from wakatools.io import kingdom_exports


@pytest.mark.unittest
def test_geocard7(seismic_geocard7_file):
    df = kingdom_exports.geocard7(seismic_geocard7_file)
    assert df is not None
    assert set(df["reflector"]) == {"1st reflector", "2nd reflector"}
    assert df.columns.__contains__("x")
    assert df.columns.__contains__("y")
    assert df.columns.__contains__("time")
    assert df.columns.__contains__("ID")

    # Assert if all data is read
    assert len(df) == 8864
