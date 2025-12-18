import re

import pandas as pd
import pytest

from wakatools.io import kingdom_exports, read


@pytest.mark.unittest
def test_read_seismics(single_horizon_seismic_file):
    read.read_seismics(single_horizon_seismic_file)


@pytest.mark.unittest
def test_read_multi_horizon_seismic(seismic_multi_horizon_file):
    read.read_multi_horizon_seismic(seismic_multi_horizon_file)


@pytest.mark.unittest
def test_geocard7(seismic_multi_horizon_file):
    df = kingdom_exports.geocard7(seismic_multi_horizon_file)
    assert isinstance(df, pd.DataFrame)
