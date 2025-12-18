import re

import pandas as pd
import pytest

from wakatools.io import read


@pytest.mark.unittest
def test_read_seismics(single_horizon_seismic_file):
    read.read_seismics(single_horizon_seismic_file)


@pytest.mark.unittest
def test_read_multi_horizon_seismic(seismic_multi_horizon_file):
    read.read_multi_horizon_seismic(seismic_multi_horizon_file)
