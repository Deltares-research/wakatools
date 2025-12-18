import re

import pandas as pd
import pytest

from wakatools.io import read


@pytest.mark.unittest
def test_read_seismics(seismic_single_horizon_file):
    read.read_seismics(seismic_single_horizon_file)
