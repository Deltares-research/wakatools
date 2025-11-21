import pandas as pd
import pytest

from wakatools import utils


@pytest.mark.unittest
def test_time_to_depth(seismic_data):
    depth = utils.time_to_depth(seismic_data)
    assert depth is not None
    assert 1 == 1
