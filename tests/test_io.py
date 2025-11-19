import pytest

from wakatools import read


@pytest.mark.unittest
def test_read_seismics(seismic_file):
    data = read.read_seismics(seismic_file)
    assert data is None
