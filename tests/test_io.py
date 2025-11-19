import pytest

from wakatools import io


@pytest.mark.unittest
def test_read_seismics(seismic_file):
    data = io.read_seismics(seismic_file)
    assert data is None
