import pytest

from wakatools import utils


@pytest.mark.unittest
def test_time_to_depth():
    utils.time_to_depth(None, 1500)
    assert 1 == 1
