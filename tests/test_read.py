import geost
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from wakatools.io import read


@pytest.mark.parametrize(
    "file",
    ["geocard7.dat", "xylinetracetimeamplitude.dat", "invalid-file.dat"],
    ids=["multi-horizon", "single-horizon", "invalid-type"],
)
def test_read_seismics(file, request, testdatadir):
    type_ = request.node.callspec.id
    if type_ == "invalid-type":
        with pytest.raises(ValueError, match="Unsupported or wrong type: invalid-type"):
            read.read_seismics(testdatadir / file, type_)
    else:
        data = read.read_seismics(testdatadir / file, type_)
        assert isinstance(data, pd.DataFrame)


@pytest.mark.unittest
def test_read_borehole_xml(testdatadir):
    files = testdatadir.glob("87078_HB*.xml")
    cores = read.read_borehole_xml(files, type_="geotechnical", company="Wiertsema")
    assert isinstance(cores, geost.BoreholeCollection)
    assert len(cores) == 2

    with pytest.raises(ValueError, match="Unsupported borehole type: unsupported_type"):
        read.read_borehole_xml(files, type_="unsupported_type")
