import geost
import pytest

from wakatools.io import read


@pytest.mark.unittest
def test_read_seismics(seismic_single_horizon_file):
    read.read_seismics(seismic_single_horizon_file)


@pytest.mark.unittest
def test_read_borehole_xml(testdatadir):
    files = testdatadir.glob("87078_HB*.xml")
    cores = read.read_borehole_xml(files, type_="geotechnical", company="Wiertsema")
    assert isinstance(cores, geost.BoreholeCollection)
    assert len(cores) == 2

    with pytest.raises(ValueError, match="Unsupported borehole type: unsupported_type"):
        read.read_borehole_xml(files, type_="unsupported_type")
