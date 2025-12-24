import geost
import pandas as pd
import pytest

from wakatools.io import read


@pytest.mark.unittest
def test_read_seismics(seismic_geocard7_file, seismic_xyltta_file):
    data = read.read_seismics(seismic_geocard7_file, "multi-horizon")
    assert isinstance(data, pd.DataFrame)
    data = read.read_seismics(seismic_xyltta_file, "single-horizon")
    assert isinstance(data, pd.DataFrame)
    # ToDo Eline: ADD MORE TESTS NOT FINISHED YET
    with pytest.raises(ValueError, match="Expected 6 columns, got 1"):
        read.read_seismics(seismic_geocard7_file, "single-horizon")
    with pytest.raises(ValueError, match="No objects to concatenate"):
        read.read_seismics(seismic_xyltta_file, "multi-horizon")


@pytest.mark.unittest
def test_read_borehole_xml(testdatadir):
    files = testdatadir.glob("87078_HB*.xml")
    cores = read.read_borehole_xml(files, type_="geotechnical", company="Wiertsema")
    assert isinstance(cores, geost.BoreholeCollection)
    assert len(cores) == 2

    with pytest.raises(ValueError, match="Unsupported borehole type: unsupported_type"):
        read.read_borehole_xml(files, type_="unsupported_type")
