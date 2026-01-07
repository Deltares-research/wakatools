import geost
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from wakatools.io import read


@pytest.mark.unittest
def test_read_seismics(seismic_geocard7_file, seismic_xyltta_file):
    data = read.read_seismics(seismic_geocard7_file, "multi-horizon")
    assert isinstance(data, pd.DataFrame)
    assert_array_equal(data["reflector"].unique(), ["1st reflector", "2nd reflector"])
    assert {"x", "y", "time", "ID"}.issubset(data.columns)
    assert len(data) == 8864

    data = read.read_seismics(seismic_xyltta_file, "single-horizon")
    assert isinstance(data, pd.DataFrame)
    assert {"x", "y", "time", "ID"}.issubset(data.columns)
    assert len(data) == 4842

    with pytest.raises(ValueError, match="Expected 6 columns, got 1"):
        read.read_seismics(seismic_geocard7_file, "single-horizon")

    with pytest.raises(ValueError, match="No objects to concatenate"):
        read.read_seismics(seismic_xyltta_file, "multi-horizon")


@pytest.mark.xfail(reason="Function implementation incomplete.")
@pytest.mark.unittest
def test_read_seismics_as_seismiccollection(seismic_xyltta_file):
    # Vraag aan bas, hoe test ik dit ding ? oja, en wat is een goede naam hiervoor?
    collection = read.read_seismics_as_SeismicCollection(
        seismic_xyltta_file, "single-horizon"
    )
    print(collection)  # zodat ie niet gaat zeuren
    assert 1 == 1


@pytest.mark.unittest
def test_read_borehole_xml(testdatadir):
    files = testdatadir.glob("87078_HB*.xml")
    cores = read.read_borehole_xml(files, type_="geotechnical", company="Wiertsema")
    assert isinstance(cores, geost.BoreholeCollection)
    assert len(cores) == 2

    with pytest.raises(ValueError, match="Unsupported borehole type: unsupported_type"):
        read.read_borehole_xml(files, type_="unsupported_type")
