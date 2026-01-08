import geost
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from wakatools.io import read


@pytest.mark.parametrize(
    "file, type_",
    [
        ("geocard7.dat", "multi-horizon"),
        ("xylinetracetimeamplitude.dat", "single-horizon"),
        ("invalid-file.dat", "invalid-type"),
    ],
    ids=["multi-horizon", "single-horizon", "invalid-type"],
)
def test_read_seismics(file, type_, testdatadir):
    if type_ == "invalid-type":
        with pytest.raises(ValueError, match="Unsupported or wrong type: invalid-type"):
            read.read_seismics(testdatadir / file, type_)
    else:
        data = read.read_seismics(testdatadir / file, type_)
        assert isinstance(data, pd.DataFrame)


@pytest.mark.xfail(reason="Function implementation incomplete.")
@pytest.mark.unittest
def test_read_seismics_as_seismiccollection(testdatadir):
    # Vraag aan bas, hoe test ik dit ding ? oja, en wat is een goede naam hiervoor?
    xyltta_file = testdatadir / "xylinetracetimeamplitude.dat"
    collection = read.read_seismics_as_SeismicCollection(xyltta_file, "single-horizon")
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
