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


@pytest.mark.parametrize(
    "buffer, ncpts", [(None, 1), (110, 2)], ids=["no-buffer", "with-buffer"]
)
def test_bro_cpts_in(buffer, ncpts):
    bbox = (132781.52, 448029.34, 132783.52, 448031.34)
    cpts = read.bro_cpts_in(bbox=bbox, buffer=buffer)
    assert isinstance(cpts, geost.CptCollection)
    assert len(cpts) == ncpts


@pytest.mark.parametrize(
    "buffer, nbhrgt", [(None, 1), (110, 2)], ids=["no-buffer", "with-buffer"]
)
def test_bro_bhrgt_in(buffer, nbhrgt):
    bbox = (132780.327, 448030.0, 132782.327, 448032.1)
    bhrgt = read.bro_bhrgt_in(bbox=bbox, buffer=buffer)
    assert isinstance(bhrgt, geost.BoreholeCollection)
    assert len(bhrgt) == nbhrgt


@pytest.mark.parametrize(
    "buffer, nbhrg", [(None, 1), (10, 3)], ids=["no-buffer", "with-buffer"]
)
def test_bro_bhrg_in(buffer, nbhrg):
    bbox = (126148, 452161, 126150, 452163)
    bhrg = read.bro_bhrg_in(bbox=bbox, buffer=buffer)
    assert isinstance(bhrg, geost.BoreholeCollection)
    assert len(bhrg) == nbhrg
