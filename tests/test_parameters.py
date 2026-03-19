import pandas as pd
import pytest

from wakatools.parameters import base_layer, top_layer


@pytest.mark.unittest
def test_top_layer(boreholes):
    top = top_layer(boreholes, "geotechnicalSoilName", "silt")
    assert top.shape[0] == 2


@pytest.mark.unittest
def test_base_layer(boreholes):
    base = base_layer(boreholes, "geotechnicalSoilName", "silt")
    assert base.shape[0] == 2
