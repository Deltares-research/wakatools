import re
from functools import partial
from pathlib import Path
from typing import Iterable, Literal

import geost
import pandas as pd

from wakatools.io import kingdom_exports

BOREHOLE_READERS = {
    "geotechnical": geost.read_bhrgt,
    "geological": geost.read_bhrg,
    "pedological": geost.read_bhrp,
}
BoreholeType = Literal["geotechnical", "geological", "pedological"]

SEISMIC_READERS = {
    "multi-horizon": kingdom_exports.geocard7,
    "single-horizon": kingdom_exports.single_horizon,
}

SeismicFile = Literal["single-horizon", "multi-horizon"]


def read_seismics(
    files: str | Path | Iterable[str | Path],
    type_: SeismicFile,
    **kwargs,
):
    """
    General reader for seismic data files. Provides support for multiple Kingdom export formats via dedicated readers.

    Parameters
    ----------
    files : str | Path | Iterable[str  |  Path]
        Seismic data files
    type_ : SeismicFile, optional
        Type of seismic data file, this can be single-horizon or
        multi-horizon. Type of data file is based on the export method:
        multi-horizon = Kingdom Geocard7 export, single-horizon = Kingdom
        “X Y Line Trace Time Amplitude” export.

    Returns
    -------
    DataFrame
        DataFrame with seismic data from file. Format is based on input type,
        contains at least columns: [x, y, ID, time]

    Raises
    ------
    ValueError
        If input file is (yet) unsupported.

    """
    reader = SEISMIC_READERS.get(type_)
    if reader is None:
        raise ValueError(f"Unsupported or wrong type: {type_}")

    return reader(files, **kwargs)


def read_seismics_as_seismiccollection(
    files: str | Path | Iterable[str | Path],
    type_: SeismicFile,
    **kwargs,
):
    # Vraag aan bas: kan dit dan ook in read_seismics, is dat handig ?
    # en miss een andere n aam want dit ziet er niet uit maar ik heb even geen inspiratie
    from wakatools.parameters import SeismicCollection

    seismic_data = read_seismics(
        files,
        type_,
    )

    return SeismicCollection(seismic_data)


def read_borehole_xml(
    files: str | Path | Iterable[str | Path],
    type_: BoreholeType = "geotechnical",
    **kwargs,
) -> geost.BoreholeCollection:
    """
    Read XML files containing borehole information. The type of data can be one of
    "geotechnical", "geological", or "pedological" borehole XML files.

    Parameters
    ----------
    files : str | Path | Iterable[str  |  Path]
        XML file or files containing the borehole data to read.
    type_ : BoreholeType, optional
        The type of borehole data to read. Can be one of "geotechnical", "geological",
        or "pedological". The default is "geotechnical". Each different data type use
        the specific reader function from the GeoST library. See the relevant documentation
        for more information and usage details.
    **kwargs
        Additional keyword arguments to pass to the specific reader function.

        GeoST readers:
        - geost.read_bhrgt: for "geotechnical" borehole XML files.
        - geost.read_bhrg: for "geological" borehole XML files.
        - geost.read_bhrp: for "pedological" borehole XML files.

    Returns
    -------
    `geost.BoreholeCollection`
        GeoST BoreholeCollection containing the borehole data.

    Raises
    ------
    ValueError
        If an unsupported borehole type is provided.

    Examples
    --------
    For reading geotechnical borehole XML files from the Basis Registratie Ondergrond (BRO):

    >>> files = ["borehole1.xml", "borehole2.xml"] # list of XML files from the BRO
    >>> cores = read_borehole_xml(files, type_="geotechnical")

    See the GeoST documentation for more details on the specific reader functions and for
    options for files from other sources than the BRO.

    """
    reader = BOREHOLE_READERS.get(type_)
    if reader is None:
        raise ValueError(f"Unsupported borehole type: {type_}")

    return reader(files, **kwargs)
