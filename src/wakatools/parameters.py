import pandas as pd
from geost.base import BoreholeCollection


def top_layer(
    data: BoreholeCollection | pd.DataFrame, column: str, lith: str
) -> BoreholeCollection | pd.DataFrame:
    """
    Returns the top layer of a specified lithology.

    Parameters
    ----------
    data : BoreholeCollection | pd.DataFrame
        Borehole data in either a BoreholeCollection or a DataFrame format*.
        *not implemented for DataFrame yet.
    column : str
        The name of the column to filter on.
    lith : str
        The value in the column to filter for.

    Returns
    -------
    BoreholeCollection | pd.DataFrame
        The top layer of the specified lithology.

    """
    if isinstance(data, BoreholeCollection):
        return data.get_layer_top(column, lith)
    else:
        pass


def base_layer(
    data: BoreholeCollection | pd.DataFrame, column: str, lith: str
) -> BoreholeCollection | pd.DataFrame:
    """
    Returns the base layer of a specified lithology.

    Parameters
    ----------
    data : BoreholeCollection | pd.DataFrame
        Borehole data in either a BoreholeCollection or a DataFrame format*.
        *not implemented for DataFrame yet.
    column : str
        The name of the column to filter on.
    lith : str
        The value in the column to filter for.
    Returns
    -------
    BoreholeCollection | pd.DataFrame
        The top layer of the specified lithology.

    """
    if isinstance(data, BoreholeCollection):
        return data.get_layer_base(column, lith)
    else:
        pass
