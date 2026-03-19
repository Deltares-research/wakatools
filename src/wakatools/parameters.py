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
    if hasattr(data, "get_layer_top"):
        return data.get_layer_top(
            column, lith
        )  # hier kwargs doen later kijken lukt nu niet bas vragen

    if isinstance(data, pd.DataFrame):
        raise NotImplementedError(
            "Top layer extraction not implemented for DataFrame yet."
        )
    raise TypeError(
        "Unsupported data type. Provide a geost collection or pandas DataFrame."
    )


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
    if hasattr(data, "get_layer_base"):
        return data.get_layer_base(
            column, lith
        )  # hier kwargs doen later kijken lukt nu niet bas vragen

    if isinstance(data, pd.DataFrame):
        raise NotImplementedError(
            "Base layer extraction not implemented for DataFrame yet."
        )

    raise TypeError(
        "Unsupported data type. Provide a geost collection or pandas DataFrame."
    )
