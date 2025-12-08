import numpy as np


def scale(
    array: np.ndarray, min_: int | float = None, max_: int | float = None
) -> np.ndarray:
    """
    Scale an array to a 0-1 range based on provided minimum and maximum values.

    Parameters
    ----------
    array : np.ndarray
        The input array to be scaled.
    min_, max_ : int | float, optional
        The minimum and maximum values to scale data between. The default is None, then
        the array's min and max are used.

    Returns
    -------
    np.ndarray
        The scaled array.

    """
    if min_ is None:
        min_ = array.min()
    if max_ is None:
        max_ = array.max()
    return (array - min_) / (max_ - min_)
