Wakatools accessors
=====================

.. currentmodule:: wakatools.base

Wakatools extends `Pandas <https://pandas.pydata.org/docs/development/extending.html>`__
and `Xarray <http://xarray.pydata.org/en/stable/internals/extending-xarray.html>`__
with the ``waka`` accessor for common DataFrame and raster operations associated with
interpolation. The ``waka`` accessor is activated by importing wakatools like so:

.. code-block:: python

    import wakatools


DataFrameAccessor
---------------------
Once wakatools is imported you can access the ``waka`` accessor on any Pandas DataFrame.

.. code-block:: python

    import pandas as pd

    # create a DataFrame with x and y coordinates
    df = pd.DataFrame({"y": [1.2, 2.3, 3.4], "x": [0.8, 1.9, 2.0]})
    bounding_box = df.waka.bounds() # get the bounding box from the coordinates
    print(bounding_box)
    # Output:
    # (0.8, 1.2, 2.0, 3.4)

.. autosummary::
   :toctree: generated/

   DataFrameAccessor
   DataFrameAccessor.bounds
   DataFrameAccessor.coordinates
   DataFrameAccessor.coordinates_scaled
   DataFrameAccessor.get_raster_values

DataArrayAccessor
---------------------
Once wakatools is imported you can access the ``waka`` accessor on any xarray DataArray.
This accessor provides useful methods for working with gridded data, for instances to get
all grid_coordinates of arrays as input for interpolation. Like so:

.. code-block:: python

    import xarray as xr

    da = xr.DataArray(
        [[1, 2], [3, 4]], coords={"y": [1, 0], "x": [0, 1]}, dims=("y", "x")
    )
    coords = da.waka.grid_coordinates()
    print(coords)
    # Output:
    # array([[0, 1],
    #        [1, 1],
    #        [0, 0],
    #        [1, 0]])


.. autosummary::
   :toctree: generated/

   DataArrayAccessor
   DataArrayAccessor.grid_coordinates
   DataArrayAccessor.grid_coordinates_scaled
