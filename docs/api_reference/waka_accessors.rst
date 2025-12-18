Wakatools accessors
=============================================

.. currentmodule:: wakatools.base

Wakatools extends `xarray <http://xarray.pydata.org/en/stable/internals/extending-xarray.html>`__
with the ``waka`` accessor for common raster operations associated with interpolation. The
``waka`` accessor is activated by importing wakatools like so:

.. code-block:: python

    import wakatools

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
