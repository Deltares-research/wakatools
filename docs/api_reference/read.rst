Input/Output
============

.. currentmodule:: wakatools

``wakatools`` provides functions to read various types of data files commonly used in
geospatial and geological applications.

Borehole data
--------------
.. autosummary::
   :toctree: generated/

    read_borehole_xml


Seismic data
----------------
.. autosummary::
   :toctree: generated/

    read_seismics


.. currentmodule:: wakatools.io.kingdom_exports

Readers for Kingdom export files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parsers for different types of The Kingdom Software exports. These parsers function as
input for :func:`wakatools.read_seismics`

.. autosummary::
   :toctree: generated/

   geocard7
   single_horizon
