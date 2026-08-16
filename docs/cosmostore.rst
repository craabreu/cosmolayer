Cosmostore
==========

The ``cosmostore`` command builds a segment-data store from ``.cosmo``
files (if one is not already present) and prints summary statistics for
atom- and molecule-level charges, areas, and sigma profiles.

.. argparse::
   :module: cosmolayer.store.__main__
   :func: get_parser
   :prog: cosmostore
