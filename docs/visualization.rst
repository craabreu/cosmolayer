Visualization
=============

Cosmoview
---------

CosmoLayer offers a command line interface for visualizing COSMO files. 
The ``cosmoview`` command launches a window displaying the molecular structure along
with a tessellated COSMO cavity surface, where each segment is color-coded according to
its charge density.

.. argparse::
   :module: cosmolayer.cosmosac.visualize
   :func: get_parser
   :prog: cosmoview