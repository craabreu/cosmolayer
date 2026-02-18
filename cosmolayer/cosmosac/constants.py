"""
.. module:: cosmolayer.cosmosac.constants
   :synopsis: Constants for COSMO-SAC model variants.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import numpy as np

# COSMO-SAC 2002 parameters
COSMO_SAC_2002_EXPONENTS: tuple[int, ...] = (1,)
COSMO_SAC_2002_AREA_PER_SEGMENT: float = 7.5  # Å²
COSMO_SAC_2002_AVERAGING_RADIUS: float = 0.8176300195  # Å
COSMO_SAC_2002_F_DECAY: float = 1.0
COSMO_SAC_2002_SIGMA_0: None = None

# COSMO-SAC 2010 parameters
COSMO_SAC_2010_EXPONENTS: tuple[int, ...] = (1, 3)
COSMO_SAC_2010_AREA_PER_SEGMENT: float = 7.25  # Å²
COSMO_SAC_2010_AVERAGING_RADIUS: float = np.sqrt(7.25 / np.pi)  # Å
COSMO_SAC_2010_F_DECAY: float = 3.57
COSMO_SAC_2010_SIGMA_0: float = 0.007  # e/Å²
