"""
.. module:: cosmolayer.sac.segment_groups
   :synopsis: Define segment groups for COSMO-SAC activity coefficient calculations.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from typing import Literal

NHB: Literal["NHB"] = "NHB"  # Non-hydrogen-bonding segment group
OH: Literal["OH"] = "OH"  # Hydrogen-bonding segment group (hydroxyl)
OT: Literal["OT"] = "OT"  # Hydrogen-bonding segment group associated with other groups

SEGMENT_GROUPS: list[Literal[NHB, OH, OT]] = [NHB, OH, OT]
