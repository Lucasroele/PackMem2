"""Shared constants for PackMem2 core calculations."""

import math

DECIMALS = 2
# The size of the grid cells in Angstroms
SIZE = 1.000
SIZE_SIDE = 0.5 * math.sqrt(SIZE**2 * 3.0)
MAX_GRID_SEARCH_CELLS = 5
MAX_Z_DISTANCE = 5.0