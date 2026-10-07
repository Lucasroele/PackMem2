"""Funtions for matrix."""
# R. Gautier A. Bacle 2015
# M. Zygadlo 2025

import math
import numpy as np
from packmem2.core.constants import MAX_GRID_SEARCH_CELLS, SIZE, SIZE_SIDE


def initialize_matrix2D(val1: int, val2: int, default: int | float | str) -> np.array:
    """
    Initialise a 2D matrix with a default value.

    --------------------
    INPUT
    val1: int
        Dimension in X/Y for the array
    val2: int
        Dimension in X/Y for the array
    default : int / float / string
        Default value in the matrix cells

    --------------------
    OUTPUT
    numpy matrix
        Matrix initialised with default value
    """
    matrix = np.full((val1, val2), default)
    return matrix


def diff_Z(arrayZ: np.array, coordZ: float) -> float:
    """
    Compute the difference between the last value in listZ and the Z for one atom.

    --------------------
    INPUT
    arrayZ: numpy array
        List from zmax+1 to z_C2_coord-1 by step of 1.0  OR
        List from z_C2_coord+1 to zmin-1 by step of 1.0
    coordZ: float
        The z coordinate for one atom

    --------------------
    OUPUT
    float
        The difference between the last value in listZ and the Z for one atom
    """
    return arrayZ[-1] - coordZ


def find_X_Y(coord: np.array, matX: np.array, matY: np.array) -> tuple[int, int]:
    """
    Determine from x,y coordinates the index in matX and matY.

    Has been said to determine the central value in matrix

    --------------------
    INPUT
    coord: numpy array
        Contains the coordinates x,y,z of an atom
    matX: numpy array
        Contains integer from xmin-1 to xmax+1 by step of 1.0
    matY: numpy array
        Contains integer from ymin-1 to ymax+1 by step of 1.0

    --------------------
    OUTPUT
    tuple of int
        the index of the coord x and y of the atom in matX and matY
    """
    # Convert to integer
    if float(SIZE) != 1.000:
        tmpX = int(coord[0] / float(SIZE))
        tmpY = int(coord[1] / float(SIZE))
    else:
        tmpX = int(coord[0])
        tmpY = int(coord[1])
    # Find the index of the value given (x_atom or y_atom)
    iX = np.where(matX == tmpX)[0][0]
    iY = np.where(matY == tmpY)[0][0]
    return iX, iY


#def check_edges(val: int, val_lim1: int, val_lim2: int) -> int:
#    """
#    Check if the cell index is lower than val_lim1 or higher than val_lim2. Changes the value of the cell index if so.
#
#    --------------------
#    INPUT
#    val: int
#        Index of the coordinate of the atom in listX or listY
#    val_lim1: int
#        The number of cell to work around the atom
#    val_lim2: int
#        The length of listX or listY
#
#    --------------------
#    OUTPUT
#    int
#        The cell index that represents the center of the search
#    """
#    if val < val_lim1:
#        return val_lim1
#    elif val >= val_lim2 - val_lim1:
#        return val_lim2 - (val_lim1 + 1)
#    else:
#        return val


# This is used once for each leaflet
def fill_matrix(
    mat: np.array,
    radius_atm: float,
    aliph_atom: str,
    coordtmp: np.array,
    arrayX: np.array,
    arrayY: np.array,
    arrayZ: np.array,
) -> np.array:
    """
    Fill the matrix depending on the atom type.

    ---------------
    INPUT
    mat: numpy array 2D
        Matrix to be filled where there are atoms
        Initialised to 0.0
    radius_res: float
        the radius of the atom type for the residue
    aliph_atoms: string
        The nature of the atom. a : aliphatique / n : polar
    coordtmp: numpy array
        Contains the coordinates x,y,z of the atom
    arrayX: numpy array
        Array from xmin-1 to xmax+1 by step of 1.0
    arrayY: numpy array
        Array from ymin-1 to ymax+1 by step of 1.0
    arrayZ: numpy array
        Array from zmax+1 to z_C2_coord-1 by step of 1.0  OR
        Array from z_C2_coord+1 to zmin-1 by step of 1.0

    -----------------
    OUTPUT
    numpy array 2D
        Contains a value 0 < a < 1 for aliphatic atom
        > 1 if polar
        0 if it's a defect
    """
    # Number of cells to work around
    v = min(MAX_GRID_SEARCH_CELLS, len(arrayX) // 2, len(arrayY) // 2)
    # Limit distance to roughly select the cells that are near the atom+radius
    sqrtdist_lim = (SIZE + radius_atm) ** 2
    # Limit distance to select the cells that intersect the atom+radius
    sqrtdist_meet = (SIZE_SIDE + radius_atm) ** 2
    # Find the equivalent index in the matrix of the x,y positions
    iX, iY = find_X_Y(coordtmp, arrayX, arrayY)
    # Change if too close to the edge
    # Select the cells to work in at i+-v (to not spend too much time on useless cells)
    if iX < v:
        gridX = np.concatenate((arrayX[len(arrayX) - (v - iX) : len(arrayX)], arrayX[0 : iX + v + 1]))
    elif iX >= len(arrayX) - v:
        gridX = np.concatenate((arrayX[iX - v : len(arrayX)], arrayX[0 : v - (len(arrayX) - iX) + 1]))
    else:
        gridX = arrayX[iX - v : iX + v + 1]
    assert len(gridX) == 2 * v + 1
    if iY < v:
        gridY = np.concatenate((arrayY[len(arrayY) - (v - iY) : len(arrayY)], arrayY[0 : iY + v + 1]))
    elif iY >= len(arrayY) - v:
        gridY = np.concatenate((arrayY[iY - v : len(arrayY)], arrayY[0 : v - (len(arrayY) - iY) + 1]))
    else:
        gridY = arrayY[iY - v : iY + v + 1]
    assert len(gridY) == 2 * v + 1
    # Find the closest distance in Z
    sqrtdZ_min = np.min((arrayZ - coordtmp[2]) ** 2)
    if sqrtdZ_min > sqrtdist_lim:  # no close Z found
        return mat

    # Get all the pair of x,y possible around the atom
    # This need to yield x and y acrross the pbc
    Xg, Yg = np.meshgrid(gridX, gridY, indexing="ij")

    sqrt_distances = (Xg - coordtmp[0]) ** 2 + (Yg - coordtmp[1]) ** 2
    # Get the distances that are in the effective radius of the atom
    mask = sqrt_distances <= sqrtdist_meet - sqrtdZ_min

    # Convert the indexes to get the location in the matrix (global index) and not in the sublist (local)
    Xind = np.arange(iX - v, iX + v + 1)
    Yind = np.arange(iY - v, iY + v + 1)
    # Put the indexes that are in the effective radius (local) to be globalised to the matrix length
    X_idx, Y_idx = np.where(mask)  # local
    X_idx = Xind[X_idx]  # Get to global
    Y_idx = Yind[Y_idx]

    # Update the matrix
    if aliph_atom == "a":
        mat[X_idx, Y_idx] += 0.001
    elif aliph_atom == "p":
        mat[X_idx, Y_idx] += 1

    return mat


def binarize_matrix_without0(
    mat: np.array, val1: float = 0.0, val2: float = 0.99
) -> np.array:
    """
    Binarise the presence / absence of aliphatic atom (and packing defects) in matrix.

    --------------------
    INPUT
    mat: numpy matrix
        Contains where there are atoms in the simulation box and their type
    val1: float
        lower limit value
    val2: float
        higher limit value

    --------------------
    OUTPUT
    numpy matrix
        Boolean mask containing the position of the aliphatic / polar atoms
        (+ packing defects) in the simulation box
    """
    return (mat >= val2) | (mat <= val1)


def count_edge_area(area_defects: dict, edge_labels: list) -> int:
    """
    Count the total area of the clusters on the edge.

    --------------------
    INPUT
    area_defects: dictionary
        Contains the labels of each defects and their areas
    edge_labels: list
        Contains the labels of the defects on the edge

    --------------------
    OUTPUT
    int
        The sum of the defect areas on the edge
    """
    total_edge_area = 0
    for key in edge_labels:
        total_edge_area += area_defects[key]
    return total_edge_area
