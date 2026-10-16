"""Functions about PDB data and output files."""
# R. Gautier A. Bacle 2015
# M. Zygadlo 2025

import numpy as np


def outputTXT_defects(
    out_name: str,
    area_defects: dict,
    first_coord: dict,
    total_size: int,
    arrayX: np.array,
    arrayY: np.array,
) -> None:
    """
    Create output file for packing defects in TXT format.

    ## MatrixSize  9801
    ## Total   51   582 11.41 6.034     # Total_packing_defect_number, total_packing_defect_area, average_size, percent_of_membrane
    1    7     7.00    55.00            # Defect_label  defect_size  X_position  Y_position
    2    1     7.00    58.00

    --------------------
    INPUT
    out_name: string
        Name of the output result txt file
    area_defects: dictionary
        Contains the area of each label
    first_coord: dictionary
        Contains the first appearance of the label in the matrix
    total_size: int
        The total area of the matrix
    arrayX: numpy array
        Array from xmin-1 to xmax+1 by step of 1.0
    arrayY: numpy array
        Array from ymin-1 to ymax+1 by step of 1.0
    """
    with open(f"{out_name}.txt", "w") as f_result:
        f_result.write(f"## MatrixSize {total_size:5d} \n")
        if len(first_coord) != 0:
            f_result.write(
                f"## Total {len(area_defects):4d} {sum(area_defects.values()):5d} {sum(area_defects.values()) / float(len(area_defects)):4.2f} {(sum(area_defects.values()) * 100.0) / (total_size):5.3f}\n"
            )
        else:
            # Exception if no defect
            f_result.write("## Total 0 0 0.0 0.0\n")
            return
        i = 0
        for key in first_coord:
            f_result.write(
                f"{i + 1:3d} {area_defects[key]:4d}   {arrayX[first_coord[key][0]]:6.2f}   {arrayY[first_coord[key][1]]:6.2f} \n"
            )
            i += 1


def write_a_pdb_line(
    nb_atm: int, atm_name: str, nb_res: int, coords: list, nb_defect: int
) -> str:
    """
    Write a PDB line.

    --------------------
    INPUT
    nb_atm: int
        The atom number
    atm_name: string
        The atom name
    nb_res: int
        The residue number
    coords: list
        Contains the coordinates x,y,z of the atom
    nb_defect: int
        The label of the defect
    """
    return f"{'ATOM  ':6s}{nb_atm:5d} {'  H1':4s} {atm_name:3s}  {nb_res:4d}    {float(coords[0]):8.3f}{float(coords[1]):8.3f}{float(coords[2]):8.3f}{1.0:6.2f}{nb_defect:6.2f}\n"


def outputPDB_Total_matrix(
    out_name: str,
    num_frame: int,
    arrayX: np.array,
    arrayY: np.array,
    z_extr: float,
    mat_final: np.array,
) -> None:
    """
    Create output file with Total Matrix in pdb format.

    --------------------
    INPUT
    out_name: string
        Name of the output PDB file
    num_frame: int
        The  frame number
    arrayX: numpy array
        Array from xmin-1 to xmax+1 by step of 1.0
    arrayY: numpy array
        Array from ymin-1 to ymax+1 by step of 1.0
    z_extr: float
        The maximum or  minimum z value
    mat_final: numpy array 2D
        Contains the types of the defects or np.nan for the edges
    """
    with open(f"{out_name}.pdb", "w") as f_tot:
        f_tot.write(f"MODEL      {num_frame:3d}\n")
        nb = 0
        for i, ind_matX in enumerate(arrayX):
            nb += 1
            for j, ind_matY in enumerate(arrayY):
                coordtmp = [ind_matX, ind_matY, z_extr]
                if np.isnan(mat_final[i][j]):
                    f_tot.write(write_a_pdb_line(nb, "EDG", nb, coordtmp, -1))
                else:
                    f_tot.write(
                        write_a_pdb_line(nb, "MAT", nb, coordtmp, mat_final[i][j])
                    )
        f_tot.write("ENDMDL\n")


def outputPDB_defects(
    out_name: str,
    num_frame: int,
    arrayX: np.array,
    arrayY: np.array,
    z_extr: float,
    mat_final: np.array,
    edge_labels: list,
) -> None:
    """
    Create output file with defects only in pdb format.

    --------------------
    INPUT
    out_name: string
        Name of the output PDB file
    num_frame: int
        The  frame number
    arrayX: numpy array
        Array from xmin-1 to xmax+1 by step of 1.0
    arrayY: numpy array
        Array from ymin-1 to ymax+1 by step of 1.0
    z_extr: float
        The maximum or  minimum z value
    mat_final: numpy array 2D
        Contains the label of the defects or 0.0 for the edges
    edge_labels: list
        Contains the labels on the edge of the matrix
    """
    with open(f"{out_name}.pdb", "w") as f:
        f.write(f"MODEL      {num_frame:3d}\n")
        nb = 0
        dict_Def = {}
        for i, ind_matX in enumerate(arrayX):
            for j, ind_matY in enumerate(arrayY):
                label_def = mat_final[i][j]
                if label_def != 0.0 and label_def not in edge_labels:
                    if int(label_def) not in dict_Def:
                        dict_Def[int(label_def)] = []
                    nb += 1
                    coordtmp = [ind_matX, ind_matY, z_extr]
                    dict_Def[int(label_def)].append(
                        write_a_pdb_line(nb, "DEF", int(label_def), coordtmp, label_def)
                    )
        keys_def = list(dict_Def.keys())
        keys_def.sort()
        nb = 0
        for key in keys_def:
            for line in dict_Def[key]:
                nb += 1
                f.write(f"{line[:6]}{nb:5d}{line[11:]}")
        f.write("ENDMDL\n")
