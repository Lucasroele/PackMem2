import pytest
import numpy as np

import packmem2.core.pdb as pdb

def test_outputTXT_defects(tmp_path):
    dict_area = {2: 1}
    dict_coord = {2: [1, 1]}
    tot_area =  1
    arrayX = [-1., 0.]
    arrayY = [-3., -2.]

    pdb.outputTXT_defects(f"{tmp_path}/output", dict_area, dict_coord, tot_area, arrayX, arrayY)

    with open(f"{tmp_path}/output.txt", 'r') as file_test:
        content = file_test.readlines()
    wanted_lines = ["## MatrixSize     1 \n",
                    "## Total    1     1 1.00 100.000\n",
                    "  1    1     0.00    -2.00 \n"]
    assert content == wanted_lines

def test_write_a_pdb_line():
    content = pdb.write_a_pdb_line(1, 'EDG', 1, [-8.0, -9.0, 7.310], -1.0)
    wanted_line = f"ATOM      1   H1 EDG     1      -8.000  -9.000   7.310  1.00 -1.00\n"
    assert content == wanted_line

def  test_outputPDB_Total_matrix(tmp_path):
    arrayX = [-1., 0.]
    arrayY = [-3., -2.]
    z_extr = 4.430
    Matrix_final = np.array([[1, np.nan],
                             [1, 1]])

    pdb.outputPDB_Total_matrix(f"{tmp_path}/output", 1, arrayX, arrayY, z_extr, Matrix_final)

    with open(f"{tmp_path}/output.pdb", 'r') as file_test:
        content = file_test.readlines()
    wanted_lines = ["MODEL        1\n",
                    "ATOM      1   H1 MAT     1      -1.000  -3.000   4.430  1.00  1.00\n",
                    "ATOM      1   H1 EDG     1      -1.000  -2.000   4.430  1.00 -1.00\n",
                    "ATOM      2   H1 MAT     2       0.000  -3.000   4.430  1.00  1.00\n",
                    "ATOM      2   H1 MAT     2       0.000  -2.000   4.430  1.00  1.00\n",
                    "ENDMDL\n"]
    assert content == wanted_lines

def test_outputPDB_defects(tmp_path):
    arrayX = [-1., 0.]
    arrayY = [-3., -2.]
    z_extr = 4.430
    edge_labels = [1]
    Matrix_final = np.array([[1, 0],
                             [1, 2]])

    pdb.outputPDB_defects(f"{tmp_path}/output", 1, arrayX, arrayY, z_extr, Matrix_final, edge_labels)

    with open(f"{tmp_path}/output.pdb", 'r') as file_test:
        content = file_test.readlines()
    wanted_lines = ["MODEL        1\n",
                    "ATOM      1   H1 DEF     2       0.000  -2.000   4.430  1.00  2.00\n",
                    "ENDMDL\n"]
    assert content == wanted_lines
