import numpy as np

import packmem2.core.matrix as m

def test_initialize_matrix2D():
    tested_ouput = m.initialize_matrix2D(4, 4, np.nan)
    wanted_output = np.array([[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]])
    np.testing.assert_array_equal(tested_ouput, wanted_output)

def test_diff_Z():
    listZ = np.array([63.54, 62.54, 61.54, 60.54, 59.54, 58.54, 57.54, 56.54, 55.54, 54.54, 53.54, 52.54, 51.54, 50.54])
    tested_ouput = round(m.diff_Z(listZ, 52.18),2)
    wanted_output = -1.64
    assert(tested_ouput == wanted_output)

def test_find_X_Y():
    coord = np.array([44.79, 32.78, 52.18])
    listX = np.arange(-9.0, 95.0)
    listY =  np.arange(-11.0, 96.0)
    tested_ouputX, tested_ouputY = m.find_X_Y(coord, listX, listY)
    wanted_outputX, wanted_outputY = 53, 43
    assert(tested_ouputX == wanted_outputX)
    assert(tested_ouputY == wanted_outputY)

def test_fill_matrix():
    listX = np.arange(-1., 6., 1)
    listY = np.arange(-1., 6., 1)
    Matrix = np.full((len(listX), len(listY)), 0.0)
    tested_ouput = m.fill_matrix(Matrix, 1.34, 'a', [0, 4, 2], listX, listY, np.array([4, 3, 2, 1, 0]))
    wanted_output = np.array([[0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.001],
                              [0.0, 0.0, 0.0, 0.001, 0.001, 0.001, 0.001],
                              [0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.001],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(tested_ouput, wanted_output)

def test_binarize_matrix_without0():
    Matrix = np.array([[0.0, 0.003, 5.0, 0.5], [0.001, 0.003, 1.005, 0.003], [0.99, 0.003, 0.003, 0.003], [0.0, 0.0, 0.001, 0.0]])
    tested_ouput = m.binarize_matrix_without0(Matrix, -0.01, 0.99)
    wanted_output = np.array([[False, False, True, False], [False, False, True, False], [True, False, False, False], [False, False, False, False]])
    assert tested_ouput.dtype == bool
    np.testing.assert_array_equal(tested_ouput, wanted_output)

def test_count_edge_area():
    cluster_edge = np.array([1, 5])
    area_cluster = {1: 6, 2 : 3, 3 : 2, 4 : 2, 5 : 5}
    tested_ouput = m.count_edge_area(area_cluster, cluster_edge)
    wanted_output = 11
    assert(tested_ouput == wanted_output)
