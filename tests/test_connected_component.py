import numpy as np
import MDAnalysis as mda
import sys

import packmem2.core.connected_component as cc

def test_find_neighbours():
    mat = np.array([[1, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1]])
    tested_out = cc.find_neighbours(mat, 1, 1)
    wanted_out = []
    assert tested_out == wanted_out

    tested_out = cc.find_neighbours(mat, 1, 2)
    wanted_out = [[1, 1]]
    assert tested_out == wanted_out

    tested_out = cc.find_neighbours(mat, 2, 4)
    wanted_out = [[1, 4], [2, 3]]
    assert tested_out == wanted_out

    #tested_out = cc.find_neighbours(mat, 0, 4)
    #wanted_out = [[3, 4]]
    #assert tested_out == wanted_out

def test_find_neighbours_with_edge():
    mat = np.array([[0, 0, 1, 1, 1],
                    [1, 1, 0, 0, 1],
                    [0, 1, 1, 0, 0]])
    tested_out = cc.find_neighbours(mat, 0, 0)
    wanted_out = [[2,4], [2,0]]
    assert tested_out == wanted_out
    tested_out = cc.find_neighbours(mat, 0, 1)
    wanted_out = [[2,0], [0,0]]
    assert tested_out == wanted_out
    tested_out = cc.find_neighbours(mat, 2, 0)
    wanted_out = [[2,4]]
    assert tested_out == wanted_out

def test_find_label_neighbours():
    mat = np.array([[1, 1, 1, 1, 2, 1],
                    [1, 3, 3, 1, 2, 1],
                    [1, 3, 1, 2, 2, 1],
                    [1, 3, 1, 1, 2, 1]])
    defect_neighbours_OK = [[1, 4], [2, 3]]
    defect_neighbours_NO = [[1, 2], [1, 4]]

    tested_out = cc.find_label_neighbours(mat, defect_neighbours_OK)
    wanted_out = [2]
    assert tested_out == wanted_out

    tested_out = cc.find_label_neighbours(mat, defect_neighbours_NO)
    wanted_out = [2, 3]
    assert tested_out == wanted_out

def test_intersect():
    array_1 = np.array([1, 2, 3, 4, 5])
    array_2 = np.array([2, 4, 5, 5, 6])

    tested_out = cc.intersect(array_1, array_2)
    wanted_out = np.array([2, 4, 5])
    assert np.allclose(tested_out, wanted_out)

def test_sort_without_duplicate_merged_lists():
    list_A = [3,4,5,7]
    list_B = [4, 9]

    tested_out = cc.sort_without_duplicate_merged_lists(list_A, list_B)
    wanted_out = [3,4,5,7, 9]
    assert tested_out == wanted_out

def test_merge_list_of_lists():
    list_of_lists =  [[1, 2, 3, 9], [4, 5, 8], [1, 7, 9], [8, 9, 10], [13, 16], [12, 44], [16, 54]]

    tested_out = cc.merge_list_of_lists(list_of_lists)
    wanted_out = [[1, 2, 3, 4, 5, 7, 8, 9, 10], [13, 16, 54], [12, 44]]
    assert tested_out == wanted_out

def test_get_uniq_labels():
    equiv_labels = [[1,2], [3,8,9], [6,7]]

    tested_out1, tested_out2 = cc.get_uniq_labels(9, equiv_labels)
    wanted_out1 = {1: 1, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 3, 9: 3}
    wanted_out2 = [1, 3, 4, 5, 6]
    assert tested_out1 == wanted_out1
    assert tested_out2 == wanted_out2

def test_connect_labels_in_matrix():
    mat = np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 2, 0, 0, 8, 0],
                    [0, 1, 0, 0, 3, 9, 0],
                    [0, 0, 0, 0, 0, 9, 0],
                    [0, 6, 0, 4, 0, 0, 0],
                    [0, 7, 0, 0, 0, 5, 0],
                    [0, 0, 0, 0, 0, 0, 0]])
    dict_connected_labels = {1: 1, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 3, 9: 3}
    
    tested_out = cc.connect_labels_in_matrix(mat, dict_connected_labels)
    wanted_out = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 3, 0],
                        [0, 1, 0, 0, 3, 3, 0],
                        [0, 0, 0, 0, 0, 3, 0],
                        [0, 6, 0, 4, 0, 0, 0],
                        [0, 6, 0, 0, 0, 5, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(tested_out, wanted_out)

def test_connect_labels_in_matrix_with_edge():
    mat = np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 2, 0, 0, 5, 0],
                    [0, 1, 0, 0, 3, 9, 0],
                    [0, 0, 0, 0, 0, 9, 0],
                    [0, 6, 0, 4, 0, 0, 0],
                    [0, 7, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 8, 0]])
    dict_connected_labels = {1: 1, 2: 1, 3: 3, 4: 4, 5: 3, 6: 6, 7: 6, 8: 3, 9: 3}
    
    tested_out = cc.connect_labels_in_matrix(mat, dict_connected_labels)
    wanted_out = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0, 3, 0],
                           [0, 1, 0, 0, 3, 3, 0],
                           [0, 0, 0, 0, 0, 3, 0],
                           [0, 6, 0, 4, 0, 0, 0],
                           [0, 6, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 3, 0]])
    try:
        np.testing.assert_array_equal(tested_out, wanted_out)
    except AssertionError:
        with np.printoptions(threshold=np.inf, linewidth=200):
            print(f"Tested:\n{np.array2string(tested_out, separator=' ')}", file=sys.stderr)
            print(f"Wanted:\n{np.array2string(wanted_out, separator=' ')}", file=sys.stderr)
        raise
    mat = np.array([[1, 2, 0, 0, 8, 0],
                    [1, 0, 0, 3, 3, 0],
                    [0, 0, 0, 0, 3, 0],
                    [6, 0, 4, 0, 0, 0],
                    [7, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 5, 0]])
    dict_connected_labels = {1: 1, 2: 1, 3: 3, 4: 4, 5: 3, 6: 6, 7: 6, 8: 3}
    tested_out = cc.connect_labels_in_matrix(mat, dict_connected_labels)
    wanted_out = np.array([[1, 1, 0, 0, 3, 0],
                           [1, 0, 0, 3, 3, 0],
                           [0, 0, 0, 0, 3, 0],
                           [6, 0, 4, 0, 0, 0],
                           [6, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 3, 0]])
    try:
        np.testing.assert_array_equal(tested_out, wanted_out)
    except AssertionError:
        with np.printoptions(threshold=np.inf, linewidth=200):
            print(f"Tested:\n{np.array2string(tested_out, separator=' ')}", file=sys.stderr)
            print(f"Wanted:\n{np.array2string(wanted_out, separator=' ')}", file=sys.stderr)
        raise
    mat = np.array([[1, 2, 0, 0, 3, 0],
                    [1, 0, 0, 4, 3, 0],
                    [0, 0, 0, 0, 3, 0],
                    [5, 0, 6, 0, 0, 0],
                    [7, 5, 0, 0, 0, 0],
                    [0, 0, 0, 8, 0, 0]])
    dict_connected_labels = {1: 1, 2: 1, 3: 3, 4: 3, 5: 5, 6: 5, 7: 5, 8: 3}
    tested_out = cc.connect_labels_in_matrix(mat, dict_connected_labels)
    wanted_out = np.array([[1, 1, 0, 0, 3, 0],
                           [1, 0, 0, 3, 3, 0],
                           [0, 0, 0, 0, 3, 0],
                           [5, 0, 5, 0, 0, 0],
                           [5, 5, 0, 0, 0, 0],
                           [0, 0, 0, 3, 0, 0]])
    try:
        np.testing.assert_array_equal(tested_out, wanted_out)
    except AssertionError:
        with np.printoptions(threshold=np.inf, linewidth=200):
            print(f"Tested:\n{np.array2string(tested_out, separator=' ')}", file=sys.stderr)
            print(f"Wanted:\n{np.array2string(wanted_out, separator=' ')}", file=sys.stderr)
        raise
    mat = np.array([[1, 0, 0, 2],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [3, 0, 0, 4]])
    dict_connected_labels = {1: 1, 2: 1, 3: 1, 4: 1}
    tested_out = cc.connect_labels_in_matrix(mat, dict_connected_labels)
    wanted_out = np.array([[1, 0, 0, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [1, 0, 0, 1]])
    try:
        np.testing.assert_array_equal(tested_out, wanted_out)
    except AssertionError:
        with np.printoptions(threshold=np.inf, linewidth=200):
            print(f"Tested:\n{np.array2string(tested_out, separator=' ')}", file=sys.stderr)
            print(f"Wanted:\n{np.array2string(wanted_out, separator=' ')}", file=sys.stderr)
        raise

def test_get_area_first_coor_defects():
    mat = np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 3, 0],
                    [0, 1, 0, 0, 3, 3, 0],
                    [0, 0, 0, 0, 0, 3, 0],
                    [0, 6, 0, 4, 0, 0, 0],
                    [0, 6, 0, 0, 0, 5, 0],
                    [0, 0, 0, 0, 0, 0, 0]])
    uniq_labels = [1, 3, 4, 5, 6]

    tested_area, tested_coor = cc.get_area_first_coor_defects(mat, uniq_labels)
    wanted_area = {1:3, 3:4, 4:1, 5:1, 6:2}
    wanted_coor = {1:[1,1], 3:[1,5], 4:[4,3], 5:[5,5], 6:[4,1]}
    assert tested_area == wanted_area
    assert tested_coor  == wanted_coor

def test_get_connected_components():
    mat_labels = np.full((7, 8), 0)
    mat = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 1, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 0, 1, 1],
                    [1, 0, 1, 0, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]])
    
    tested_out1, tested_out2, tested_out3, tested_out4 = cc.get_connected_components(mat, mat_labels)
    wanted_out1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 2, 0],
                            [0, 1, 0, 0, 2, 2, 2, 0],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                            [0, 4, 0, 5, 0, 0, 0, 0],
                            [0, 4, 0, 0, 0, 6, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]])
    wanted_out2 = [1, 2, 4, 5, 6]
    wanted_out3 = {1:3, 2:5, 4:2, 5:1, 6:1}
    wanted_out4 = {1: [1,1], 2:[1,6], 4:[4,1], 5:[4,3], 6:[5,5]}

    try:
        assert np.array_equal(tested_out1, wanted_out1)
    except AssertionError:
        with np.printoptions(threshold=np.inf, linewidth=200):
            print(f"tested_out1:\n{np.array2string(tested_out1, separator=' ')}", file=sys.stderr)
            print(f"wanted_out1:\n{np.array2string(wanted_out1, separator=' ')}", file=sys.stderr)
        raise AssertionError("tested_out1 != wanted_out1")
    assert tested_out2 == wanted_out2
    assert tested_out3 == wanted_out3
    assert tested_out4 == wanted_out4

def test_get_connected_components_with_edge():
    mat_labels = np.full((4,4),0)
    mat = np.array([[0, 1, 1, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 1, 1, 0]])
    tested_out1, tested_out2, tested_out3, tested_out4 = cc.get_connected_components(mat, mat_labels)
    wanted_out1 = np.array([[1, 0, 0, 2],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [3, 0, 0, 4]])
    wanted_out2 = [1, 2, 3, 4]
    wanted_out3 = {1: 1, 2: 1, 3: 1, 4: 1}
    wanted_out4 = {1: [0, 0], 2: [0, 3], 3: [3, 0], 4: [3, 3]}
    try:
        assert np.array_equal(tested_out1, wanted_out1)
    except AssertionError:
        with np.printoptions(threshold=np.inf, linewidth=200):
            print(f"tested_out1:\n{np.array2string(tested_out1, separator=' ')}", file=sys.stderr)
            print(f"wanted_out1:\n{np.array2string(wanted_out1, separator=' ')}", file=sys.stderr)
        raise AssertionError("tested_out1 != wanted_out1")
    assert tested_out2 == wanted_out2
    assert tested_out3 == wanted_out3
    assert tested_out4 == wanted_out4

    mat_labels = np.full((5,5),0)
    mat = np.array([[1, 1, 1, 1, 1],
                    [1, 0, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 0, 1, 1, 0]])
    tested_out1, tested_out2, tested_out3, tested_out4 = cc.get_connected_components(mat, mat_labels)
    wanted_out1 = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 2],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 3, 0, 0, 4]])
    wanted_out2 = [1, 2, 3, 4]
    wanted_out3 = {1: 1, 2: 1, 3: 1, 4: 1}
    wanted_out4 = {1: [1, 1], 2: [1, 4], 3: [4, 1], 4: [4, 4]}
    try:
        assert np.array_equal(tested_out1, wanted_out1)
    except AssertionError:
        with np.printoptions(threshold=np.inf, linewidth=200):
            print(f"tested_out1:\n{np.array2string(tested_out1, separator=' ')}", file=sys.stderr)
            print(f"watned_out1:\n{np.array2string(wanted_out1, separator=' ')}", file=sys.stderr)
        raise AssertionError("tested_out1 != wanted_out1")
    assert tested_out2 == wanted_out2
    assert tested_out3 == wanted_out3
    assert tested_out4 == wanted_out4

def test_get_edge_defects():
    mat = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0, 0, 3, 0],
                    [0, 2, 0, 0, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0, 3, 0, 0],
                    [0, 5, 0, 6, 0, 0, 0, 8],
                    [0, 5, 0, 0, 0, 7, 0, 8],
                    [0, 0, 0, 1, 0, 0, 0, 8]])
    tested_out = cc.get_edge_defects(mat)
    wanted_out = [1, 8]
    assert tested_out == wanted_out    
