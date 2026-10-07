from pathlib import Path
import pytest
from packmem2 import packmem2

def test_packmem2(tmp_path):
    topo = "tests/data/edge_data/POPC_memb.gro"
    traj = "tests/data/edge_data/2_frames.xtc"
    lipid = "POPC"
    start = 0
    end = 1
    paramFile = "data/param_Charmm.txt"
    radiiFile = "data/vdw_radii_Charmm.txt"
    indexFile = None
    output_dir = "tests/data/edge_data"
    outputname = "POPC"
    dist_suppl_Z = 1.0
    protein = False
    pdbout = False

    packmem2.launch(
        topo,
        traj,
        lipid,
        start,
        end,
        paramFile,
        radiiFile,
        indexFile,
        output_dir,
        outputname,
        dist_suppl_Z,
        protein,
        pdbout,
    )

    filename_template = outputname + "{i}_{side}_{depth}_result.txt"
    count = 2
    sides = ["Up", "Lo"]
    depths = ["All", "Shallow", "Deep"]

    for side in sides:
        for depth in depths:
            for i in range(count):
                filename = filename_template.format(i=i, side=side, depth=depth)
                file = Path(output_dir) / filename
                exp_file = Path(f"tests/data/edge_data/{filename}")
                assert file.exists()
                assert file.read_text() == exp_file.read_text()

def test_packmem2_edgemath(tmp_path):
    topo = "tests/data/edge_data/POPC_memb.gro"
    traj = "tests/data/edge_data/2_frames.xtc"
    lipid = "POPC"
    start = 0
    end = 1
    paramFile = "data/param_Charmm.txt"
    radiiFile = "data/vdw_radii_Charmm.txt"
    indexFile = None
    output_dir = tmp_path
    outputname = "POPC"
    dist_suppl_Z = 1.0
    protein = False
    pdbout = False

    packmem2.launch(
        topo,
        traj,
        lipid,
        start,
        end,
        paramFile,
        radiiFile,
        indexFile,
        output_dir,
        outputname,
        dist_suppl_Z,
        protein,
        pdbout,
    )
    
    filename_template = outputname + "{i}_{side}_{depth}_result.txt"
    count = 2
    sides = ["Up", "Lo"]
    depths = ["All", "Shallow", "Deep"]

    for side in sides:
        for depth in depths:
            for i in range(count):
                filename = filename_template.format(i=i, side=side, depth=depth)
                file = Path(output_dir) / filename
                with open(file, "r") as file:
                    firstline = file.readline()
                    line_l = firstline.split()
                    if depth == "Deep" and side == "Up" and i == 0 or depth == "Shallow":
                        assert line_l[-1] == line_l[-2], filename
                    else:
                        assert line_l[-1] != line_l[-2], filename
