from pathlib import Path
import pytest
import packmem2.launch_packmem2 as launch_packmem2


@pytest.mark.slowest
def test_launch_DMPC(tmp_path):
    """DMPC"""
    cores = 1
    topo = "tests/data/end_to_end_DMPC/md.gro"
    traj = "tests/data/end_to_end_DMPC/md_10ns.xtc"
    lipid = "DMPC"
    start = 0
    end = 100
    paramFile = "data/param_Charmm.txt"
    radiiFile = "data/vdw_radii_Charmm.txt"
    indexFile = None
    output_dir = str(tmp_path)
    outputname = "DMPC"
    dist_suppl_Z = 1.0
    protein = False
    pdbout = False
    limx = 15
    limy = 1e-4
    precision = 3
    nb_block = 5

    launch_packmem2.launch(
        cores,
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
        limx,
        limy,
        precision,
        nb_block
    )

    # Check Total files
    expected_output_Deep = Path(f"{output_dir}/Total_Deep.csv")
    expected_output_Shallow = Path(f"{output_dir}/Total_Shallow.csv")
    expected_output_All = Path(f"{output_dir}/Total_All.csv")
    with open("tests/data/end_to_end_DMPC/Total_Deep.csv", "r") as f_in:
        expected_content_Deep = f_in.read()

    assert expected_output_Deep.exists()
    assert expected_output_Shallow.exists()
    assert expected_output_All.exists()
    assert expected_output_Deep.read_text() == expected_content_Deep

    # Check pdf
    expected_final_output = Path(f"{output_dir}/DMPC.pdf")
    expected_final_summary = Path(f"{output_dir}/DMPC.csv")

    assert expected_final_output.exists()
    assert expected_final_summary.exists()


@pytest.mark.slowest
def test_launch_DMPC_protein(tmp_path):
    """DMPC + protein"""
    cores = 1
    topo = "tests/data/end_to_end_DMPC_prot/md.gro"
    traj = "tests/data/end_to_end_DMPC_prot/md_10ns.xtc"
    lipid = "DMPC"
    start = 0
    end = 100
    paramFile = "data/param_Charmm.txt"
    radiiFile = "data/vdw_radii_Charmm.txt"
    indexFile = None
    output_dir = str(tmp_path)
    outputname = "DMPC"
    dist_suppl_Z = 1.0
    protein = True
    pdbout = False
    limx = 15
    limy = 1e-4
    precision = 3
    nb_block = 3

    with pytest.raises(TypeError):
        launch_packmem2.launch(
            cores,
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
            limx,
            limy,
            precision,
            nb_block
        )

    # Check Total files
    expected_output_Deep = Path(f"{output_dir}/Total_Deep.csv")
    expected_output_Shallow = Path(f"{output_dir}/Total_Shallow.csv")
    expected_output_All = Path(f"{output_dir}/Total_All.csv")
    with open("tests/data/end_to_end_DMPC_prot/Total_Deep.csv", "r") as f_in:
        expected_content_Deep = f_in.read()

    assert expected_output_Deep.exists()
    assert expected_output_Shallow.exists()
    assert expected_output_All.exists()
    assert expected_output_Deep.read_text() == expected_content_Deep

    # Check Total files for protein
    expected_output_Deep = Path(f"{output_dir}/Total_Up_Deep_prot.csv")
    expected_output_Shallow = Path(f"{output_dir}/Total_Up_Shallow_prot.csv")
    expected_output_All = Path(f"{output_dir}/Total_Up_All_prot.csv")
    with open("tests/data/end_to_end_DMPC_prot/Total_Up_Deep_prot.csv", "r") as f_in:
        expected_content_Deep = f_in.read()

    assert expected_output_Deep.exists()
    assert expected_output_Shallow.exists()
    assert expected_output_All.exists()
    assert expected_output_Deep.read_text() == expected_content_Deep

    # Check pdf
    expected_final_output = Path(f"{output_dir}/DMPC.pdf")

    assert expected_final_output.exists()


@pytest.mark.slowest
def test_launch_DLPC(tmp_path):
    """DLPC (Martini)"""
    cores = 1
    topo = "tests/data/end_to_end_DLPC/md.gro"
    traj = "tests/data/end_to_end_DLPC/md_10ns.xtc"
    lipid = "DLPC"
    start = 0
    end = 100
    paramFile = "data/param_Martini.txt"
    radiiFile = "data/vdw_radii_Martini_old.txt"
    indexFile = None
    output_dir = str(tmp_path)
    outputname = "DLPC"
    dist_suppl_Z = 1.0
    protein = False
    pdbout = False
    limx = 15
    limy = 1e-4
    precision = 3
    nb_block = 3

    launch_packmem2.launch(
        cores,
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
        limx,
        limy,
        precision,
        nb_block,
    )

    # Check Total files
    expected_output_Deep = Path(f"{output_dir}/Total_Deep.csv")
    expected_output_Shallow = Path(f"{output_dir}/Total_Shallow.csv")
    expected_output_All = Path(f"{output_dir}/Total_All.csv")
    with open("tests/data/end_to_end_DLPC/Total_Deep.csv", "r") as f_in:
        expected_content_Deep = f_in.read()

    assert expected_output_Deep.exists()
    assert expected_output_Shallow.exists()
    assert expected_output_All.exists()
    assert expected_output_Deep.read_text() == expected_content_Deep

    # Check pdf
    expected_final_output = Path(f"{output_dir}/DLPC.pdf")
    expected_final_summary = Path(f"{output_dir}/DLPC.csv")

    assert expected_final_output.exists()
    assert expected_final_summary.exists()


def test_launch_DLPC_multiprocess(tmp_path):
    """DLPC (Martini) multiprocess"""
    cores = 4
    topo = "tests/data/end_to_end_DLPC/md.gro"
    traj = "tests/data/end_to_end_DLPC/md_10ns.xtc"
    lipid = "DLPC"
    start = 0
    end = 100
    paramFile = "data/param_Martini.txt"
    radiiFile = "data/vdw_radii_Martini_old.txt"
    indexFile = None
    output_dir = str(tmp_path)
    outputname = "DLPC"
    dist_suppl_Z = 1.0
    protein = False
    pdbout = False
    limx = 15
    limy = 1e-4
    precision = 3
    nb_block = 3

    launch_packmem2.launch(
        cores,
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
        limx,
        limy,
        precision,
        nb_block
    )

    # Check Total files
    expected_output_Deep = Path(f"{output_dir}/Total_Deep.csv")
    expected_output_Shallow = Path(f"{output_dir}/Total_Shallow.csv")
    expected_output_All = Path(f"{output_dir}/Total_All.csv")
    with open("tests/data/end_to_end_DLPC/Total_Deep.csv", "r") as f_in:
        expected_content_Deep = f_in.read()

    assert expected_output_Deep.exists()
    assert expected_output_Shallow.exists()
    assert expected_output_All.exists()
    assert expected_output_Deep.read_text() == expected_content_Deep

    # Check pdf
    expected_final_output = Path(f"{output_dir}/DLPC.pdf")
    expected_final_summary = Path(f"{output_dir}/DLPC.csv")

    assert expected_final_output.exists()
    assert expected_final_summary.exists()
