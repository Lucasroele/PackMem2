from pathlib import Path
import packmem2.launch_packmem2 as launch_packmem2

def test_launch_POPC_multiprocess(tmp_path):
    """POPC (Charmm) multiprocess"""
    cores = 6
    topo = "tests/data/end_to_end_POPC/POPC_memb_m.gro"
    traj = "tests/data/end_to_end_POPC/160_frames_m.xtc"
    lipid = "POPC"
    start = 0
    end = 159
    paramFile = "data/param_Charmm.txt"
    radiiFile = "data/vdw_radii_Charmm.txt"
    indexFile = None
    output_dir = str(tmp_path)
    outputname = "POPC"
    dist_suppl_Z = 1.0
    protein = False
    pdbout = False
    limx = 15
    limy = 1e-4
    precision = 3
    nb_block = 3
    wrapped = False

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
        wrapped
    )

    # Check Total files
    expected_output_Deep = Path(f"{output_dir}/Total_Deep.csv")
    expected_output_Shallow = Path(f"{output_dir}/Total_Shallow.csv")
    expected_output_All = Path(f"{output_dir}/Total_All.csv")
    with open("tests/data/end_to_end_POPC/Total_Deep.csv", "r") as f_in:
        expected_content_Deep = f_in.read()

    assert expected_output_Deep.exists()
    assert expected_output_Shallow.exists()
    assert expected_output_All.exists()
    assert expected_output_Deep.read_text() == expected_content_Deep

    # Check pdf
    expected_final_output = Path(f"{output_dir}/POPC.pdf")
    expected_final_summary = Path(f"{output_dir}/POPC.csv")

    assert expected_final_output.exists()
    assert expected_final_summary.exists()

    filename_template = "Total{ul_}{depth}.csv"
    sides = ["_Up_", "_Lo_", "_"]
    depths = ["All", "Shallow", "Deep"]

    for side in sides:
        for depth in depths:
            filename = filename_template.format(ul_=side, depth=depth)
            file = Path(output_dir) / filename
            exp_file = Path(f"tests/data/end_to_end_POPC/{filename}")
            assert file.exists()
            assert file.read_text() == exp_file.read_text()