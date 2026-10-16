"""Get the arguments and parameter files."""
# R. Gautier A. Bacle 2015
# M. Zygadlo 2025

import argparse
import numpy as np
from pathlib import Path
import MDAnalysis as mda
import warnings
from multiprocessing import cpu_count


def build_args() -> argparse.Namespace:
    """
    Get the arguments for launch_packmem2 and packmem2.

    --------------------
    OUTPUT
    parser.parse_args
        Contains all the arguments for the script
    """
    parser = argparse.ArgumentParser(description="Arguments for PackMem_prot.py")
    parser.add_argument(
        "-f", action="store", dest="traj", required=True, help="Trajectory file (.xtc)"
    )
    parser.add_argument(
        "-s", action="store", dest="topo", required=True, help="Topology file (.gro)"
    )
    parser.add_argument(
        "-l",
        action="store",
        dest="lipid",
        required=True,
        help="Lipid name(s) in the .gro file.. If multiple, seperate them with _",
    )
    parser.add_argument(
        "-b",
        action="store",
        dest="start",
        type=int,
        default=0,
        help="Frame number to start the analysis (default: 0)",
    )
    parser.add_argument(
        "-e",
        action="store",
        dest="end",
        type=int,
        help="Frame number to end the analysis (default: None)",
    )
    parser.add_argument(
        "-r",
        action="store",
        dest="radiiFile",
        default="data/vdw_radii_Charmm.txt",
        help="File for the atom radii.\nThe options are: data/vdw_radii_Charmm.txt (default), data/vdw_radii_Martini_old.txt, data/vdw_radii_Martini2.txt, data/vdw_radii_Martini2P.txt and data/vdw_radii_Martini3.txt",
    )
    parser.add_argument(
        "-p",
        action="store",
        dest="paramFile",
        default="data/param_Charmm.txt",
        help="File for lipid parameters.\nThe options are: data/param_Charmm.txt (default), data/param_Martini.txt",
    )
    parser.add_argument(
        "-o",
        action="store",
        dest="outputname",
        default="output",
        help="Name for output file (default: output)",
    )
    parser.add_argument(
        "-od",
        action="store",
        dest="output_dir",
        default="./",
        help="Name for output directory (default: ./)",
    )
    parser.add_argument(
        "-d",
        action="store",
        dest="dist_suppl_Z",
        type=float,
        default=1.0,
        help="Distance to differenciate Deep from Shallow defects (default: 1.0)",
    )
    parser.add_argument(
        "-n",
        action="store",
        dest="indexFile",
        help="Index file (Gromacs ndx style) of only Lower/Upper group",
    )
    parser.add_argument(
        "-pdb",
        dest="pdbout",
        action="store_true",
        help="Get .pdb outputs of the packing defects",
    )
    parser.add_argument(
        "-prot",
        dest="protein",
        action="store_true",
        help="Analyse the packing defects close/far of the protein",
    )
    parser.add_argument(
    "--wrapped",
    dest="wrapped",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Trajectory coordinates are already wrapped into the box. "
         "Use --no-wrapped if they are unwrapped and need wrapping first.",
    )
    return parser


def check_ending_frame(end, topo, traj):
    """
    Check if there was a user input of the end argument
    or if the user's end argument is correct

    --------------------
    INPUT
    end: int
        The new ending frame
u    topo: string
        The name of the topology file
    traj: string
        The name of the trajectory file
    """
    u = mda.Universe(topo, traj, to_guess=())
    total_frames = len(u.trajectory)
    if end == None:
        end = total_frames - 1
    elif end >= total_frames:
        warnings.warn("The ending frame number given is greater than the number of frames in the trajectory given, so the ending frame number is set to the last frame of the trajectory.")
        end = total_frames - 1
    return end


def file_present(filename: str) -> None:
    """
    Check if the file exists.

    --------------------
    INPUT
    filename: str
        The name if the file to test the presence of a file
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"ERROR: file '{filename}' not found.")


def get_args_packmem2() -> argparse.Namespace:
    """
    Get the arguments the packmem2 and check that the inputfiles are valid.

    --------------------
    OUTPUT
    parser.parse_args
        Contains all the arguments for packmem2
    """
    # Get the argument parser
    parser = build_args()

    args = parser.parse_args()

    # Check ending frame
    args.end = check_ending_frame(args.end, args.topo, args.traj)    

    # Check that the files exist
    file_present(args.traj)
    file_present(args.topo)
    file_present(args.paramFile)
    file_present(args.radiiFile)

    # Check that the variable d is positive
    if args.dist_suppl_Z < 0.0:
        raise Exception("ERROR : The distance for the -d option must be > 0.0")

    return args


def get_args_launch_packmem2() -> argparse.Namespace:
    """
    Get the arguments for launch_packmem2 and check that the inputfiles are valid.

    --------------------
    OUTPUT
    parser.parse_args
        Contains all the arguments for launch_packmem2
    """
    parser = build_args()
    parser.add_argument(
        "-c",
        action="store",
        dest="cores",
        help="The number of cores you want the script to work on",
        default=max(cpu_count() - 2, 1),
        type=int,
    )
    parser.add_argument(
        "-nb",
        action="store",
        dest="nb_block",
        type=int,
        default=3,
        help="The number of block for block averaging(default: 3)",
    )
    parser.add_argument(
        "-prec",
        action="store",
        dest="precision",
        type=int,
        default=2,
        help="The precision for writing packdef constants (nb of decimals) in the output. Default = 2",
    )
    parser.add_argument(
        "-lx",
        action="store",
        dest="limx",
        type=int,
        default=15,
        help="The lowest defect area used for the fit (we recommand not to touch to this value). Default = 15",
    )
    parser.add_argument(
        "-ly",
        action="store",
        dest="limy",
        type=float,
        default=1e-4,
        help="The lowest probability used for the fit (we recommand not to touch to this value). Default = 1e-4",
    )

    args = parser.parse_args()

    # Check ending frame
    args.end = check_ending_frame(args.end, args.topo, args.traj)

    return args


def read_file(filename: str) -> list:
    """
    Read an input file and test if file is readable.

    --------------------
    INPUT
    filename: str
        Name of the input file

    --------------------
    OUTPUT
    list
        Contains the lines of the file as a list of strings
    """
    try:
        with open(filename) as f:
            data = f.readlines()
    except:
        print(f"ERROR : Something went wrong with the file {filename}")
    return data


def dict_2columns(list_str: list) -> dict:
    """
    Transform list of strings with 2 columns into a dictionary.

    --------------------
    INPUT
    list_str: list
        Contains strings as values

    --------------------
    OUTPUT
    dictionary
        Contains for each string the key as first word and value as second word
    """
    dic = {}
    for line in list_str:
        # Use space a separator
        data = line.strip().split(" ")
        dic[data[0]] = data[1]
    return dic


def set_params(filename: str) -> dict:
    """
    Read the parameter file and return a dictionary of the information.

    --------------------
    INPUT
    filename: str
        Name of the parameter file

    --------------------
    OUTPUT
    dictionary
        Contains the name of the lipids as key and their central atoms as value
    """
    # Read the file
    lines = read_file(filename)
    # Create a dictionary from the file
    resname_glyc = dict_2columns(lines)
    return resname_glyc


def dict_4columns(list_str: list, nb: int) -> dict:
    """
    Transform list of strings with 4 columns into a dictionary.

    --------------------
    INPUT
    list_str: list
        Contains strings as values
    nb: int
        index of the column that will be put as key

    --------------------
    OUTPUT
    dictionary
        Contains for each string the key as first and second word
        and value as third or fourth word
    """
    dic = {}
    for line in list_str:
        # Use space a separator
        data = line.strip().split()
        if nb == 2:
            dic[data[0] + " " + data[1]] = float(data[nb])
        else:
            dic[data[0] + " " + data[1]] = data[nb]
    return dic


def set_rad_ali(filename: str) -> tuple[dict, dict]:
    """
    Read the vdw_radii file and return two dictionaries.

    --------------------
    INPUT
    filename: str
        Name of the parameter file

    --------------------
    OUTPUT
    dictionary
        Contains the radius of the atom
    dictionary
        Contains the aliphatic information of the atom
    """
    lines = read_file(filename)
    radius = dict_4columns(lines, 2)
    aliph = dict_4columns(lines, 3)
    return radius, aliph


def read_ndx(ndx_file: str) -> np.array:
    """
    Read a .ndx file to set the lower/upper residue number lists.

    --------------------
    INPUT
    ndx_file: str
        Name if the index file
    --------------------
    OUTPUT
    arrays numpy
        Contains the residue number of each lipid in the upper/lower leaflet
    """
    # Read the lines in the index file
    lines = read_file(ndx_file)
    # Get the 2nd and last lines into a numpy array
    # Because these lines are where the residue numbers are
    index1 = np.array(lines[1].strip().split(" "), dtype=int)
    index2 = np.array(lines[3].strip().split(" "), dtype=int)

    # If the first line : "[ Upper leaflet ]" or "[ upper leaflet ]"
    if "upper" in lines[0] or "Upper" in lines[0]:
        return index1, index2
    # If the first  line : "[ Lower leaflet ]" or "[ lower leaflet ]"
    else:
        return index2, index1
