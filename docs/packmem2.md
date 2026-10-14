# `packmem2`

`packmem2` reads a trajectory and classifies packing defects as shallow, deep,
or all defects combined.

## Usage

```console
packmem2 -f TRAJECTORY -s STRUCTURE -l LIPID [options]
```

## Options

### Input files

- `-f TRAJECTORY`: trajectory file, such as `.xtc` or `.trr` (required).
- `-s STRUCTURE`: topology file, such as `.gro` (required).
- `-l LIPID`: lipid name, or multiple names separated with `_` (required).
- `-r RADII_FILE`: van der Waals radii file. Default: `data/vdw_radii_Charmm.txt`.
- `-p PARAMETER_FILE`: lipid parameter file. Default: `data/param_Charmm.txt`.
- `-n INDEX_FILE`: optional GROMACS index file containing the upper and lower leaflets.

### Analysis settings

- `-b FRAME`: first frame. Default: `0`.
- `-e FRAME`: last frame (required).
- `-d DISTANCE`: distance separating deep and shallow defects in Angstroms. Default: `1.0`.
- `--wrapped`: assume trajectory coordinates are already wrapped. This is the default.
- `--no-wrapped`: wrap trajectory coordinates before analysis.
- `-prot`: analyze defects near a protein.
- `-pdb`: create PDB output files.

### Output

- `-o NAME`: output name. Default: `output`.
- `-od DIRECTORY`: output directory. Default: `./`.

Per-frame result files use names such as:

```text
NAME7_Up_Deep_result.txt
NAME7_Lo_Shallow_result.txt
NAME7_Up_All_result.txt
```

Each result file contains defect number, area, and the first matrix
coordinates. With `-prot`, additional files contain the defect distance group
(`close` or `far`). With `-pdb`, PDB files contain the matrix or individual
defects.

## Output formatting

The output comprises of multiple .txt file:
* `[outputname][Frame]_[Up/Lo]_[Shallow/Deep/All]_result.txt`\
        ## MatrixSize  9646  9801   # Membrane matrix size, Total matrix size\
        ## Total   51   582 11.41 6.034   # number of packing defects, total area of packing defects, average size, pourcent of membrane (Membrane matrix size)\
        1    1    77.00     2.00          # for each packing defect number, size, x_position (first cell), y_mean (first cell)\
        2    1    21.00     3.00\
        3    8    59.12     7.00
* `Prot_[outputname][Frame]_[Up/Lo]_[Deep/Shallow/All].txt` (if `-prot` flag)\
        1,far,2 # defects number, distance category, size\
        2,far,3\
        3,close,6
* `[outputname]_Total[Up/Lo]_[Shallow/Deep/All].pdb`\
            Matrix x,y with, for each cell, the value of "packing defect" (in B_factor column, the last column)\
            if 0 = Deep packing defects The value increases with the number of atoms in the cell.\
            if > 0 and < 1 = Shallow defect\
            if -1 = Edge\
            ATOM      3   H1 EDG     3       4.000  54.000  56.940  1.00 -1.00\
            ATOM      3   H1 MAT     3       4.000  54.000  56.940  1.00  0.00\
            ATOM      3   H1 MAT     3       4.000  54.000  56.940  1.00  0.00
* `[outputname]_Defect[Up/Lo]_[Shallow/Deep/All].pdb`
            Packing defects in pdb format:\
            the residue number corresponds to the different packing defects.\
            ATOM      5   H1 DEF     2       6.000  22.000  56.940  1.00  2.00\
            ATOM      5   H1 DEF     3       6.000  85.000  56.940  1.00  3.00\
            ATOM      5   H1 DEF     3       6.000  86.000  56.940  1.00  3.00\
            ATOM      6   H1 DEF     3       7.000  86.000  56.940  1.00  3.00