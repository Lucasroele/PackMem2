# `launch_packmem2`

`launch_packmem2` runs the PackMem2 analysis, concatenates the per-frame
results, and generates the final analysis plots.

## Usage

```console
launch_packmem2 -f TRAJECTORY -s STRUCTURE -l LIPID [options]
```

## Options

### Input files

- `-f TRAJECTORY`: trajectory file, such as `.xtc` or `.trr` (required).
- `-s STRUCTURE`: topology file, such as `.gro` (required).
- `-r RADII_FILE`: van der Waals radii file. Default: `data/vdw_radii_Charmm.txt`.
- `-p PARAMETER_FILE`: lipid parameter file. Default: `data/param_Charmm.txt`.
- `-n INDEX_FILE`: optional GROMACS index file containing the upper and lower leaflets.

### Analysis settings

- `-l LIPID`: lipid name, or multiple names separated with `_` (required).
- `-b FRAME`: first frame. Default: `0`.
- `-e FRAME`: last frame.
- `-c CORES`: number of processes. Default: the number of available CPUs minus two.
- `-d DISTANCE`: distance separating deep and shallow defects in Angstroms. Default: `1.0`.
- `-nb BLOCKS`: number of blocks for block averaging. Default: `3`.
- `-prec DIGITS`: number of decimal places in the output. Default: `2`.
- `-lx AREA`: minimum defect area used for fitting. Default: `15`.
- `-ly PROBABILITY`: minimum probability used for fitting. Default: `1e-4`.
- `--wrapped`: assume trajectory coordinates are already wrapped. This is the default.
- `--no-wrapped`: wrap trajectory coordinates before analysis.
- `-prot`: analyze defects near a protein.
- `-pdb`: create PDB output files.

### Output

- `-o NAME`: output name. Default: `output`.
- `-od DIRECTORY`: output directory. Default: `./`.
