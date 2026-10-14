# `calculate_radii`

This script computes the van der Waals radii from .itp files.

The output is given in the terminal:
* `Res_membrane.pdf` that contains all the plots of the defects analysis.\
        In order: the fitted decay plots, the block averaging bar plots and the final bar plots\
        For each defect type

## Command line

The complete options for analysis are:
```console
calculate_radii -mol [molecule file] -ff [forcefield file] -o [output name] [-martini] [-martini3] [-h]
```

## Options

## Options

Input files options:
- `-mol MOLECULE_FILE`: molecule parameter file (`.itp`).
- `-ff FORCEFIELD_FILE`: force-field parameter file (`.itp`).

Input options:
- `-martini`: compute radii using MARTINI information.
- `-martini3`: compute radii using MARTINI 3 information.

Output options:
- `-o NAME`: output name. Default: `vdw_out`.

## Trouble shooting

This script is here for helping compute the radius of each atom. It adds the information of the aliphathic nature of the atom. This should always be double checked as the order the atom appears it the molecule.itp file may vary !