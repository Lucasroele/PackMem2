# `analysis`

This script runs the analysis of the packmem2 results.

The output is a single .pdf file:
* `Res_membrane.pdf` that contains all the plots of the defects analysis.\
        In order: the fitted decay plots, the block averaging bar plots and the final bar plots\
        For each defect type

## Command line

The complete options for analysis are:
```console
analysis -p [precision wanted] -lx [limit in x] -ly [limit in y] -o [output name] -od [output directory] [-prot] [-h]
```

## Options

Input options:
- `-p INPUT_PATH`: directory containing the concatenated CSV files. Default: `./`.
- `-n BLOCKS`: number of blocks for block averaging. Default: `3`.
- `-prot`: include protein-distance results.
- `-pr DIGITS`: precision for defect constants. Default: `3`.
- `-lx AREA`: minimum defect area used for fitting. Default: `15`.
- `-ly PROBABILITY`: minimum probability used for fitting. Default: `1e-4`.

Output options:
- `-o NAME`: output PDF name. Default: `Res_membrane`.
- `-od DIRECTORY`: output directory. Default: `./`.