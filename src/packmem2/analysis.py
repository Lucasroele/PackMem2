# M. Zygadlo 2025

import argparse
import pandas as pd
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings


def get_arguments() -> argparse.Namespace:
    """
    Get the arguments for the script and check that the inputfiles are valid.

    --------------------
    OUTPUT
    parser.parse_args
    """
    # Getting the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        action="store",
        dest="path",
        default="./",
        help="The path to the Total files. Default = ./",
    )
    parser.add_argument(
        "-n",
        action="store",
        dest="nb_block",
        type=int,
        default=3,
        help="The number of block for block averaging. Default = 3",
    )
    parser.add_argument(
        "-prot", action="store_true", dest="prot", help="If there is a protein"
    )
    parser.add_argument(
        "-pr",
        action="store",
        dest="precision",
        type=int,
        default=3,
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
    parser.add_argument(
        "-o",
        action="store",
        dest="output",
        default="Res_membrane",
        help="The name of the output .pdf file. Default = Res_membrane",
    )
    parser.add_argument(
        "-od",
        action="store",
        dest="output_dir",
        default="./",
        help="Name for output directory (default: ./)",
    )
    args = parser.parse_args()

    return args


def log(y_list: list) -> list:
    """
    Calculate the logarithm of a list.

    --------------------
    INPUT
    y_list: list
        The y values.

    --------------------
    OIUTPUT
    list
        The y values that have been through the logarithm.
    """
    log_y = []
    for y_ind in y_list:
        if y_ind > 0:
            log_y.append(math.log(y_ind))
        else:
            log_y.append(0)
    return log_y


def fit_decay(x: np.array, y: np.array, limx: int | float, limy: float) -> tuple[np.array, float]:
    """
    Function does a linear fit.

    The linear fit is made on the probability of having a certain packing area.

    --------------------
    INPUT
    x: numpy array
        Packing area.
    y: numpy array
        probability of having a certain packing area.
    limx: int
        The lowest defect area used for the fit.
    limy: float
        The lowest probability used for the fit

    --------------------
    OUTPUT
    tuple
        array[float, float]: the fit of the distribution
        float: the r_squared of the fit
    """
    # fit with defects above LIMX nm^2 and proba > LIMY
    y = y[x >= limx]
    x = x[x >= limx]
    x = x[y >= limy]
    y = y[y >= limy]
    if len(x) < 2 or len(y) < 2 or np.all(y == y.flat[0]):
        raise TypeError("Not enough data to perform the fit."
            "\nPlease check that enough frames were given to packmem2"
            " or that you have well supplied all the lipids in your system with the -l option(separated by '_') to packmem2"
            " or that you have not misspelled a lipid.")
    FIT = np.polyfit(x, log(y), 1)
    # As the fit is linear
    r_squared = np.corrcoef(x, log(y))[0, 1] ** 2

    if r_squared < 0.9:
        warnings.warn(f"Warning: r_squared is low ({r_squared:.2f})."
                      "Please provide more frames to packmem2")
    return FIT, r_squared


def plot_defect_fit(
    name: str,
    defect: str,
    fit: np.array,
    x: list,
    y: list,
    r_squared: float,
    limx: int | float,
    limy: float,
    pdf: PdfPages,
) -> None:
    """
    Compute and plot the fit of the defects distribution.

    --------------------
    INPUT
    name: string
        The type to analyse: Total/Total_Up/Total_Lo
    defect: string
        The type  of  defect: Deep/Shallow/All
    fit: numpy array
        Contains the fit on the area data
    x: list
        Contains x data for the plot
    y: list
        Contains the y data for the plot - areas distribution
    r_squared: float
        Measure of the quality of the fit
    limx : int
        The lowest defect area used for the fit
    limy : float
        The lowest probability used for the fit
    pdf : matplotlib.backends.backend_pdf.PdfPages
        Contains the figures in the final pdf
    """

    fit_function = np.poly1d(fit)

    plt.clf()
    plt.scatter(x, log(y), marker="o", facecolor="none", edgecolor="black")
    plt.xlim(-2, 102)
    plt.ylim(-10, -3.5)
    plt.yticks(
        [math.log(1e-4), math.log(1e-3), math.log(1e-2)],
        labels=[str(1e-4), str(1e-3), str(1e-2)],
    )
    plt.ylabel("Probability")
    plt.xlabel("Defect area ${Å^2}$")
    plt.title(f"{defect} {name}")
    plt.axvline(limx, color="gray", linestyle="--")
    plt.axhline(math.log(limy), color="gray", linestyle="-")
    plt.plot(x, fit_function(x), color="red", label=f"Fit (R² = {r_squared:.2f})")
    plt.legend()
    pdf.savefig()  # Save the current figure to the PDF
    plt.close()


def compute_decay(
        name: str,
        defect: str, 
        area_dtf: pd.DataFrame,
        limx: int | float,
        limy: float,
        pdf: PdfPages,
        plot: bool = False
) -> float:
    """
    Compute packing defects distributions
    
    --------------------
    INPUT
    name: string
        The type to analyse: Total/Total_Up/Total_Lo
    defect: string
        The type  of  defect: Deep/Shallow/All
    area_dtf: pandas dataframe
        Conatins the size of the packing defects.
    limx: int
        The lowest defect area used for the fit.
    limy: float
        The lowest probability used for the fit.
    pdf : matplotlib.backends.backend_pdf.PdfPages
        Contains the figures in the final pdf.
    plot: boolean
        If a plot is done for the fit.

    --------------------
    OUTPUT
    float
        The inverse decay.
    """
    H = plt.hist(area_dtf, bins=np.arange(0.5, max(area_dtf) + 1.5))
    # Length of the defects
    x = H[1][:-1] + 0.5
    # nb of observations of a certain defect length
    y = H[0] / sum(H[0])
    FIT, r_squared = fit_decay(x, y, limx, limy)
    if r_squared is np.nan:
        warnings.warn(
            f"r_squared is undefined for {defect} {name}.\n"
            "Please provide more frames to packmem2",
            RuntimeWarning,
            stacklevel=2,
        )
    if plot:
        plot_defect_fit(name, defect, FIT, x, y, r_squared, limx, limy, pdf)

    # compute inv decay
    inv_decay = abs(1 / FIT[0])

    return inv_decay


def block_averaging(
    def_area: pd.DataFrame,
    nb_block: int,
    limx: int | float,
    limy: float,
    name: str,
    defect: str,
    pdf: PdfPages,
) -> list[float, float, float]:
    """
    Divide the packing data into n_block blocks. These blocks are partitions of the simulation trajectory.

    --------------------
    INPUT
    def_area : pandas dataframe
        The size of the packing defect.
    nb_block : int
        The number of blocks we want to have.
    limx : int
        The lowest defect area used for the fit.
    limy : float
        The lowest probability used for the fit.
    name: string
        The type to analyse: Total/Total_Up/Total_Lo
    defect: string
        The type  of  defect: Deep/Shallow/All
    pdf : matplotlib.backends.backend_pdf.PdfPages
        Contains the figures in the final pdf.

    --------------------
    OUTPUT
    list
        A vector of nb_block decays.
    """
    limits = [int((len(def_area) / nb_block) * nb) for nb in range(nb_block + 1)]
    decays = []
    for i in range(nb_block):
        subvect = def_area[limits[i] : limits[i + 1]]
        inv_decay = compute_decay(name, defect, subvect, limx, limy, pdf)
        decays.append(inv_decay)
    return decays


def plot_defect_constants_blocks(
    type: str, packdef_constants: pd.DataFrame, errors: pd.Series, pdf: PdfPages
) -> None:
    """
    Plot the defect constants of all block and their average.

    --------------------
    INPUT
    type: string
        the type to analyse: Total/Total_Up/Total_Lo
    packdef_constants: pandas DataFrame
        Will contains the statistics of the defects found in this function
    errors: pandas Series
        Contains the errors of each defect
    pdf : matplotlib.backends.backend_pdf.PdfPages
        Contains the figures in the final pdf
    """
    # Plot and save the second figure (bar plot for just one row)
    packdef_constants[[f"Deep_{type}", f"Shallow_{type}", f"All_{type}"]][
        :-1
    ].T.plot.bar(
        color=["darkred", "firebrick", "indianred", "lightcoral", "mistyrose", "pink", "palevioletred", "orchid"],
        yerr=errors,
        capsize=3,
        rot=0,
    )
    plt.ylabel("Defect size constant ${Å^2}$")
    plt.ylim(
        0,
        int(
            max(
                packdef_constants[[f"Deep_{type}", f"Shallow_{type}", f"All_{type}"]][
                    :5
                ].max()
            )
        )
        + 2,
    )
    plt.title("Packing defect constants")
    pdf.savefig()  # Save the current figure to the PDF
    plt.close()


def plot_defect_constants(
    type: str, packdef_constants: pd.DataFrame, errors: pd.Series, pdf: PdfPages
) -> None:
    """
    Plot the defects constants for each defect.

    --------------------
    INPUT
    type: string
        the type to analyse: Total/Total_Up/Total_Lo
    packdef_constants: pandas DataFrame
        Will contains the statistics of the defects found in this function
    errors: pandas Series
        Contains the errors of each defect
    pdf : matplotlib.backends.backend_pdf.PdfPages
        Contains the figures in the final pdf
    """
    cst_packing = pd.DataFrame(
        packdef_constants[[f"Deep_{type}", f"Shallow_{type}", f"All_{type}"]]
    )

    # Create the figure
    fig, ax = plt.subplots()

    # Get the variables needed for the graph
    x_index = cst_packing.columns
    packdef_values = cst_packing.loc["cst_mean_blocks"]
    colour = ["firebrick", "forestgreen", "royalblue"]
    text_pos = [0.0, 1.01, 2.01]

    ax.bar(x_index, packdef_values, color=colour, width=0.5, yerr=errors, error_kw=dict(ecolor='darkgrey'), capsize=3)
    # Add the text for the packing constant and the percentage
    for i, defect in enumerate(["Deep", "Shallow", "All"]):
        ax.text(
            text_pos[i],
            packdef_values.loc[f"{defect}_{type}"] + (packdef_values.loc[f"{defect}_{type}"]/100),
            f"{packdef_values.loc[f"{defect}_{type}"]:.1f} $Å^2$",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=12,
        )
    ax.set_ylabel("Defect size constant ${Å^2}$")
    ax.set_ylim(0, int(max(packdef_values)) + (20*max(packdef_values)/100))
    ax.set_title("Packing defect constants computed by block averaging")
    pdf.savefig()  # Save the current figure to the PDF
    plt.close()


def write_packdef_csts(output_dir, output, packdef_csts, precision):
    """
    Write the packing defects results

    --------------------
    INPUT
    output_dir: str
        The name of the output directory
    output: str
        The name of the output file
    packdef_csts: pandas DataFrame
        Contains the statistics of the defects
    precision: float
        The precision in the output file
    """
    packdef_csts = packdef_csts.astype(float)
    packdef_csts.to_csv(f"{output_dir}/{output}.csv", float_format=f"%.{precision}f")


def launch(
    input_path: str,
    output_dir: str,
    output: str,
    prot: bool,
    nb_block: int,
    limx: int | float,
    limy: float,
    precision: int,
) -> None:
    """
    Analyse the defect size computed by PackMem2
    """
    # Open a pdf device
    # Create a single PDF file
    pdf = PdfPages(f"{output_dir}/{output}.pdf")

    columns_dtf = [
        "Deep_Total", "Shallow_Total", "All_Total",
        "Deep_Total_Up", "Shallow_Total_Up", "All_Total_Up",
        "Deep_Total_Lo", "Shallow_Total_Lo", "All_Total_Lo",
    ]

    index_dtf = ["cst_global"]
    for i in range(1, nb_block+1):
        index_dtf.append(f"cst_block{i}")
    index_dtf.append("cst_mean_blocks")
    index_dtf.append("error_mean_blocks")
        
    if prot:
        columns_dtf += [
        "Deep_Total_Up_close", "Shallow_Total_Up_close", "All_Total_Up_close",
        "Deep_Total_Up_far", "Shallow_Total_Up_far", "All_Total_Up_far",
        "Deep_Total_Lo_close", "Shallow_Total_Lo_close", "All_Total_Lo_close",
        "Deep_Total_Lo_far", "Shallow_Total_Lo_far", "All_Total_Lo_far",
        ]
    # Initialize a data frame to store packdef constants + errors
    packdef_csts = pd.DataFrame(columns=columns_dtf, index=index_dtf)

    for name in ["Total", "Total_Up", "Total_Lo"]:
        # Now loop over the three default types
        for defect in ["Deep", "Shallow", "All"]:
            filename = f"{input_path}/{name}_{defect}.csv"
            def_area = pd.read_csv(filename, header=None)[1]

            # Compute the packing defect constant on the whole traj
            global_inv_decay = compute_decay(name, defect, def_area, limx, limy, pdf, plot=True)
            packdef_csts.loc["cst_global", f"{defect}_{name}"] = global_inv_decay

            # Compute the packdef constants on 3 blocks of the traj
            decays_Xblocks = block_averaging(def_area, nb_block, limx, limy, name, defect, pdf)
            for i in range(1, nb_block + 1):
                packdef_csts.loc[f"cst_block{i}", f"{defect}_{name}"] = decays_Xblocks[i-1]

            # Compute the mean of the X blocks averages
            mean_inv_decay = sum(decays_Xblocks) / len(decays_Xblocks)
            # Compute the standard deviation of the 3 values
            error_mean_inv_decay = np.std(decays_Xblocks)
            # Add values to the dtf
            packdef_csts.loc["cst_mean_blocks", f"{defect}_{name}"] = (
                mean_inv_decay
            )
            packdef_csts.loc["error_mean_blocks", f"{defect}_{name}"] = (
                error_mean_inv_decay
            )

        # Plot all packdef constants on a single barplot
        # Allows to estimate the relative convergence of the simulation
        errors = packdef_csts.T.error_mean_blocks.to_frame("cst_mean_blocks")
        plot_defect_constants_blocks(name, packdef_csts, errors, pdf)

        # Plot the final decays + errors computed with block averaging
        # (for each packdef) on a barplot
        plot_defect_constants(
            name,
            packdef_csts,
            errors.loc[
                [f"Deep_{name}", f"Shallow_{name}", f"All_{name}"],
                "cst_mean_blocks",
            ],
            pdf,
        )

    if prot:
        for name in ["Total_Up", "Total_Lo"]:
            # Now loop over the three default types
            for defect in ["Deep", "Shallow", "All"]:
                # Load PackMem data
                filename = f"{input_path}/{name}_{defect}_prot.csv"
                if not Path(filename).is_file():
                    continue
                def_area_prot = pd.read_csv(filename, header=None).iloc[:, 1:]
                def_area_prot_close = def_area_prot[def_area_prot[1] == "close"][2]
                def_area_prot_far = def_area_prot[def_area_prot[1] == "far"][2]

                # Compute the packing defect constant on the whole traj
                global_inv_decay = compute_decay(f"{name}_close", defect, def_area_prot_close, limx, limy, pdf, plot=True)
                packdef_csts.loc["cst_global", f"{defect}_{name}_close"] = global_inv_decay
                global_inv_decay = compute_decay(f"{name}_far", defect, def_area_prot_far, limx, limy, pdf, plot=True)
                packdef_csts.loc["cst_global", f"{defect}_{name}_far"] = global_inv_decay

    
                # Compute the packdef constants on X blocks of the traj
                decays_Xblocks_close = block_averaging(def_area_prot_close, nb_block, limx, limy, f"{name}_close", defect, pdf)
                for i in range(1, nb_block + 1):
                    packdef_csts.loc[f"cst_block{i}", f"{defect}_{name}_close"] = decays_Xblocks_close[i-1]
    
                # Compute the mean of the 3 blocks averages
                mean_inv_decay = sum(decays_Xblocks_close) / len(decays_Xblocks_close)
                # Compute the standard deviation of the 3 values
                error_mean_inv_decay = np.std(decays_Xblocks_close)
                # Add values to the dtf
                packdef_csts.loc["cst_mean_blocks", f"{defect}_{name}_close"] = (
                    mean_inv_decay
                )
                packdef_csts.loc["error_mean_blocks", f"{defect}_{name}_close"] = (
                    error_mean_inv_decay
                )

                decays_Xblocks_far = block_averaging(def_area_prot_far, nb_block, limx, limy, f"{name}_far", defect, pdf)
                for i in range(1, nb_block + 1):
                    packdef_csts.loc[f"cst_block{i}", f"{defect}_{name}_far"] = decays_Xblocks_far[i-1]

                # Compute the mean of the X blocks averages
                mean_inv_decay = sum(decays_Xblocks_far) / len(decays_Xblocks_far)
                # Compute the standard deviation of the 3 values
                error_mean_inv_decay = np.std(decays_Xblocks_far)
                # Add values to the dtf
                packdef_csts.loc["cst_mean_blocks", f"{defect}_{name}_far"] = (
                    mean_inv_decay
                )
                packdef_csts.loc["error_mean_blocks", f"{defect}_{name}_far"] = (
                    error_mean_inv_decay
                )

            if (packdef_csts[
                    [f"Deep_{name}_close",
                     f"Shallow_{name}_close",
                     f"All_{name}_close",
                     f"Deep_{name}_far",
                     f"Shallow_{name}_far",
                     f"All_{name}_far",
                    ]
                ].isna().all().all()
            ):
                continue
    
            # Plot all packdef constants on a single barplot
            # Allows to estimate the relative convergence of the simulation
            errors = packdef_csts.T.error_mean_blocks.to_frame(
                        "cst_mean_blocks"
                    )
            plot_defect_constants_blocks(
                f"{name}_close", packdef_csts, errors, pdf
            )
            plot_defect_constants_blocks(f"{name}_far", packdef_csts, errors, pdf)


            # Plot the final decays + errors computed with block averaging
            # (for each packdef) on a barplot
            plot_defect_constants(
                f"{name}_close",
                packdef_csts,
                errors.loc[
                    [f"Deep_{name}_close", f"Shallow_{name}_close", f"All_{name}_close"],
                    "cst_mean_blocks",
                ],
                pdf,
            )
            plot_defect_constants(
                f"{name}_far",
                packdef_csts,
                errors.loc[
                    [f"Deep_{name}_far", f"Shallow_{name}_far", f"All_{name}_far"],
                    "cst_mean_blocks",
                ],
                pdf,
            )

    # Close the PDF file
    pdf.close()
    
    write_packdef_csts(output_dir, output, packdef_csts, precision)


def main() -> None:
    # Get arguments
    args = get_arguments()

    launch(
         args.path, args.output_dir, args.output, args.prot, args.nb_block, args.limx, args.limy, args.precision
        )


if __name__ == "__main__":
    main()
