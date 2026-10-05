"""Compute packing defects on membranes simulations"""
# R. Gautier A. Bacle april 2016
# R. Gerard may 2024
# M. Zygadlo august 2024

import sys
import time
import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from scipy.spatial.distance import cdist
from packmem2.core import arrays as a
from packmem2.core import matrix as m
from packmem2.core import pdb as pdb
from packmem2.core import connected_component as cc
from packmem2.core import dict as d
from packmem2.core import param as p
from packmem2.core import protein as prot

# Ignore the warning for longdouble due to MDAnalysis' import of h5py
warnings.filterwarnings(
    "ignore",
    message="Signature .* for <class 'numpy.longdouble'> does not match any known type.*",
    category=UserWarning,
)


##########################################################################################
def launch(
    topo: str,
    traj: str,
    lipid: str,
    start: int,
    end: int,
    paramFile: str,
    radiiFile: str,
    indexFile: str | None,
    output_dir: str,
    outputname: str,
    dist_suppl_Z: float,
    protein: bool,
    pdbout: bool,
) -> None:
    """
    Python
    Script to compute Packing defect in flat bilayers
    Lipids parameters adapted to either CHARMM36 (with Klauda corrections) or Martini FF.

    Output files:
        outputname_Up/Lo_Shallow/Deep/All_result.txt
        ## MatrixSize  9646  9801         # Membrane matrix size, Total matrix size
        ## Total   51   582 11.41 6.034   # number of packing defects, total area of packing defects, average size, pourcent of membrane (Membrane matrix size)
        1    1    77.00     2.00          # for each packing defect num size x_position (first pixel) y_mean (first pixel)
        2    1    21.00     3.00
        3    8    59.12     7.00

        if -pdb option:
            outputname_TotalUp/Lo_Shallow/Deep/All.pdb
            Matrix x,y with, for each cell, the value of "packing defect" (in B_factor column, the last column)
            if 0 = Deep packing defects The value increases with the number of atoms in the cell.
            if > 0 and < 1 = Shallow defect
            if -1 = Edge
            ATOM      3   H1 EDG     3       4.000  54.000  56.940  1.00 -1.00
            ATOM      3   H1 MAT     3       4.000  54.000  56.940  1.00  0.00
            ATOM      3   H1 MAT     3       4.000  54.000  56.940  1.00  0.00

            outputname+"_DefectUp/Lo_Shallow/Deep/All.pdb
            Packing defects in pdb format:
            the residue number corresponds to the different packing defects.
            ATOM      5   H1 DEF     2       6.000  22.000  56.940  1.00  2.00
            ATOM      5   H1 DEF     3       6.000  85.000  56.940  1.00  3.00
            ATOM      5   H1 DEF     3       6.000  86.000  56.940  1.00  3.00
            ATOM      6   H1 DEF     3       7.000  86.000  56.940  1.00  3.00

        if -prot option:
            Prot_outputnameFrame_Up/Lo_Deep/Shallow/All_prot.txt
            1,far,2
            2,far,3
            3,close,6
    """
    start_time = time.time()

    ######### READ PARAM FILES #########
    # RESNAME_GLYC = {'DOP': 'C2', 'DOE': 'C2', 'DPP': 'C2', etc}
    RESNAME_GLYC = p.set_params(paramFile)
    # rad = {'ALA N': 1.85, 'ALA HN': 0.22, 'ALA HT1': 0.22, etc}
    # ali = {'ALA N': 'p', 'ALA HN': 'p', 'ALA HT1': 'p', etc }
    # for the amino acids then the lipids
    radius, aliphatic = p.set_rad_ali(radiiFile)

    ######### LOAD UNIVERSE  #########
    u = mda.Universe(topo, traj, to_guess=())
    # If multiple lipids
    lipid_names = lipid.replace("_", " ")
    # select the lipids in system
    lipids = u.select_atoms(f"resname {lipid_names}")
    res_names = list(set(lipids.resnames))

    ######### Extract UPPER/LOWER leaflet ##########
    # Get the glycerol atom name(s)
    glyc_mb = a.get_glyc_lipids(res_names, RESNAME_GLYC)
    # If no index file seperating the leaflets
    if indexFile is None:
        # Create lists of the residue number  for upper and lower leaflets
        L = LeafletFinder(lipids, f"name {glyc_mb}")
        upper_leaflet_ori = np.sort(np.array(list(set(L.groups(0).resids))))
        lower_leaflet_ori = np.sort(np.array(list(set(L.groups(1).resids))))
    else:
        upper_leaflet_ori, lower_leaflet_ori = p.read_ndx(indexFile)

    ############################## Main loop ##################################
    for ts in u.trajectory[start : end + 1]:
        print(f"Frame {ts.frame} {f'/ {end}':>5}", end="\r", flush=True)
        # select all atoms in systems
        system = u.select_atoms(f"resname {lipid_names} or protein")

        # Get all the residues in the membrane
        res_ids = system.resids
        md_uniq_ids = list(set(res_ids))

        # If there are a lot of lipids (> 9999)
        if len(md_uniq_ids) > 9999:
            print("The number of lipids in your membrane is very high (>9999)!")
            print(
                "Please check that you only give to PackMem your membrane (+ protein)"
            )
            sys.exit()

        if protein:
            protein = u.select_atoms("protein")
            zpos_prot = protein.positions[:, 2]

            # Get the zpos of the upper and lower leaflet
            mean_zpos_lipids = np.mean(lipids.positions[:, 2])
            zpos_upper = lipids.positions[lipids.positions[:, 2] > mean_zpos_lipids, 2]
            zpos_lower = lipids.positions[lipids.positions[:, 2] < mean_zpos_lipids, 2]

            # Compute the minimal distance between the protein and the upper / lower leaflet
            dmin_up = np.min(
                (cdist(zpos_upper.reshape(-1, 1), zpos_prot.reshape(-1, 1))) ** 2
            )
            dmin_lo = np.min(
                (cdist(zpos_lower.reshape(-1, 1), zpos_prot.reshape(-1, 1))) ** 2
            )

            resids_prot = np.sort(np.array(list(set(protein.resids))))
            # If the protein is close enough to have an Hbond with the membrane
            if dmin_up < 3.0:
                upper_leaflet = np.concatenate([resids_prot, upper_leaflet_ori])
                lower_leaflet = lower_leaflet_ori
            elif dmin_lo < 3.0:
                upper_leaflet = upper_leaflet_ori
                lower_leaflet = np.concatenate([resids_prot, lower_leaflet_ori])
            else:
                upper_leaflet = upper_leaflet_ori
                lower_leaflet = lower_leaflet_ori
        else:
            upper_leaflet = upper_leaflet_ori
            lower_leaflet = lower_leaflet_ori

        # Get all x, y and z
        x_atoms = system.positions[:, 0].round(2)
        y_atoms = system.positions[:, 1].round(2)
        z_atoms = system.positions[:, 2].round(2)
        coords = np.stack((x_atoms, y_atoms, z_atoms), axis=1)
        # Get membrane dimension
        xmin, xmax, xmean = a.min_max_mean(x_atoms)
        ymin, ymax, ymean = a.min_max_mean(y_atoms)
        zmin, zmax, zmean = a.min_max_mean(z_atoms)

        upper_arrayZ = a.create_arrayZ(
            system.residues, upper_leaflet, RESNAME_GLYC, dist_suppl_Z, zmax
        )
        lower_arrayZ = a.create_arrayZ(
            system.residues, lower_leaflet, RESNAME_GLYC, dist_suppl_Z, zmin, up=False
        )

        # Build an array from xmin-1 to xmax+1 every 1.0
        arrayX = a.create_array(int(xmin - 1), int(xmax + 2), m.SIZE)
        # Build an array from ymin-1 to ymax+1 every 1.0
        arrayY = a.create_array(int(ymin - 1), int(ymax + 2), m.SIZE)

        ####################  Compute Matrix    #################
        Matrix_Up = m.initialize_matrix2D(len(arrayX), len(arrayY), 0.0)
        Matrix_Lo = m.initialize_matrix2D(len(arrayX), len(arrayY), 0.0)

        v = 5.0
        # For each atoms of lipids
        for i, (res_id, atom_name, res_name) in enumerate(
            zip(res_ids, system.names, system.resnames)
        ):
            radius_atm = d.get_value(radius, res_name, atom_name)
            aliph_atom = d.get_value(aliphatic, res_name, atom_name)
            coordtmp = coords[i]
            #### Upper leaflet ####
            if res_id in upper_leaflet:
                # dZ = z_C2_res - z_atom
                dZ = round(m.diff_Z(upper_arrayZ[res_id], coordtmp[2]), 2)
                # If dZ < 5.0
                # To limit the search around an atom to 5 cells
                if dZ < v:
                    # Fill the matrix with value 0 < a < 1 for aliphatic
                    # Or with > 1 if polar OR deep
                    # Defects = 0
                    Matrix_Up = m.fill_matrix(
                        Matrix_Up,
                        radius_atm,
                        aliph_atom,
                        coordtmp,
                        arrayX,
                        arrayY,
                        upper_arrayZ[res_id],
                    )
            #### Lower leaflet ####
            if res_id in lower_leaflet:
                # dZ = z_C2_res - z_atom
                dZ = round(m.diff_Z(lower_arrayZ[res_id], coordtmp[2]), 2)
                # If dfZ > -5.0
                # To limit the search around an atom to 5 cells
                if dZ > -v:
                    # Fill the matrix with value 0 < a < 1 for aliphatic
                    # Or with > 1 if polar OR deep
                    # Defects = 0
                    Matrix_Lo = m.fill_matrix(
                        Matrix_Lo,
                        radius_atm,
                        aliph_atom,
                        coordtmp,
                        arrayX,
                        arrayY,
                        lower_arrayZ[res_id],
                    )

        ####################  Binarise the matrix    #################
        # Initalise matrices to 0.0
        MatrixUp_Deepbin = m.initialize_matrix2D(len(arrayX), len(arrayY), 0.0)
        MatrixLo_Deepbin = m.initialize_matrix2D(len(arrayX), len(arrayY), 0.0)
        MatrixUp_Allbin = m.initialize_matrix2D(len(arrayX), len(arrayY), 0.0)
        MatrixLo_Allbin = m.initialize_matrix2D(len(arrayX), len(arrayY), 0.0)
        MatrixUp_Shallowbin = m.initialize_matrix2D(len(arrayX), len(arrayY), 1.0)
        MatrixLo_Shallowbin = m.initialize_matrix2D(len(arrayX), len(arrayY), 1.0)

        #### Deep ####
        # Binarise
        MatrixUp_Deepbin = m.binarize_matrix_without0(
            Matrix_Up, MatrixUp_Deepbin, -0.01, 0.001
        )
        MatrixLo_Deepbin = m.binarize_matrix_without0(
            Matrix_Lo, MatrixLo_Deepbin, -0.01, 0.001
        )
        # Packing defects determination
        MatrixUp_labels_Deep = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)
        MatrixLo_labels_Deep = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)
        # Connect the packing defects + label them + count the area
        (
            MatrixUp_labels_Deep,
            uniq_labelsUp_Deep,
            area_defectsUp_Deep,
            firstCoorUp_defects_Deep,
        ) = cc.get_connected_components(MatrixUp_Deepbin, MatrixUp_labels_Deep)
        (
            MatrixLo_labels_Deep,
            uniq_labelsLo_Deep,
            area_defectsLo_Deep,
            firstCoorLo_defects_Deep,
        ) = cc.get_connected_components(MatrixLo_Deepbin, MatrixLo_labels_Deep)
        # Get cluster on the edge
        edge_labelsUp_Deep = cc.get_edge_defects(MatrixUp_labels_Deep)
        edge_labelsLo_Deep = cc.get_edge_defects(MatrixLo_labels_Deep)
        # Count area of the edge
        area_edgeUp_Deep = m.count_edge_area(area_defectsUp_Deep, edge_labelsUp_Deep)
        area_edgeLo_Deep = m.count_edge_area(area_defectsLo_Deep, edge_labelsLo_Deep)
        # Clean dico defects (without edge)
        area_defectsUp_Deep = d.del_key_dict(area_defectsUp_Deep, edge_labelsUp_Deep)
        firstCoorUp_defects_Deep = d.del_key_dict(
            firstCoorUp_defects_Deep, edge_labelsUp_Deep
        )
        area_defectsLo_Deep = d.del_key_dict(area_defectsLo_Deep, edge_labelsLo_Deep)
        firstCoorLo_defects_Deep = d.del_key_dict(
            firstCoorLo_defects_Deep, edge_labelsLo_Deep
        )

        # Preserve unfiltered labels for shallow classification.
        MatrixUp_labels_Deep_raw = MatrixUp_labels_Deep.copy()
        MatrixLo_labels_Deep_raw = MatrixLo_labels_Deep.copy()
        
        for edge_lab in edge_labelsUp_Deep:
            ind = np.where(MatrixUp_labels_Deep == edge_lab)
            MatrixUp_labels_Deep[ind] = 0
        for edge_lab in edge_labelsLo_Deep:
            ind = np.where(MatrixLo_labels_Deep == edge_lab)
            MatrixLo_labels_Deep[ind] = 0

        #### All ####
        # Binarise
        MatrixUp_Allbin = m.binarize_matrix_without0(
            Matrix_Up, MatrixUp_Allbin, -0.01, 0.99
        )
        MatrixLo_Allbin = m.binarize_matrix_without0(
            Matrix_Lo, MatrixLo_Allbin, -0.01, 0.99
        )
        # Packing defects determination
        MatrixUp_labels_All = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)
        MatrixLo_labels_All = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)
        # Connect the packing defects + label them + count the area
        (
            MatrixUp_labels_All,
            uniq_labelsUp_All,
            area_defectsUp_All,
            firstCoorUp_defects_All,
        ) = cc.get_connected_components(MatrixUp_Allbin, MatrixUp_labels_All)
        (
            MatrixLo_labels_All,
            uniq_labelsLo_All,
            area_defectsLo_All,
            firstCoorLo_defects_All,
        ) = cc.get_connected_components(MatrixLo_Allbin, MatrixLo_labels_All)
        # Get cluster on the edge
        edge_labelsUp_All = cc.get_edge_defects(MatrixUp_labels_All)
        edge_labelsLo_All = cc.get_edge_defects(MatrixLo_labels_All)
        # Count area of the edge
        area_edgeUp_All = m.count_edge_area(area_defectsUp_All, edge_labelsUp_All)
        area_edgeLo_All = m.count_edge_area(area_defectsLo_All, edge_labelsLo_All)
        # Clean dico defects (without edge)
        area_defectsUp_All = d.del_key_dict(area_defectsUp_All, edge_labelsUp_All)
        firstCoorUp_defects_All = d.del_key_dict(
            firstCoorUp_defects_All, edge_labelsUp_All
        )
        area_defectsLo_All = d.del_key_dict(area_defectsLo_All, edge_labelsLo_All)
        firstCoorLo_defects_All = d.del_key_dict(
            firstCoorLo_defects_All, edge_labelsLo_All
        )

        # Preserve unfiltered labels for shallow classification.
        MatrixUp_labels_All_raw = MatrixUp_labels_All.copy()
        MatrixLo_labels_All_raw = MatrixLo_labels_All.copy()

        for edge_lab in edge_labelsUp_All:
            ind = np.where(MatrixUp_labels_All == edge_lab)
            MatrixUp_labels_All[ind] = 0
        for edge_lab in edge_labelsLo_All:
            ind = np.where(MatrixLo_labels_All == edge_lab)
            MatrixLo_labels_All[ind] = 0

        #### Shallow ####
        # Binarise
        # Get where there are labels
        ind_lab_Deep_Up = np.argwhere(MatrixUp_labels_Deep_raw != 0)
        ind_lab_All_Up = np.argwhere(MatrixUp_labels_All_raw != 0)
        ind_lab_Deep_Lo = np.argwhere(MatrixLo_labels_Deep_raw != 0)
        ind_lab_All_Lo = np.argwhere(MatrixLo_labels_All_raw != 0)
        # Convert them to be array of tuple to compare them
        set_ind_Deep_Up = set(map(tuple, ind_lab_Deep_Up))
        set_ind_Deep_Lo = set(map(tuple, ind_lab_Deep_Lo))
        # Get the  indexes that are in All but not in Deep
        ind_diff_Up = np.array(
            [row for row in ind_lab_All_Up if tuple(row) not in set_ind_Deep_Up]
        )
        ind_diff_Lo = np.array(
            [row for row in ind_lab_All_Lo if tuple(row) not in set_ind_Deep_Lo]
        )
        # Get the indexes that differs
        if len(ind_diff_Up) != 0:
            MatrixUp_Shallowbin[ind_diff_Up[:, 0], ind_diff_Up[:, 1]] = 0.0
        if len(ind_diff_Lo) != 0:
            MatrixLo_Shallowbin[ind_diff_Lo[:, 0], ind_diff_Lo[:, 1]] = 0.0
        # Packing defects determination
        MatrixUp_labels_Shallow = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)
        MatrixLo_labels_Shallow = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)
        # Connect the packing defects + label them + count the area
        (
            MatrixUp_labels_Shallow,
            uniq_labelsUp_Shallow,
            area_defectsUp_Shallow,
            firstCoorUp_defects_Shallow,
        ) = cc.get_connected_components(MatrixUp_Shallowbin, MatrixUp_labels_Shallow)
        (
            MatrixLo_labels_Shallow,
            uniq_labelsLo_Shallow,
            area_defectsLo_Shallow,
            firstCoorLo_defects_Shallow,
        ) = cc.get_connected_components(MatrixLo_Shallowbin, MatrixLo_labels_Shallow)
        # Get cluster on the edge
        edge_labelsUp_Shallow = cc.get_edge_defects(MatrixUp_labels_Shallow)
        edge_labelsLo_Shallow = cc.get_edge_defects(MatrixLo_labels_Shallow)
        # Count area of the edge
        area_edgeUp_Shallow = m.count_edge_area(
            area_defectsUp_Shallow, edge_labelsUp_Shallow
        )
        area_edgeLo_Shallow = m.count_edge_area(
            area_defectsLo_Shallow, edge_labelsLo_Shallow
        )
        # Clean dico defects (without edge)
        area_defectsUp_Shallow = d.del_key_dict(
            area_defectsUp_Shallow, edge_labelsUp_Shallow
        )
        firstCoorUp_defects_Shallow = d.del_key_dict(
            firstCoorUp_defects_Shallow, edge_labelsUp_Shallow
        )
        area_defectsLo_Shallow = d.del_key_dict(
            area_defectsLo_Shallow, edge_labelsLo_Shallow
        )
        firstCoorLo_defects_Shallow = d.del_key_dict(
            firstCoorLo_defects_Shallow, edge_labelsLo_Shallow
        )

        ####################  Output text file  #################
        # Compute the total area of the matrix
        total_area = len(arrayX) * len(arrayY)
        pdb.outputTXT_defects(
            f"{output_dir}/{outputname}{ts.frame}_Up_Deep_result",
            area_defectsUp_Deep,
            firstCoorUp_defects_Deep,
            total_area,
            area_edgeUp_Deep,
            arrayX,
            arrayY,
        )
        pdb.outputTXT_defects(
            f"{output_dir}/{outputname}{ts.frame}_Lo_Deep_result",
            area_defectsLo_Deep,
            firstCoorLo_defects_Deep,
            total_area,
            area_edgeLo_Deep,
            arrayX,
            arrayY,
        )
        pdb.outputTXT_defects(
            f"{output_dir}/{outputname}{ts.frame}_Up_Shallow_result",
            area_defectsUp_Shallow,
            firstCoorUp_defects_Shallow,
            total_area,
            area_edgeUp_Shallow,
            arrayX,
            arrayY,
        )
        pdb.outputTXT_defects(
            f"{output_dir}/{outputname}{ts.frame}_Lo_Shallow_result",
            area_defectsLo_Shallow,
            firstCoorLo_defects_Shallow,
            total_area,
            area_edgeLo_Shallow,
            arrayX,
            arrayY,
        )
        pdb.outputTXT_defects(
            f"{output_dir}/{outputname}{ts.frame}_Up_All_result",
            area_defectsUp_All,
            firstCoorUp_defects_All,
            total_area,
            area_edgeUp_All,
            arrayX,
            arrayY,
        )
        pdb.outputTXT_defects(
            f"{output_dir}/{outputname}{ts.frame}_Lo_All_result",
            area_defectsLo_All,
            firstCoorLo_defects_All,
            total_area,
            area_edgeLo_All,
            arrayX,
            arrayY,
        )

        ####################  Output PDB files  #################
        # final matrix values PD (X,Y) with Z cooresponding to the max(Upper/lowerZlevel)
        if pdbout:
            # Get the max/min of the z_coord+1
            valZmax = float(d.max_value_dict(upper_arrayZ))
            valZmin = float(d.min_value_dict(lower_arrayZ))
            # Write the leaflets into a PDB
            # To ignore the warnings when writing a PDB
            warnings.filterwarnings("ignore", category=UserWarning)
            # Write the leaflets into a PDB
            u.select_atoms(f"resid {upper_leaflet[0]}:{upper_leaflet[-1]}").write(
                f"{output_dir}/{outputname}{ts.frame}_Upper_leaflet.pdb"
            )
            u.select_atoms(f"resid {lower_leaflet[0]}:{lower_leaflet[-1]}").write(
                f"{output_dir}/{outputname}{ts.frame}_Lower_leaflet.pdb"
            )

            # Write the Matrix (each cell) into a PDB
            pdb.outputPDB_Total_matrix(
                f"{output_dir}/{outputname}{ts.frame}_MatrixUp",
                ts.frame,
                arrayX,
                arrayY,
                valZmax,
                Matrix_Up,
            )
            pdb.outputPDB_Total_matrix(
                f"{output_dir}/{outputname}{ts.frame}_MatrixLo",
                ts.frame,
                arrayX,
                arrayY,
                valZmin,
                Matrix_Lo,
            )

            # Write the defects into a PDB
            pdb.outputPDB_defects(
                f"{output_dir}/{output_dir}/{outputname}{ts.frame}_DefectsUp_Deep",
                ts.frame,
                arrayX,
                arrayY,
                valZmax,
                MatrixUp_labels_Deep,
                edge_labelsUp_Deep,
            )
            pdb.outputPDB_defects(
                f"{output_dir}/{outputname}{ts.frame}_DefectsLo_Deep",
                ts.frame,
                arrayX,
                arrayY,
                valZmin,
                MatrixLo_labels_Deep,
                edge_labelsLo_Deep,
            )
            pdb.outputPDB_defects(
                f"{output_dir}/{outputname}{ts.frame}_DefectsUp_Shallow",
                ts.frame,
                arrayX,
                arrayY,
                valZmax,
                MatrixUp_labels_Shallow,
                edge_labelsUp_Shallow,
            )
            pdb.outputPDB_defects(
                f"{output_dir}/{outputname}{ts.frame}_DefectsLo_Shallow",
                ts.frame,
                arrayX,
                arrayY,
                valZmin,
                MatrixLo_labels_Shallow,
                edge_labelsLo_Shallow,
            )
            pdb.outputPDB_defects(
                f"{output_dir}/{outputname}{ts.frame}_DefectsUp_All",
                ts.frame,
                arrayX,
                arrayY,
                valZmax,
                MatrixUp_labels_All,
                edge_labelsUp_All,
            )
            pdb.outputPDB_defects(
                f"{output_dir}/{outputname}{ts.frame}_DefectsLo_All",
                ts.frame,
                arrayX,
                arrayY,
                valZmin,
                MatrixLo_labels_All,
                edge_labelsLo_All,
            )

        ####################  Distance from the protein  #################
        if protein:
            # Select only the protein
            protein = u.select_atoms("protein")

            # Create empty arrays
            MatrixUp_prot_Deep = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)
            MatrixLo_prot_Deep = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)
            MatrixUp_prot_Shallow = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)
            MatrixLo_prot_Shallow = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)
            MatrixUp_prot_All = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)
            MatrixLo_prot_All = m.initialize_matrix2D(len(arrayX), len(arrayY), 0)

            # Find where the protein is in the matrix
            MatrixUp_prot_Deep = prot.find_protein(
                MatrixUp_prot_Deep, "up", protein, arrayX, arrayY, zmean
            )
            MatrixUp_prot_Shallow = prot.find_protein(
                MatrixUp_prot_Shallow, "up", protein, arrayX, arrayY, zmean
            )
            MatrixUp_prot_All = prot.find_protein(
                MatrixUp_prot_All, "up", protein, arrayX, arrayY, zmean
            )
            MatrixLo_prot_Deep = prot.find_protein(
                MatrixLo_prot_Deep, "lo", protein, arrayX, arrayY, zmean
            )
            MatrixLo_prot_Shallow = prot.find_protein(
                MatrixLo_prot_Shallow, "lo", protein, arrayX, arrayY, zmean
            )
            MatrixLo_prot_All = prot.find_protein(
                MatrixLo_prot_All, "lo", protein, arrayX, arrayY, zmean
            )

            # If there are proteins on the upper leaflet
            if len(np.argwhere(MatrixUp_prot_All > 0)) > 0:
                # Get the coordinates of the matrix where the edges of the packing defects are located
                # dictionnary label coords {lab1 = [[x1, y1], [x2, y2], ...], ...}
                coor_label_edgesUp_Deep = prot.find_edges(MatrixUp_labels_Deep)
                coor_label_edgesUp_Shallow = prot.find_edges(MatrixUp_labels_Shallow)
                coor_label_edgesUp_All = prot.find_edges(MatrixUp_labels_All)

                # Get the coordinates of the matrix where the edges of the protein are located.
                # list of list [[x1, y1], [x2, y2], ...]
                coor_prot_edgesUp_Deep = prot.find_edges(MatrixUp_prot_Deep)[1]
                coor_prot_edgesUp_Shallow = prot.find_edges(MatrixUp_prot_Shallow)[1]
                coor_prot_edgesUp_All = prot.find_edges(MatrixUp_prot_All)[1]

                # Assign distance group for each packing defect, "far" or "close". Default threshold = 10 A.
                # dict {lab1 : 'group', lab2 : 'group', ... }
                DefectsUp_labels_group_Deep = prot.assign_dist_group(
                    coor_prot_edgesUp_Deep, coor_label_edgesUp_Deep, 100
                )
                DefectsUp_labels_group_Shallow = prot.assign_dist_group(
                    coor_prot_edgesUp_Shallow, coor_label_edgesUp_Shallow, 100
                )
                DefectsUp_labels_group_All = prot.assign_dist_group(
                    coor_prot_edgesUp_All, coor_label_edgesUp_All, 100
                )

                # Write the result in a text file
                # format : label,dist_group,area
                prot.outputTXT_defects_prot(
                    f"{output_dir}/Prot_{outputname}{ts.frame}_Up_Deep",
                    DefectsUp_labels_group_Deep,
                    area_defectsUp_Deep,
                )
                prot.outputTXT_defects_prot(
                    f"{output_dir}/Prot_{outputname}{ts.frame}_Up_Shallow",
                    DefectsUp_labels_group_Shallow,
                    area_defectsUp_Shallow,
                )
                prot.outputTXT_defects_prot(
                    f"{output_dir}/Prot_{outputname}{ts.frame}_Up_All",
                    DefectsUp_labels_group_All,
                    area_defectsUp_All,
                )

            # If there are proteins on the lower leaflet
            elif len(np.argwhere(MatrixLo_prot_All > 0)) > 0:
                coor_label_edgesLo_Deep = prot.find_edges(MatrixLo_labels_Deep)
                coor_label_edgesLo_Shallow = prot.find_edges(MatrixLo_labels_Shallow)
                coor_label_edgesLo_All = prot.find_edges(MatrixLo_labels_All)

                coor_prot_edgesLo_Deep = prot.find_edges(MatrixLo_prot_Deep)[1]
                coor_prot_edgesLo_Shallow = prot.find_edges(MatrixLo_prot_Shallow)[1]
                coor_prot_edgesLo_All = prot.find_edges(MatrixLo_prot_All)[1]

                DefectsLo_labels_group_Deep = prot.assign_dist_group(
                    coor_prot_edgesLo_Deep, coor_label_edgesLo_Deep, 100
                )
                DefectsLo_labels_group_Shallow = prot.assign_dist_group(
                    coor_prot_edgesLo_Shallow, coor_label_edgesLo_Shallow, 100
                )
                DefectsLo_labels_group_All = prot.assign_dist_group(
                    coor_prot_edgesLo_All, coor_label_edgesLo_All, 100
                )

                prot.outputTXT_defects_prot(
                    f"{output_dir}/Prot_{outputname}{ts.frame}_Lo_Deep",
                    DefectsLo_labels_group_Deep,
                    area_defectsLo_Deep,
                )
                prot.outputTXT_defects_prot(
                    f"{output_dir}/Prot_{outputname}{ts.frame}_Lo_Shallow",
                    DefectsLo_labels_group_Shallow,
                    area_defectsLo_Shallow,
                )
                prot.outputTXT_defects_prot(
                    f"{output_dir}/Prot_{outputname}{ts.frame}_Lo_All",
                    DefectsLo_labels_group_All,
                    area_defectsLo_All,
                )

    print("-- Analysis over --")
    ran_time = round((time.time() - start_time) / 60, 2)
    if ran_time < 1:
        print(f"-- Ran for {round((time.time() - start_time), 2)} second(s) --")
    elif ran_time < 60:
        print(f"-- Ran for {ran_time} minute(s) --")
    else:
        print(f"-- Ran for {round((ran_time / 60), 2)} hour(s) --")


def main() -> None:
    ####### PARAMETERS and INPUT #####
    try:
        args = p.get_args_packmem2()

    except (Exception, FileNotFoundError) as error:
        # Don't print Traceback
        sys.exit(error)

    launch(
        args.topo,
        args.traj,
        args.lipid,
        args.start,
        args.end,
        args.paramFile,
        args.radiiFile,
        args.indexFile,
        args.output_dir,
        args.outputname,
        args.dist_suppl_Z,
        args.protein,
        args.pdbout,
    )


if __name__ == "__main__":
    main()
