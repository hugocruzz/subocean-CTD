# %% [markdown]
# # Subocean and CTD data merging + correction
# The goal of this notebook is to show how the subocean data is currently beeing processed, and to have a base for asking the questions
# 

# %%
import numpy as np
import pandas as pd
import os 
import glob
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from profile_plot import group_related_parameters, create_diagnostic_plot, create_measurement_plot
# %%
def restore_datetime_after_groupby(subocean_downard, subocean_downard_unique):
    """
    Restore datetime information lost during groupby operation.
    
    This function efficiently adds datetime information back to the grouped dataset
    by creating a mapping from pressure to datetime values.
    
    Args:
        subocean_downard: Original dataset with datetime information
        subocean_downard_unique: Grouped dataset that lost datetime information
        
    Returns:
        xarray.Dataset: The grouped dataset with datetime information restored
    """
    # Fast way to create a mapping from pressure to datetime using a dictionary
    pressure_to_datetime = {}
    for p, dt in zip(subocean_downard.Hydrostatic_Pressure_Calibrated_bar.values, 
                     subocean_downard.datetime.values):
        if p not in pressure_to_datetime:  # Only keep first occurrence
            pressure_to_datetime[p] = dt
    
    # Use vectorized approach to create datetime array
    # Convert to item() to avoid numpy scalar type issues
    datetime_array = np.array([pressure_to_datetime.get(p.item(), np.datetime64('NaT')) 
                              for p in subocean_downard_unique.Hydrostatic_Pressure_Calibrated_bar.values])
    
    # Add datetime as a variable with the pressure coordinate
    subocean_downard_unique['datetime'] = ('Hydrostatic_Pressure_Calibrated_bar', datetime_array)
    
    return subocean_downard_unique

def process_files(ctd_file: str, subocean_file: str) -> None:
    """
    Process a pair of CTD and SubOcean files and save the merged results
    
    Args:
        ctd_file (str): Path to CTD file
        subocean_file (str): Path to SubOcean file
    """
    print(f"Processing:\nCTD: {ctd_file}\nSubocean: {subocean_file}")
    
    # Open datasets
    subocean_ds = xr.open_dataset(subocean_file).load()
    ctd_df = pd.read_csv(ctd_file)
    ctd_ds = ctd_df.to_xarray()
    
    # Rename pressure column in CTD data
    CTD_pressure_col = "pressure_dbar"
    ctd_ds = ctd_ds.rename_vars({CTD_pressure_col: "Pres"})
    
    # Set depth as coordinates
    ctd_ds = ctd_ds.swap_dims({'index': 'Pres'})
    ctd_ds = ctd_ds.set_coords('Pres')
    ctd_ds = ctd_ds.drop_vars('index')
    ctd_ds["Pres"] = ctd_ds["Pres"] / 10  # Convert from bar to dbar
    
    subocean_ds = subocean_ds.swap_dims({'datetime': 'Hydrostatic_Pressure_Calibrated_bar'})
    subocean_ds = subocean_ds.set_coords('Hydrostatic_Pressure_Calibrated_bar')

    
    # Separate downward profile
    max_pressure_ctd = ctd_ds["Pres"].argmax()
    if "downward" in subocean_file:
        ctd_ds_downard = ctd_ds.isel(Pres=slice(None, max_pressure_ctd.values))
    else:
        ctd_ds_downard = ctd_ds.isel(Pres=slice(max_pressure_ctd.values, None))
    subocean_downard = subocean_ds
    # Create pressure grid
    pressure_grid = np.linspace(subocean_downard.Hydrostatic_Pressure_Calibrated_bar.min(),
                              subocean_downard.Hydrostatic_Pressure_Calibrated_bar.max(),
                              subocean_downard.datetime.shape[0])

    # Perform the groupby operation
    subocean_downard_unique = subocean_downard.groupby("Hydrostatic_Pressure_Calibrated_bar").first()

    # Restore datetime in a single function call
    subocean_downard_unique = restore_datetime_after_groupby(subocean_downard, subocean_downard_unique)

    ctd_ds_downard_unique = ctd_ds_downard.groupby("Pres").first()
    #Rename "Pres" to "Hydrostatic_Pressure_Calibrated_bar" to match subocean
    subocean_interpolated = subocean_downard_unique.interp(Hydrostatic_Pressure_Calibrated_bar=pressure_grid)
    ctd_interpolated = ctd_ds_downard_unique.interp(Pres=pressure_grid)
    # Merge data
    #Add "_ctd" to the CTD variables to avoid conflicts
    ctd_interpolated = ctd_interpolated.rename({var: var + "_ctd" for var in ctd_interpolated.data_vars})
    ctd_interpolated = ctd_interpolated.rename({"Pres": "Hydrostatic_Pressure_Calibrated_bar"})
    merged_ds = xr.merge([subocean_interpolated, ctd_interpolated], compat="override")
    
    # Calculate oxygen percent
    merged_ds["Oxygen_percent_ctd"] = merged_ds["oxygen_saturation_percent_ctd"] * 0.21
    
    # Convert to dataframe and save
    merged_df = merged_ds.to_dataframe()
    outfile = os.path.basename(subocean_file).replace(".nc", "_CTD_merged.csv")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(subocean_file)), "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    merged_df.to_csv(os.path.join(output_dir, outfile))
    print(f"Saved to {os.path.join(output_dir, outfile)}")

    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        
        # Add datetime column if not present
        if 'datetime' not in merged_df.columns:
            # Use the first datetime from subocean file
            merged_df['datetime'] = pd.to_datetime(subocean_ds.datetime.values[0])
        
        # Get parameter groups and diagnostic params
        param_groups, diag_params = group_related_parameters(merged_df)
        
        # Create timestamp for plot titles
        timestamp = pd.to_datetime(merged_df['datetime'].iloc[0]).strftime("%Y-%m-%d %H:%M") if 'datetime' in merged_df.columns else "Unknown"
        
        # Generate and save measurement plot
        fig_meas = create_measurement_plot(merged_df, "depth_meter", "is_downcast", param_groups, timestamp)
        meas_path = os.path.join(plots_dir, outfile.replace("_CTD_merged.csv", "_measurements.html"))
        fig_meas.write_html(meas_path)
        
        # Generate and save diagnostic plot if diagnostic params exist
        if diag_params:
            fig_diag = create_diagnostic_plot(merged_df, "depth_meter", "is_downcast", diag_params, timestamp)
            diag_path = os.path.join(plots_dir, outfile.replace("_CTD_merged.csv", "_diagnostics.html"))
            fig_diag.write_html(diag_path)
        
        print(f"Plots saved to {plots_dir}")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")
# Close datasets to free memory
    subocean_ds.close()
    
def process_folder(subfolder: str, ctd_path: str, subocean_path: str) -> None:
    """
    Process all files in a subfolder
    
    Args:
        subfolder (str): Name of the subfolder
        ctd_path (str): Base path for CTD files
        subocean_path (str): Base path for SubOcean files
    """
    print(f"\nProcessing folder: {subfolder}")
    
    # Get file paths
    directory_ctd = os.path.join(ctd_path, subfolder)
    directory_subocean = os.path.join(subocean_path, subfolder)
    
    ctd_files = glob.glob(os.path.join(directory_ctd, "*.csv"))
    subocean_files = glob.glob(os.path.join(directory_subocean, "*.nc"))
    
    if not ctd_files:
        print(f"Warning: No CTD files found in {subfolder}")
        return
    
    if not subocean_files:
        print(f"Warning: No SubOcean files found in {subfolder}")
        return
    
    # Use the first CTD file if multiple exist
    ctd_file = ctd_files[0]
    if len(ctd_files) > 1:
        print(f"Warning: Multiple CTD files found in {subfolder}, using {os.path.basename(ctd_file)}")
    
    # Process each SubOcean file with the CTD file
    for subocean_file in subocean_files:
    
        process_files(ctd_file, subocean_file)

# %%
# List subfolders
ctd_path = "C:/Users/cruz/Documents/SENSE/CTD_processing/data/Level1/Forel-GroupedStn"
subocean_path = "C:/Users/cruz/Documents/SENSE/SubOcean/data/Level2/L2B/Forel-GroupedStn"

subfolders = [f.path for f in os.scandir(ctd_path) if f.is_dir()]

print("Available subfolders:")
folderlist = []
for folder in subfolders:
    sub_folders = folder.split("/")[-1]
    sub_folders = sub_folders.split("\\")[-1]
    print(f"- {sub_folders}")
    folderlist.append(sub_folders)

# %%
# Process all folders
for subfolder in folderlist:
    process_folder(subfolder, ctd_path, subocean_path)




