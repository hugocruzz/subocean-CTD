import pandas as pd
import xarray as xr
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict
import logging
import plotly.express as px

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTDProcessor:
    """Class to process and merge CTD and SubOcean data."""
    
    def __init__(self, data_root: Path, freq: str = "1S"):
        """
        Initialize CTDProcessor.
        
        Args:
            data_root: Root directory containing data
            freq: Frequency for resampling (default: "1S")
        """
        self.data_root = Path(data_root)
        self.subocean_path = self.data_root / 'subocean'
        self.freq = freq
        
    @staticmethod
    def load_ctd_profile(path: Path) -> pd.DataFrame:
        """Load CTD profile with proper encoding."""
        for encoding in ['latin1', 'utf-8']:
            try:
                df = pd.read_csv(path, encoding=encoding)
                df["datetime"] = pd.to_datetime(df["datetime"])
                return df
            except Exception as e:
                logger.debug(f"Failed to load with {encoding}: {e}")
        raise ValueError(f"Could not load CTD profile: {path}")

    def update_metadata(self, metadata_path: Path) -> pd.DataFrame:
        """Update metadata with new SubOcean files."""
        metadata = pd.read_excel(metadata_path, sheet_name="Forel")
        metadata = metadata[metadata["subocean_path"].notna()].copy()
        
        subocean_files = list(self.subocean_path.rglob('*.nc'))
        existing_paths = metadata["subocean_path"].tolist()
        
        new_rows = [
            pd.Series({
                'subocean_path': str(file),
                'CTD path': None,
                'Station': None,
                'Date': None
            }) for file in subocean_files if str(file) not in existing_paths
        ]
        
        if new_rows:
            metadata = pd.concat([metadata, pd.DataFrame(new_rows)], ignore_index=True)
        
        return metadata

    def process_profile_pair(self, row: pd.Series) -> Dict:
        """Process a pair of CTD and SubOcean profiles."""
        # Load datasets
        subocean_ds = xr.open_dataset(row["subocean_path"])
        _, index = np.unique(subocean_ds.datetime, return_index=True)
        subocean_ds = subocean_ds.isel(datetime=index)
        
        ctd_df = self.load_ctd_profile(row["CTD path"])
        ctd_ds = ctd_df.set_index('datetime').to_xarray()
        
        # Validate timestamps
        if not ctd_ds.datetime.isin(subocean_ds.datetime).all():
            logger.warning(f"CTD datetime not fully contained in SubOcean datetime for Station {row['station_ID']}")
        
        # Create common time grid and interpolate
        datetime_grid = pd.date_range(
            ctd_ds.datetime.min().item(),
            ctd_ds.datetime.max().item(),
            freq=self.freq
        )
        
        subocean_interp = subocean_ds.interp(datetime=datetime_grid)
        ctd_interp = ctd_ds.interp(datetime=datetime_grid)
        ctd_interp = ctd_interp.rename({name: name + '_ctd' for name in ctd_interp.data_vars})
        
        # Merge and add metadata
        merged_ds = self._create_merged_dataset(subocean_interp, ctd_interp, row)
        
        try:
            # Prepare dataset for saving
            save_ds = self._prepare_for_save(merged_ds)
            
            # Save outputs
            output_nc = Path(str(row['subocean_path']).replace("L2B", "MERGED") + "_merged.nc")
            output_csv = Path(str(row['subocean_path']).replace("L2B", "MERGED") + f"_{self.freq}_merged.csv")
            
            # Create parent directories if they don't exist
            output_nc.parent.mkdir(parents=True, exist_ok=True)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            
            # Save files
            save_ds.to_netcdf(output_nc)
            merged_ds.to_dataframe().to_csv(output_csv)
            
        except Exception as e:
            logger.error(f"Error saving files for station {row['station_ID']}: {e}")
            raise
        
        return merged_ds
    
    def _create_merged_dataset(self, subocean_ds: xr.Dataset, ctd_ds: xr.Dataset, 
                             row: pd.Series) -> xr.Dataset:
        """Create merged dataset with metadata."""
        merged_ds = xr.merge([subocean_ds, ctd_ds])
        
        # Add attributes
        merged_ds.attrs = {
            'station_id': row['station_ID'],
            'subocean_file': row['subocean_path'],
            'ctd_file': row['CTD path'],
            'processing_date': pd.Timestamp.now().isoformat(),
            **subocean_ds.attrs,
            **ctd_ds.attrs
        }
        

                    
        return merged_ds

    def _get_plottable_variables(self, merged_ds: xr.Dataset) -> Dict[str, List[str]]:
        """Get variables to plot, grouped by type."""
        # Remove diagnostic, FLAG and RSD variables
        exclude_patterns = ['_FLAG', '_RSD', 'Cavity_Pressure', 'Cellule_Temperature', 
                        'LShift', 'Error_Standard', 'Ringdown_Time', 'Box_Temperature',
                        'Box_Pressure', 'PWM', 'Laser', 'Norm_Signal', 'Value_Max',
                            'Delta_13_CH4_permille_', '_C2H6_dissolved_ppm_']
        
        variables = {
            'CH4 Measurements': [var for var in merged_ds.data_vars 
                            if '_CH4_' in var and not any(p in var for p in exclude_patterns)],
            'Environmental': ['Depth_meter_', 'Hydrostatic_Pressure_Calibrated_bar_'],
            'CTD Data': [var for var in merged_ds.data_vars 
                        if '_ctd' in var and not any(p in var for p in exclude_patterns)],
            'Flow Parameters': ['Flow_Carrier_Gas_sccm_', 'Total_Flow_sccm_', 'Dry_gas_Flow_sccm_']
        }
        
        return variables

    def plot_comparison(self, merged_ds: xr.Dataset, station_id: str):
        """Create plot with dual y-axis for CTD and SubOcean data."""
        variables = self._get_plottable_variables(merged_ds)
        
        fig = go.Figure()
        
        # Color schemes for each axis
        subocean_colors = px.colors.qualitative.Set1
        ctd_colors = px.colors.qualitative.Set2
        
        # Plot SubOcean data on primary y-axis
        color_idx = 0
        for group_name, vars in variables.items():
            if group_name != 'CTD Data':
                for var in vars:
                    if var in merged_ds and np.issubdtype(merged_ds[var].dtype, np.number):
                        fig.add_trace(go.Scatter(
                            x=merged_ds['datetime'],
                            y=merged_ds[var],
                            name=var,
                            visible='legendonly',
                            line=dict(color=subocean_colors[color_idx % len(subocean_colors)]),
                            legendgroup=group_name,
                            legendgrouptitle_text=group_name
                        ))
                        color_idx += 1
        
        # Plot CTD data on secondary y-axis
        color_idx = 0
        for var in variables['CTD Data']:
            if var in merged_ds and np.issubdtype(merged_ds[var].dtype, np.number):
                fig.add_trace(go.Scatter(
                    x=merged_ds['datetime'],
                    y=merged_ds[var],
                    name=var,
                    visible='legendonly',
                    line=dict(color=ctd_colors[color_idx % len(ctd_colors)]),
                    legendgroup='CTD Data',
                    legendgrouptitle_text='CTD Data',
                    yaxis='y2'
                ))
                color_idx += 1
        
        # Update layout with secondary y-axis
        fig.update_layout(
            title=f"Station {station_id} - Profile Data",
            xaxis_title="DateTime",
            yaxis_title="SubOcean Measurements",
            yaxis2=dict(
                title="CTD Measurements",
                overlaying='y',
                side='right'
            ),
            height=800,
            showlegend=True,
            legend=dict(
                groupclick="toggleitem",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            hovermode='x unified'
        )
        
        return fig

    def _get_output_paths(self, subocean_path: str, station_id: str) -> dict:
        """Generate output file paths."""
        # Create base path from subocean path
        base_path = Path(str(subocean_path).replace("L2B", "MERGED"))
        base_stem = base_path.stem  # Get filename without extension
        
        return {
            'netcdf': base_path.parent / f"{base_stem}_merged.nc",
            'csv': base_path.parent / f"{base_stem}_{self.freq}_merged.csv",
            'plot': base_path.parent / f"station_{station_id}_all_variables_{self.freq}.html"
        }

    def _save_outputs(self, merged_ds: xr.Dataset, paths: dict) -> None:
        """Save dataset to multiple formats."""
        # Create directories if they don't exist
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save dataset in different formats
            save_ds = self._prepare_for_save(merged_ds)
            save_ds.to_netcdf(paths['netcdf'])
            merged_ds.to_dataframe().to_csv(paths['csv'])
            logger.info(f"Saved outputs to {paths['netcdf'].parent}")
        except Exception as e:
            logger.error(f"Error saving outputs: {e}")
            raise

    def process_all_profiles(self, metadata_path: Path):
        """Process all profiles in metadata."""
        metadata = self.update_metadata(metadata_path)
        matched_profiles = []
        
        for idx, row in metadata.iterrows():
            # Process data
            merged_ds = self.process_profile_pair(row)
            
            # Generate paths and save outputs
            paths = self._get_output_paths(row['subocean_path'], row['station_ID'])
            self._save_outputs(merged_ds, paths)
            
            # Create visualization
            fig = self.plot_comparison(merged_ds, row['station_ID'])
            fig.write_html(paths['plot'])
            
            matched_profiles.append({
                'station': row['station_ID'],
                'merged_ds': merged_ds
            })
                
                
        logger.info(f"Matched {len(matched_profiles)} profile pairs")
        metadata.to_excel("updated_metadata.xlsx", index=False)
        return matched_profiles

    def _prepare_for_save(self, ds: xr.Dataset) -> xr.Dataset:
        """Prepare dataset for saving by handling problematic variables."""
        ds = ds.copy()
        
        # Drop problematic datetime variables
        datetime_vars = ['Date', 'Time', 'Date_calibrated', 'Time_calibrated', 
                        'Date_ctd', 'Time_ctd']
        for var in datetime_vars:
            if var in ds:
                ds = ds.drop_vars(var)
        
        # Convert object dtypes
        for var in ds.variables:
            if ds[var].dtype == 'O':
                try:
                    # Try converting to datetime
                    ds[var] = pd.to_datetime(ds[var])
                except:
                    # If conversion fails, drop the variable
                    ds = ds.drop_vars(var)
                    logger.warning(f"Dropped variable {var} due to incompatible dtype")
        
        return ds

def main():
    """Main execution function."""
    processor = CTDProcessor(
        data_root=Path('../data'),
        freq="1S"
    )
    processor.process_all_profiles(
        metadata_path=Path("data/Activity_log_GF24_Full-STNCTDcorr_path.xls")
    )

if __name__ == "__main__":
    main()