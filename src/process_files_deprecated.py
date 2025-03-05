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

CH4_AIR =  1.8  # ppm
AIR_PRESSURE = 1013.25 
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
        print(f"Pairing Subocean data: {row['subocean_path']} with CTD {row['CTD path']}")
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

        # Calculate membrane enrichment factor
        gas = "_CH4_measured_ppm_"
        Cgas = merged_ds[gas]
        Qcg = merged_ds["Flow_Carrier_Gas_sccm_"] 
        Qtot = merged_ds["Total_Flow_sccm_"]
        C_h2o = merged_ds["_H2O_measured__"]/100
        #Default constant enrichment factor
        merged_ds["meff_CH4_default"] = self.calc_meff_CH4(34,20,0.21)+(merged_ds[gas]*np.zeros(len(merged_ds[gas])))
        merged_ds[gas+"default_constant_enrichment_corrected"] = 1/merged_ds["meff_CH4_default"] * Cgas/(1-Qcg/Qtot - C_h2o)
        #CTD enrichment factor
        merged_ds['meff_CH4'] = self.calculate_meff_CH4(merged_ds)
        merged_ds[gas+"ctd_enrichment_corrected"] = 1/merged_ds["meff_CH4"] * Cgas/(1-Qcg/Qtot - C_h2o)
        #This is the equilibrium gas phase concentration of CH4 in the water, however, it's not the real concentration of CH4 in the water
        #Henr
        # After calculating enrichment factors, add dissolved CH4 calculations
        gas = "_CH4_measured_ppm_"
        
        merged_ds['_CH4_dissolved_with_constant_dry_gas_flow_ppm_'] = merged_ds["_CH4_dissolved_with_water_vapour_ppm_"] * (1 - merged_ds["_H2O_measured__"] / 100)
        # Calculate dissolved CH4 for CTD-corrected values
        merged_ds[gas+"dissolved_nmol_L_ctd"] = self.calc_dissolved_CH4(
            ch4_gas_ppm=merged_ds[gas+"ctd_enrichment_corrected"],
            temperature=merged_ds['Temp_ctd'],
            salinity=merged_ds['Sal_ctd'],
            pressure_mbar=merged_ds['Hydrostatic_Pressure_Calibrated_bar_'] * 1000  # Convert bar to mbar
        )
        
        # Calculate dissolved CH4 for default correction
        merged_ds[gas+"dissolved_nmol_L_default"] = self.calc_dissolved_CH4(
            ch4_gas_ppm=merged_ds[gas+"default_constant_enrichment_corrected"],
            temperature=merged_ds['Temp_ctd'],
            salinity=merged_ds['Sal_ctd'],
            pressure_mbar=merged_ds['Hydrostatic_Pressure_Calibrated_bar_'] * 1000
        )

        # In the process_profile_pair method, after calculating dissolved CH4
        merged_ds[gas+"saturation_level_percentage"] = self.calc_ch4_saturation(
            ch4_dissolved=merged_ds[gas+"dissolved_nmol_L_ctd"],
            temperature=merged_ds['Temp_ctd'],
            salinity=merged_ds['Sal_ctd']
        )
        # Create figure and axis objects with a single subplot
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot CH4 data on primary y-axis
        ax1.plot(merged_ds.datetime, merged_ds[gas], 
                label="CH4 measured ppm", color='blue')
        ax1.plot(merged_ds.datetime, merged_ds["_CH4_dissolved_with_water_vapour_ppm_"], 
                label="CH4 dissolved with water vapour ppm (default)", color='purple')
        
        ax1.plot(merged_ds.datetime, merged_ds["_CH4_dissolved_with_constant_dry_gas_flow_ppm_"],
                label="CH4 dissolved with constant dry gas flow ppm (default)", color='orange') 
        ax1.plot(merged_ds.datetime, merged_ds[gas+"default_constant_enrichment_corrected"], 
                label="sal=34, temp=20, o2=21", color='green')
        ax1.plot(merged_ds.datetime, merged_ds[gas+"ctd_enrichment_corrected"], 
                label="ctd", color='red')

        # Set primary y-axis labels
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('CH4 (ppm)')

        # Create secondary y-axis and plot meff data
        ax2 = ax1.twinx()
        ax2.plot(merged_ds.datetime, merged_ds['meff_CH4'], 
                label="meff ctd", color='orange', linestyle='--')
        ax2.plot(merged_ds.datetime, merged_ds['meff_CH4_default'], 
                label="meff default", color='brown', linestyle='--')

        # Set secondary y-axis label
        ax2.set_ylabel('Membrane Enrichment Factor')

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title('CH4 Measurements and Membrane Enrichment Factors')
        plt.tight_layout()
        plt.show()

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
    
    def calc_meff_CH4(self, salinity: xr.DataArray=34, temperature: xr.DataArray=20, oxygen_pct: xr.DataArray=21) -> xr.DataArray:
        """
        Calculate membrane enrichment factor for CH4.
        Formula: meff_CH4 = (1.9774+(0.0385-0.00316*S)*(T-2.67))*(1+0.2286*(O2-0.2)/(0-0.2))
        
        Args:
            salinity (xr.DataArray): Salinity in PSU
            temperature (xr.DataArray): Water temperature in °C
            oxygen_pct (xr.DataArray): Dissolved oxygen in %
        
        Returns:
            xr.DataArray: Membrane enrichment factor (dimensionless)
        """
        BASE = 1.9774
        TEMP_COEF = 0.0385
        SAL_TEMP_COEF = -0.00316
        TEMP_OFFSET = 2.67
        O2_COEF = 0.2286
        O2_THRESHOLD = 0.2
        
        temp_component = (TEMP_COEF + SAL_TEMP_COEF * salinity) * (temperature - TEMP_OFFSET)
        o2_component = 1 + O2_COEF * (oxygen_pct - O2_THRESHOLD)/(0 - O2_THRESHOLD)
        
        return (BASE + temp_component) * o2_component
        
    def calc_ch4_saturation(self, ch4_dissolved: xr.DataArray, temperature: xr.DataArray, salinity: xr.DataArray, pressure_mbar: xr.DataArray) -> xr.DataArray:
        """
        Calculate CH4 saturation level in water.
        
        Args:
            ch4_dissolved (xr.DataArray): Dissolved CH4 concentration in water (nmol/L)
            temperature (xr.DataArray): Water temperature (°C)
            salinity (xr.DataArray): Salinity (PSU)
            pressure_mbar (xr.DataArray): Pressure (mbar)
        
        Returns:
            xr.DataArray: CH4 saturation level (%)
        """
        # Standard atmospheric concentration of CH4 (ppm)
        P_CH4_air = CH4_AIR
        
        # Convert ppm to atm (1 ppm = 1e-6 atm)
        P_CH4_air_atm = P_CH4_air * 1e-6
        
        # Henry's law constants (Wiesenburg & Guinasso, 1979)
        A = -68.8862
        B = 101.4956
        C = 28.7314
        
        # Convert temperature to Kelvin
        temperature_k = temperature + 273.15
        
        # Calculate Henry's constant
        kh = np.exp(A + B/temperature_k + C * np.log(temperature_k)) * np.exp(-0.0432 * salinity)
        
        # Calculate equilibrium concentration of CH4 in water (nmol/L)
        C_eq = P_CH4_air_atm * kh * pressure_mbar / AIR_PRESSURE * 1e6  # Convert mol/L to nmol/L and adjust for pressure
        
        # Calculate saturation level (%)
        saturation_level = (ch4_dissolved / C_eq) * 100
        
        # Add attributes
        saturation_level.attrs['units'] = '%'
        saturation_level.attrs['long_name'] = 'CH4 Saturation Level'
        saturation_level.attrs['description'] = 'Calculated as the ratio of measured dissolved CH4 to equilibrium concentration'
        
        return saturation_level
    def calc_dissolved_CH4(self, ch4_gas_ppm: xr.DataArray, 
                        temperature: xr.DataArray, 
                        salinity: xr.DataArray,
                        pressure_mbar: xr.DataArray) -> xr.DataArray:
        """
        Calculate dissolved CH4 concentration using Henry's law.
        
        Args:
            ch4_gas_ppm (xr.DataArray): CH4 concentration in gas phase (ppm)
            temperature (xr.DataArray): Water temperature (°C)
            salinity (xr.DataArray): Salinity (PSU)
            pressure_mbar (xr.DataArray): Pressure (mbar)
        
        Returns:
            xr.DataArray: Dissolved CH4 concentration (nmol/L)
        """
        import numpy as np
        
        # Henry's law constants (Wiesenburg & Guinasso, 1979)
        A = -68.8862
        B = 101.4956
        C = 28.7314
        
        # Convert temperature to Kelvin
        temperature_k = temperature + 273.15
        
        # Calculate Henry's constant
        kh = np.exp(A + B/temperature_k + C * np.log(temperature_k)) * np.exp(-0.0432 * salinity)
        
        # Calculate dissolved CH4
        ch4_dissolved = ch4_gas_ppm * pressure_mbar / (kh * 1e6)
        
        # Add attributes
        ch4_dissolved.attrs['units'] = 'nmol/L'
        ch4_dissolved.attrs['long_name'] = 'Dissolved CH4 concentration'
        ch4_dissolved.attrs['description'] = 'Calculated using Henry\'s law (Wiesenburg & Guinasso, 1979)'
        
        return ch4_dissolved
    
    def calculate_meff_CH4(self, merged_ds: xr.Dataset) -> xr.DataArray:
        meff = self.calc_meff_CH4(
            salinity=merged_ds['Sal_ctd'],
            temperature=merged_ds['Temp_ctd'],
            oxygen_pct=merged_ds['O2%_ctd']/100*0.21
        )
        meff.attrs['units'] = 'dimensionless'
        meff.attrs['long_name'] = 'Membrane enrichment factor for CH4'
        return meff

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