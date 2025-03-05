# CTD Processing Summary

Based on the code analysis of your CTD processing framework, here's a summary of the calculated variables and corrections:

## Corrections Applied

1. **Air Data Removal**
   - Filtering out data where conductivity < 1
   - Calculating median pressure offset from air data
   - Calculating median oxygen saturation offset (median - 100%)
   - Applying these offsets to the pressure and oxygen saturation columns

2. **Quality Control**
   - Removing pH values outside the range 6.5-9 (replaced with NaN)

## Variables Calculated

1. **Thermodynamic Variables**
   - Absolute Salinity (SA)
   - Conservative Temperature (CT)
   - Potential Temperature (`pot_temp_C`)
   - Density (`density_kg_m3`)
   - Brunt-Väisälä frequency squared (`N2`)

2. **Oxygen Variables**
   - Oxygen solubility in mL/L (`o2_solubility_mll_{ctd_type}`)
   - Oxygen solubility in mg/L (`o2_solubility_mgl_{ctd_type}`)
   - Oxygen concentration in mg/kg (`o2_mgkg_{ctd_type}`)
   - Oxygen concentration in mg/L (`o2_mgl_{ctd_type}`)

3. **Profile Classification**
   - Downcast identification (`is_downcast`) based on maximum depth/pressure

4. **CTD-specific Calculations**
   - For SeaBird CTD: Recalculating salinity using Idronaut equation
     - Original SeaBird salinity preserved with suffix (`_seabird`)

5. **Mixed Layer Depth** (in separate function)
   - MLD based on temperature criterion (`mld_temp`)
   - MLD based on density criterion (`mld_dens`)

The code also handles standardization of parameter names across different CTD types (SeaBird and Idronaut) to ensure consistent processing regardless of the original data format.

# Unified Validation Configuration Summary
## Standard Ranges

These define acceptable value ranges for various oceanographic and instrument parameters:

- **Instrument Health Parameters**:
  - Cavity Pressure: 29.5-30.5 mbar
  - Cellule Temperature: 39.5-40.5 °C
  - Flow Carrier Gas: 0-10 sccm
  - Total Flow: 0-100 sccm
  - Ringdown time: 10-30 μs (comment notes: should be 13±1 μs for CH₄ and 26±1 μs for N₂O)

- **Data Quality Parameter**:
  - Error Standard: 0-0.1

- **Measurement Parameters**:
  - Depth: -2 to 11000 meters
  - [C₂H₆] dissolved: 0-100 ppm
  - Delta ¹³CH₄: -15000 to 15000 per-mille
  - [CH₄] dissolved with water vapor: 0-100 ppm and 0-100 nmol/L
  - [N₂O] dissolved with water vapor: 0-100 ppm and 0-100 nmol/L

## Gas Rules

Contains additional validation rules for specific gas measurements:

- **[CH₄] measured (ppm)**:
  - Range: 0-100 ppm
  - RSD (Relative Standard Deviation) threshold: 1%

Values outside these ranges get flagged during processing, and rows with flagged values (particularly those with Error Standard flags) are filtered out in the L1B data stage.