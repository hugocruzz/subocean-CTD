import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

def group_related_parameters(df: pd.DataFrame) -> Tuple[Dict, List]:
    """Group base parameters with their RSD and corrected versions"""
    param_groups = {}
    diagnostic_params = [
        'Cavity Pressure (mbar)', 
        'Cellule Temperature (Degree Celsius)',
        'Hydrostatic pressure (bar)', 
        'LShift', 
        'Error Standard',
        'Ringdown Time (microSec)', 
        'Box Temperature (Degree Celsius)',
        'Box Pressure (mbar)', 
        'PWM Cellule Temperature',
        'PWM Cellule Pressure', 
        'Laser Temperature (Degree Celsius)',
        'Laser Flux', 
        'Norm Signal', 
        'Value Max'
    ]
    #Replace all "(" and spaces by "_" in diagnostic_params
    diagnostic_params = [param.replace(" ", "_").replace("(", "_").replace(")", "") for param in diagnostic_params]
    
    # Filter diagnostic params to only those present in df
    available_diag_params = [p for p in diagnostic_params if p in df.columns]
    missing_diag_params = [p for p in diagnostic_params if p not in df.columns]
    
    # Warning for missing parameters
    for param in missing_diag_params:
        print(f"Warning: Diagnostic parameter '{param}' not found in data")
    
    # Find base parameters (excluding flags, RSD, corrected versions)
    base_params = [col for col in df.columns if 
                  not any(x in col for x in ['_FLAG', '_RSD', 'corrected']) and
                  col not in diagnostic_params and
                  col not in ['Date', 'Time', 'datetime', 'Depth (meter)']]
    
    for param in base_params:
        param_group = {
            'base': param,
            'rsd': next((col for col in df.columns if f"{param}_RSD" in col), None),
            'corrected': next((col for col in df.columns if 'corrected' in col and param in col), None)
        }
        param_groups[param] = param_group
    
    return param_groups, available_diag_params

def create_diagnostic_plot(df: pd.DataFrame, depth_col, downcast_bool_col, params: list, timestamp: str) -> go.Figure:
    """Create diagnostic parameter plots"""
    n_cols = 2
    n_rows = (len(params) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.02
    )
    
    for idx, param in enumerate(params):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Plot downcast
        mask_down = df[downcast_bool_col]
        fig.add_trace(
            go.Scatter(
                x=df[param][mask_down],
                y=df[depth_col][mask_down],
                name='Downcast',
                mode='lines+markers',
                marker=dict(size=4, color='#1f77b4'),
                line=dict(width=1, color='#1f77b4'),
                showlegend=True if idx == 0 else False
            ),
            row=row, col=col
        )
        
        # Plot upcast
        mask_up = ~mask_down
        fig.add_trace(
            go.Scatter(
                x=df[param][mask_up],
                y=df[depth_col][mask_up],
                name='Upcast',
                mode='lines+markers',
                marker=dict(size=4, color='#ff7f0e'),
                line=dict(width=1, color='#ff7f0e'),
                showlegend=True if idx == 0 else False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text=param, row=row, col=col)
        fig.update_yaxes(autorange="reversed", title_text="Depth (m)" if col == 1 else None,
                        row=row, col=col)
    
    fig.update_layout(
        height=600*n_rows,     # Adjusted height per row
        width=1200,
        title_text=f"Diagnostics - {timestamp}",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        font=dict(size=12)  ,      
        margin=dict(t=50, b=50, l=50, r=50)  # Added consistent margins
    )
    
    
    return fig

def create_measurement_plot(df: pd.DataFrame,depth_col, downcast_bool_col, param_groups: Dict, timestamp: str) -> go.Figure:
    """Create measurement plot with separate subplots for base, RSD, and corrected values"""
    n_plots = sum(1 + bool(group['rsd']) + bool(group['corrected']) 
                 for group in param_groups.values())
    
    n_cols = 2  # Changed to 2 columns
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        shared_yaxes=True,
        horizontal_spacing=0.08,  # Reduced spacing
        vertical_spacing=0.02     # Reduced spacing
    )
    
    plot_idx = 0
    for param_name, param_group in param_groups.items():
        # Base parameter plot
        row = plot_idx // n_cols + 1
        col = plot_idx % n_cols + 1
        
        # Plot downcast
        mask_down = df[downcast_bool_col]
        fig.add_trace(
            go.Scatter(
                x=df[param_group['base']][mask_down],
                y=df[depth_col][mask_down],
                name='Downcast',
                mode='lines+markers',
                marker=dict(size=4, color='#1f77b4'),
                line=dict(width=1, color='#1f77b4'),
                showlegend=True if plot_idx == 0 else False
            ),
            row=row, col=col
        )
        
        # Plot upcast
        mask_up = ~mask_down
        fig.add_trace(
            go.Scatter(
                x=df[param_group['base']][mask_up],
                y=df[depth_col][mask_up],
                name='Upcast',
                mode='lines+markers',
                marker=dict(size=4, color='#ff7f0e'),
                line=dict(width=1, color='#ff7f0e'),
                showlegend=True if plot_idx == 0 else False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text=param_name, row=row, col=col)
        fig.update_yaxes(autorange="reversed", title_text="Depth (m)" if col == 1 else None,
                        row=row, col=col)
        plot_idx += 1
        
        # RSD plot if exists
        if param_group['rsd']:
            row = plot_idx // n_cols + 1
            col = plot_idx % n_cols + 1
            
            # Plot downcast RSD
            fig.add_trace(
                go.Scatter(
                    x=df[param_group['rsd']][mask_down],
                    y=df[depth_col][mask_down],
                    name=f"{param_name} RSD Down",
                    mode='lines+markers',
                    marker=dict(size=4, color='#1f77b4'),
                    line=dict(width=1, color='#1f77b4'),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Plot upcast RSD
            fig.add_trace(
                go.Scatter(
                    x=df[param_group['rsd']][mask_up],
                    y=df[depth_col][mask_up],
                    name=f"{param_name} RSD Up",
                    mode='lines+markers',
                    marker=dict(size=4, color='#ff7f0e'),
                    line=dict(width=1, color='#ff7f0e'),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text=f"{param_name} RSD", row=row, col=col)
            fig.update_yaxes(autorange="reversed", title_text="Depth (m)" if col == 1 else None,
                            row=row, col=col)
            plot_idx += 1
        
        # Corrected plot if exists
        if param_group['corrected']:
            row = plot_idx // n_cols + 1
            col = plot_idx % n_cols + 1
            
            # Plot downcast corrected
            fig.add_trace(
                go.Scatter(
                    x=df[param_group['corrected']][mask_down],
                    y=df[depth_col][mask_down],
                    name=f"{param_name} corrected Down",
                    mode='lines+markers',
                    marker=dict(size=4, color='#1f77b4'),
                    line=dict(width=1, color='#1f77b4'),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Plot upcast corrected
            fig.add_trace(
                go.Scatter(
                    x=df[param_group['corrected']][mask_up],
                    y=df[depth_col][mask_up],
                    name=f"{param_name} corrected Up",
                    mode='lines+markers',
                    marker=dict(size=4, color='#ff7f0e'),
                    line=dict(width=1, color='#ff7f0e'),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text=f"{param_name} corrected", row=row, col=col)
            fig.update_yaxes(autorange="reversed", title_text="Depth (m)" if col == 1 else None,
                            row=row, col=col)
            plot_idx += 1
    
    fig.update_layout(
        height=600*n_rows,     # Adjusted height per row
        width=1200,              # Adjusted width
        title_text=f"Measurements - {timestamp}",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        font=dict(size=12),
        margin=dict(t=50, b=50, l=50, r=50)  # Added consistent margins
    
    )
    
    return fig

def save_profile_plots(expedition: str, parameter: str, data_dir: Path, output_dir: Path) -> None:
    """Save measurement and diagnostic plots"""
    param_dir = data_dir / expedition / "Level2" / parameter
    output_dir = output_dir / expedition / parameter
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in param_dir.glob('L2_*.csv'):
        try:
            df = pd.read_csv(file_path)
            param_groups, diag_params = group_related_parameters(df)
            
            # Create and save measurement plot
            timestamp = pd.to_datetime(df['datetime'].iloc[0]).strftime("%Y-%m-%d %H:%M")
            fig_meas = create_measurement_plot(df, param_groups, timestamp)
            meas_file = output_dir / f"{file_path.stem}_measurements.html"
            fig_meas.write_html(str(meas_file))
            
            # Create and save diagnostic plot
            fig_diag = create_diagnostic_plot(df, diag_params, timestamp)
            diag_file = output_dir / f"{file_path.stem}_diagnostics.html"
            fig_diag.write_html(str(diag_file))
            
            print(f"Saved plots for {file_path.stem}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    base_dir = Path.cwd()
    data_dir = base_dir / "data"
    output_dir = base_dir / "output" / "profile_plots"
    
    # Example usage for specific expedition and parameter
    expedition = "forel"
    parameter = "CH4"
    
    save_profile_plots(expedition, parameter, data_dir, output_dir)