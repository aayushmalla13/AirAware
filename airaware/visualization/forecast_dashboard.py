"""Forecast visualization dashboard for AirAware."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for forecast visualization dashboard."""
    figure_size: tuple = (12, 8)
    dpi: int = 100
    colors: List[str] = None
    output_dir: str = "data/artifacts/plots"
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

class ForecastDashboard:
    """Interactive forecast visualization dashboard."""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        logger.info(f"Dashboard initialized - Interactive: {PLOTLY_AVAILABLE}, Static: {MATPLOTLIB_AVAILABLE}")
    
    def plot_forecast_comparison(self, 
                               data: pd.DataFrame,
                               forecasts: Dict[str, Any],
                               actuals: Optional[pd.Series] = None,
                               title: str = "Forecast Comparison") -> Optional[Any]:
        """Plot comparison of multiple forecast models."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for static plots")
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Plot historical data
        if 'pm25' in data.columns:
            ax.plot(data.index, data['pm25'], 
                   color='black', linewidth=1.5, alpha=0.7, 
                   label='Historical Data')
        
        # Plot forecasts
        colors = self.config.colors
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            color = colors[i % len(colors)]
            timestamps = pd.to_datetime(forecast.timestamps)
            ax.plot(timestamps, forecast.predictions, 
                   color=color, linewidth=2, 
                   label=f'{model_name} Forecast')
        
        # Plot actuals if provided
        if actuals is not None:
            ax.plot(actuals.index, actuals.values, 
                   color='red', linewidth=2, linestyle='--',
                   label='Actual Values')
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('PM₂.₅ (μg/m³)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_forecast(self, 
                                data: pd.DataFrame,
                                forecasts: Dict[str, Any],
                                actuals: Optional[pd.Series] = None,
                                title: str = "Interactive Forecast Comparison") -> Optional[Any]:
        """Create interactive forecast comparison plot."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive plots")
            return None
        
        fig = go.Figure()
        
        # Plot historical data
        if 'pm25' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['pm25'],
                mode='lines',
                name='Historical Data',
                line=dict(color='black', width=2),
                opacity=0.7
            ))
        
        # Plot forecasts
        colors = self.config.colors
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            color = colors[i % len(colors)]
            timestamps = pd.to_datetime(forecast.timestamps)
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=forecast.predictions,
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=color, width=3)
            ))
        
        # Plot actuals if provided
        if actuals is not None:
            fig.add_trace(go.Scatter(
                x=actuals.index,
                y=actuals.values,
                mode='lines',
                name='Actual Values',
                line=dict(color='red', width=3, dash='dash')
            ))
        
        # Formatting
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title='Time',
            yaxis_title='PM₂.₅ (μg/m³)',
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600
        )
        
        return fig
    
    def plot_interactive_forecast_with_uncertainty(self, 
                                                 data: pd.DataFrame,
                                                 forecasts: Dict[str, Any],
                                                 actuals: Optional[pd.Series] = None,
                                                 title: str = "Interactive Forecast with Uncertainty") -> Optional[Any]:
        """Create interactive forecast comparison plot with uncertainty bands."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive plots")
            return None
        
        fig = go.Figure()
        
        # Plot historical data
        if 'pm25' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['pm25'],
                mode='lines',
                name='Historical Data',
                line=dict(color='black', width=2),
                opacity=0.7
            ))
        
        # Plot forecasts with uncertainty bands
        colors = self.config.colors
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            color = colors[i % len(colors)]
            timestamps = pd.to_datetime(forecast.timestamps)
            
            # Main forecast line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=forecast.predictions,
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=color, width=3)
            ))
            
            # Add uncertainty bands if available
            if hasattr(forecast, 'confidence_intervals') and forecast.confidence_intervals:
                ci = forecast.confidence_intervals
                
                # 90% confidence interval
                if 'lower_90' in ci and 'upper_90' in ci:
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=ci['upper_90'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=ci['lower_90'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba({self._hex_to_rgb(color)}, 0.2)',
                        name=f'{model_name} 90% CI',
                        hoverinfo='skip'
                    ))
                
                # 50% confidence interval
                if 'lower_50' in ci and 'upper_50' in ci:
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=ci['upper_50'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=ci['lower_50'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba({self._hex_to_rgb(color)}, 0.4)',
                        name=f'{model_name} 50% CI',
                        hoverinfo='skip'
                    ))
        
        # Plot actuals if provided
        if actuals is not None:
            fig.add_trace(go.Scatter(
                x=actuals.index,
                y=actuals.values,
                mode='lines',
                name='Actual Values',
                line=dict(color='red', width=3, dash='dash')
            ))
        
        # Formatting
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title='Time',
            yaxis_title='PM₂.₅ (μg/m³)',
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_quantile_forecast(self, 
                              data: pd.DataFrame,
                              quantile_forecast: Any,
                              actuals: Optional[pd.Series] = None,
                              title: str = "Quantile Forecast") -> Optional[Any]:
        """Create quantile forecast visualization."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive plots")
            return None
        
        fig = go.Figure()
        
        # Plot historical data
        if 'pm25' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['pm25'],
                mode='lines',
                name='Historical Data',
                line=dict(color='black', width=2),
                opacity=0.7
            ))
        
        timestamps = pd.to_datetime(quantile_forecast.timestamps)
        
        # Plot quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        colors = ['rgba(255,0,0,0.1)', 'rgba(255,0,0,0.3)', 'rgba(255,0,0,0.6)', 'rgba(255,0,0,0.3)', 'rgba(255,0,0,0.1)']
        
        for i, (q, color) in enumerate(zip(quantiles, colors)):
            if hasattr(quantile_forecast, f'quantile_{int(q*100)}'):
                quantile_values = getattr(quantile_forecast, f'quantile_{int(q*100)}')
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=quantile_values,
                    mode='lines',
                    name=f'{int(q*100)}th Quantile',
                    line=dict(color=f'rgba(255,0,0,{0.1 + i*0.2})', width=2),
                    opacity=0.8
                ))
        
        # Plot actuals if provided
        if actuals is not None:
            fig.add_trace(go.Scatter(
                x=actuals.index,
                y=actuals.values,
                mode='lines',
                name='Actual Values',
                line=dict(color='red', width=3, dash='dash')
            ))
        
        # Formatting
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title='Time',
            yaxis_title='PM₂.₅ (μg/m³)',
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600
        )
        
        return fig
    
    def plot_model_performance_heatmap(self, 
                                     performance_data: Dict[str, Any],
                                     title: str = "Model Performance Heatmap") -> Optional[Any]:
        """Create performance heatmap by time of day and season."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive plots")
            return None
        
        # Extract performance data
        models = list(performance_data.keys())
        hours = list(range(24))
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        
        # Create heatmap data
        heatmap_data = []
        for model in models:
            model_data = []
            for season in seasons:
                season_data = []
                for hour in hours:
                    # Get performance for this model, season, hour combination
                    key = f"{model}_{season}_{hour}"
                    mae = performance_data.get(key, {}).get('mae', 0)
                    season_data.append(mae)
                model_data.append(season_data)
            heatmap_data.append(model_data)
        
        # Create subplots for each model
        fig = make_subplots(
            rows=len(models), cols=1,
            subplot_titles=[f"{model} Performance" for model in models],
            vertical_spacing=0.1
        )
        
        for i, (model, data) in enumerate(zip(models, heatmap_data)):
            fig.add_trace(
                go.Heatmap(
                    z=data,
                    x=hours,
                    y=seasons,
                    colorscale='RdYlBu_r',
                    showscale=True if i == 0 else False,
                    name=model
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            height=200 * len(models),
            template='plotly_white'
        )
        
        return fig
    
    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB string for Plotly."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"{r},{g},{b}"
    
    def plot_rolling_cv_results(self, 
                              cv_results: Dict[str, Any],
                              title: str = "Rolling-Origin Cross-Validation Results") -> Optional[Any]:
        """Plot rolling-origin cross-validation results."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive plots")
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('MAE by Horizon', 'Average MAE by Model'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot MAE by horizon
        horizons = cv_results.get('horizons', [])
        model_maes = cv_results.get('model_maes', {})
        
        for model, maes in model_maes.items():
            fig.add_trace(
                go.Scatter(x=horizons, y=maes, mode='lines+markers', name=model),
                row=1, col=1
            )
        
        # Plot average MAE by model
        model_mae_avg = {}
        for model, maes in model_maes.items():
            model_mae_avg[model] = np.mean(maes)
        
        fig.add_trace(
            go.Bar(x=list(model_mae_avg.keys()), y=list(model_mae_avg.values()),
                   name='Average MAE', marker_color='lightblue'),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def save_plots(self, 
                  data: pd.DataFrame,
                  forecasts: Dict[str, Any],
                  cv_results: Optional[Dict[str, Any]] = None,
                  actuals: Optional[pd.Series] = None,
                  output_dir: Optional[str] = None) -> Dict[str, str]:
        """Save all plots to files."""
        output_dir = output_dir or self.config.output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save static forecast comparison
        if MATPLOTLIB_AVAILABLE:
            fig = self.plot_forecast_comparison(data, forecasts, actuals)
            if fig:
                static_path = os.path.join(output_dir, "forecast_comparison.png")
                fig.savefig(static_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close(fig)
                saved_files['static_forecast'] = static_path
        
        # Save interactive plots
        if PLOTLY_AVAILABLE:
            # Forecast comparison
            forecast_plot = self.plot_interactive_forecast(data, forecasts, actuals)
            if forecast_plot:
                forecast_path = os.path.join(output_dir, "forecast_comparison.html")
                forecast_plot.write_html(forecast_path)
                saved_files['interactive_forecast'] = forecast_path
            
            # CV results
            if cv_results:
                cv_plot = self.plot_rolling_cv_results(cv_results)
                if cv_plot:
                    cv_path = os.path.join(output_dir, "rolling_cv_results.html")
                    cv_plot.write_html(cv_path)
                    saved_files['cv_results'] = cv_path
        
        logger.info(f"Saved {len(saved_files)} plots to {output_dir}")
        return saved_files