"""
Streamlit UI for AirAware

This module provides the main Streamlit application for interactive air quality forecasting
with comprehensive guidance, prevention tips, and multi-location support.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time

# Page configuration
st.set_page_config(
    page_title="AirAware - PM₂.₅ Nowcasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .good-air {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .moderate-air {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .unhealthy-air {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .hazardous-air {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .prevention-tip {
        background-color: #e2e3e5;
        border-left: 4px solid #28a745;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .location-selector {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"


class AirAwareUI:
    """Main UI class for AirAware"""
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.session = requests.Session()
        
    def check_api_health(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.session.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_stations(self) -> List[Dict[str, Any]]:
        """Get available stations"""
        try:
            response = self.session.get(f"{self.api_base_url}/stations", timeout=10)
            if response.status_code == 200:
                return response.json()["stations"]
            return []
        except:
            return []
    
    def get_models(self) -> Dict[str, Any]:
        """Get available models"""
        try:
            response = self.session.get(f"{self.api_base_url}/models", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def generate_forecast(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate forecast"""
        try:
            response = self.session.post(
                f"{self.api_base_url}/forecast",
                json=request_data,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def analyze_explainability(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze explainability"""
        try:
            response = self.session.post(
                f"{self.api_base_url}/explainability",
                json=request_data,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def analyze_what_if(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze what-if scenarios"""
        try:
            response = self.session.post(
                f"{self.api_base_url}/what-if",
                json=request_data,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def execute_agent(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute an intelligent agent"""
        try:
            response = self.session.post(
                f"{self.api_base_url}/agents/execute",
                json=request_data,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_system_status(self) -> Optional[Dict[str, Any]]:
        """Get system status including agent information"""
        try:
            response = self.session.get(f"{self.api_base_url}/system/status", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None


def get_air_quality_info(pm25: float) -> Dict[str, Any]:
    """Get air quality information based on PM2.5 level"""
    if pm25 <= 12:
        return {
            "level": "Good",
            "color": "green",
            "description": "Air quality is satisfactory, and air pollution poses little or no risk.",
            "css_class": "good-air",
            "icon": "🟢"
        }
    elif pm25 <= 35:
        return {
            "level": "Moderate",
            "color": "yellow", 
            "description": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.",
            "css_class": "moderate-air",
            "icon": "🟡"
        }
    elif pm25 <= 55:
        return {
            "level": "Unhealthy for Sensitive Groups",
            "color": "orange",
            "description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
            "css_class": "unhealthy-air",
            "icon": "🟠"
        }
    elif pm25 <= 150:
        return {
            "level": "Unhealthy",
            "color": "red",
            "description": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.",
            "css_class": "unhealthy-air",
            "icon": "🔴"
        }
    else:
        return {
            "level": "Hazardous",
            "color": "purple",
            "description": "Health warnings of emergency conditions. The entire population is more likely to be affected.",
            "css_class": "hazardous-air",
            "icon": "🟣"
        }


def get_prevention_tips(pm25: float) -> List[str]:
    """Get prevention tips based on PM2.5 level"""
    if pm25 <= 12:
        return [
            "✅ Enjoy outdoor activities - air quality is good",
            "✅ Keep windows open for fresh air circulation",
            "✅ Consider outdoor exercise and activities"
        ]
    elif pm25 <= 35:
        return [
            "⚠️ Sensitive individuals should consider reducing prolonged outdoor exertion",
            "🪟 Keep windows closed if you have respiratory issues",
            "😷 Consider wearing a mask if you're sensitive to air pollution"
        ]
    elif pm25 <= 55:
        return [
            "🚫 Avoid outdoor activities if you have heart or lung disease",
            "😷 Wear N95 masks when going outside",
            "🏠 Stay indoors with windows and doors closed",
            "💨 Use air purifiers if available"
        ]
    elif pm25 <= 150:
        return [
            "🚫 Avoid all outdoor activities",
            "😷 Wear N95 or better masks if you must go outside",
            "🏠 Stay indoors with windows and doors closed",
            "💨 Use air purifiers and avoid activities that increase indoor pollution",
            "🚗 Avoid driving and reduce vehicle emissions"
        ]
    else:
        return [
            "🚨 EMERGENCY: Stay indoors at all times",
            "😷 Wear N95 or better masks if you must go outside",
            "🏠 Seal windows and doors, use air purifiers",
            "💨 Avoid all activities that create indoor pollution",
            "🚑 Seek medical attention if you experience breathing difficulties",
            "📞 Follow local health department emergency guidelines"
        ]


def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">🌍 AirAware - PM₂.₅ Nowcasting System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Welcome to AirAware!</strong> This system provides 6-24 hour PM₂.₅ predictions 
        with calibrated uncertainty bands. Get real-time air quality forecasts, understand 
        what the data means, and learn how to protect yourself from air pollution.
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(ui: AirAwareUI):
    """Render the sidebar with configuration options"""
    st.sidebar.header("🔧 Configuration")
    
    # API Health Check
    if ui.check_api_health():
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.markdown("""
        <div class="warning-box">
            <strong>API Connection Issue:</strong> Please ensure the FastAPI server is running 
            on localhost:8000. You can start it with:<br><br>
            <code>python -m airaware.api.app</code>
        </div>
        """, unsafe_allow_html=True)
        return None
    
    # Location Selection (for future expansion)
    st.sidebar.subheader("🌍 Location Selection")
    st.sidebar.markdown("""
    <div class="location-selector">
        <strong>Current Coverage:</strong><br>
        🇳🇵 Nepal - Kathmandu Valley<br><br>
        <em>More locations coming soon!</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Country Selection (placeholder for future)
    country_options = {"Nepal": "NP"}
    selected_country = st.sidebar.selectbox(
        "Country:",
        options=list(country_options.keys()),
        index=0,
        disabled=True  # Disabled for now since we only have Nepal
    )
    
    # City Selection (placeholder for future)
    city_options = {"Kathmandu Valley": "kathmandu"}
    selected_city = st.sidebar.selectbox(
        "City/Region:",
        options=list(city_options.keys()),
        index=0,
        disabled=True  # Disabled for now since we only have Kathmandu
    )
    
    # Get available data
    stations = ui.get_stations()
    models = ui.get_models()
    
    if not stations:
        st.sidebar.error("No stations available")
        return None
    
    if not models:
        st.sidebar.error("No models available")
        return None
    
    # Station Selection
    st.sidebar.subheader("📍 Station Selection")
    station_options = {f"{s['name']} ({s['station_id']})": s['station_id'] for s in stations}
    selected_station = st.sidebar.selectbox(
        "Choose a station:",
        options=list(station_options.keys()),
        index=0
    )
    selected_station_id = station_options[selected_station]
    
    # Model Selection
    st.sidebar.subheader("🤖 Model Selection")
    available_models = {k: v for k, v in models.items() if v.get('is_available', False)}
    if not available_models:
        st.sidebar.error("No available models")
        return None
    
    model_options = {f"{k.upper()} (MAE: {v.get('performance_metrics', {}).get('mae', 'N/A')})": k 
                    for k, v in available_models.items()}
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        options=list(model_options.keys()),
        index=0
    )
    selected_model_type = model_options[selected_model]
    
    # Forecast Horizon
    st.sidebar.subheader("⏰ Forecast Horizon")
    horizon_options = {"6 hours": 6, "12 hours": 12, "24 hours": 24}
    selected_horizon = st.sidebar.selectbox(
        "Forecast horizon:",
        options=list(horizon_options.keys()),
        index=2  # Default to 24 hours
    )
    horizon_hours = horizon_options[selected_horizon]
    
    # Uncertainty Level
    st.sidebar.subheader("📊 Uncertainty Level")
    uncertainty_options = {"80%": 0.8, "90%": 0.9, "95%": 0.95}
    selected_uncertainty = st.sidebar.selectbox(
        "Confidence level:",
        options=list(uncertainty_options.keys()),
        index=1  # Default to 90%
    )
    uncertainty_level = uncertainty_options[selected_uncertainty]
    
    # Additional Options
    st.sidebar.subheader("🔍 Additional Options")
    include_explanations = st.sidebar.checkbox("Include feature explanations", value=False)
    
    # Create station names mapping for cross-station tab
    station_names = {s['station_id']: s['name'] for s in stations}
    
    return {
        "station_id": selected_station_id,
        "model_type": selected_model_type,
        "horizon_hours": horizon_hours,
        "uncertainty_level": uncertainty_level,
        "include_explanations": include_explanations,
        "language": "en",  # Default language
        "country": country_options[selected_country],
        "city": city_options[selected_city],
        "available_stations": [s['station_id'] for s in stations],
        "station_names": station_names
    }


def render_forecast_tab(ui: AirAwareUI, config: Dict[str, Any]):
    """Render the forecast tab"""
    st.header("📈 PM₂.₅ Forecast")
    
    # Generate Forecast Button
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            # Prepare request
            request_data = {
                "station_ids": [config["station_id"]],
                "horizon_hours": config["horizon_hours"],
                "model_type": config["model_type"],
                "uncertainty_level": config["uncertainty_level"],
                "language": "en",  # Default to English
                "include_explanations": config["include_explanations"]
            }
            
            # Generate forecast
            forecast_response = ui.generate_forecast(request_data)
            
            if forecast_response:
                st.success("✅ Forecast generated successfully!")
                
                # Display forecast data
                station_forecasts = forecast_response["station_forecasts"]
                if config["station_id"] in station_forecasts:
                    forecasts = station_forecasts[config["station_id"]]
                    
                    # Create forecast DataFrame
                    forecast_df = pd.DataFrame([
                        {
                            "timestamp": pd.to_datetime(f["timestamp"]),
                            "pm25_mean": f["pm25_mean"],
                            "pm25_lower": f.get("pm25_lower"),
                            "pm25_upper": f.get("pm25_upper"),
                            "confidence_level": f.get("confidence_level", config["uncertainty_level"])
                        }
                        for f in forecasts
                    ])
                    
                    # Get current air quality info
                    current_pm25 = forecast_df['pm25_mean'].iloc[0]
                    air_quality_info = get_air_quality_info(current_pm25)
                    prevention_tips = get_prevention_tips(current_pm25)
                    
                    # Display current air quality status
                    st.markdown(f"""
                    <div class="{air_quality_info['css_class']}">
                        <h3>{air_quality_info['icon']} Current Air Quality: {air_quality_info['level']}</h3>
                        <p><strong>PM₂.₅ Level:</strong> {current_pm25:.1f} μg/m³</p>
                        <p>{air_quality_info['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display prevention tips
                    st.subheader("🛡️ Health Recommendations")
                    for tip in prevention_tips:
                        st.markdown(f'<div class="prevention-tip">{tip}</div>', unsafe_allow_html=True)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Current PM₂.₅",
                            f"{forecast_df['pm25_mean'].iloc[0]:.1f} μg/m³",
                            delta=None
                        )
                    
                    # Compact feature-importance panel
                    try:
                        with st.expander("🔍 Top drivers (last 60 days)", expanded=False):
                            explain_req = {
                                "station_id": str(config["station_id"]),
                                "methods": ["permutation"],
                                "language": "en"
                            }
                            exp_resp = ui.post("/explainability", explain_req)
                            if exp_resp and exp_resp.get("feature_importance"):
                                impor = exp_resp["feature_importance"]
                                # Take top 6 by score
                                impor_sorted = sorted(impor, key=lambda x: x.get("importance_score", 0), reverse=True)[:6]
                                names = [i.get("feature_name", "?") for i in impor_sorted]
                                scores = [float(i.get("importance_score", 0)) for i in impor_sorted]
                                fi_df = pd.DataFrame({"feature": names, "score": scores})
                                st.bar_chart(fi_df.set_index("feature"))
                            else:
                                st.caption("No explainability available right now.")
                    except Exception:
                        st.caption("Explainability unavailable.")

                    with col2:
                        st.metric(
                            "Peak PM₂.₅",
                            f"{forecast_df['pm25_mean'].max():.1f} μg/m³",
                            delta=None
                        )
                    
                    with col3:
                        st.metric(
                            "Average PM₂.₅",
                            f"{forecast_df['pm25_mean'].mean():.1f} μg/m³",
                            delta=None
                        )
                    
                    with col4:
                        st.metric(
                            "Processing Time",
                            f"{forecast_response['processing_time_ms']:.0f} ms",
                            delta=None
                        )
                    
                    # Create forecast plot
                    fig = go.Figure()
                    
                    # Add mean forecast line
                    fig.add_trace(go.Scatter(
                        x=forecast_df['timestamp'],
                        y=forecast_df['pm25_mean'],
                        mode='lines+markers',
                        name='PM₂.₅ Forecast',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=6)
                    ))
                    
                    # Add uncertainty bands if available
                    if forecast_df['pm25_lower'].notna().any() and forecast_df['pm25_upper'].notna().any():
                        fig.add_trace(go.Scatter(
                            x=forecast_df['timestamp'],
                            y=forecast_df['pm25_upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_df['timestamp'],
                            y=forecast_df['pm25_lower'],
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(31, 119, 180, 0.2)',
                            name=f'{config["uncertainty_level"]*100:.0f}% Confidence',
                            hoverinfo='skip'
                        ))
                    
                    # Add air quality thresholds with better labels
                    fig.add_hline(y=12, line_dash="dash", line_color="green", 
                                 annotation_text="Good (12 μg/m³)")
                    fig.add_hline(y=35, line_dash="dash", line_color="yellow", 
                                 annotation_text="Moderate (35 μg/m³)")
                    fig.add_hline(y=55, line_dash="dash", line_color="orange", 
                                 annotation_text="Unhealthy for Sensitive Groups (55 μg/m³)")
                    fig.add_hline(y=150, line_dash="dash", line_color="red", 
                                 annotation_text="Unhealthy (150 μg/m³)")
                    
                    # Update layout
                    fig.update_layout(
                        title=f"PM₂.₅ Forecast - {config['horizon_hours']} Hours",
                        xaxis_title="Time",
                        yaxis_title="PM₂.₅ (μg/m³)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Air Quality Index Explanation
                    st.subheader("📊 Understanding Air Quality Levels")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **Air Quality Index (AQI) Categories:**
                        - 🟢 **Good (0-12 μg/m³)**: Air quality is satisfactory
                        - 🟡 **Moderate (12-35 μg/m³)**: Acceptable for most people
                        - 🟠 **Unhealthy for Sensitive Groups (35-55 μg/m³)**: Sensitive individuals may experience health effects
                        - 🔴 **Unhealthy (55-150 μg/m³)**: Everyone may experience health effects
                        - 🟣 **Hazardous (150+ μg/m³)**: Emergency conditions, entire population affected
                        """)
                    
                    with col2:
                        st.markdown("""
                        **What is PM₂.₅?**
                        - PM₂.₅ refers to fine particulate matter with diameter ≤ 2.5 micrometers
                        - These particles can penetrate deep into the lungs and bloodstream
                        - Sources include vehicle emissions, industrial processes, and natural sources
                        - Long-term exposure can cause respiratory and cardiovascular diseases
                        """)
                    
                    # Display forecast table
                    st.subheader("📋 Forecast Details")
                    st.dataframe(
                        forecast_df.round(2),
                        width='stretch'
                    )
                    
                    # Model information
                    st.subheader("🤖 Model Information")
                    model_info = forecast_response["model_info"]
                    
                    # Badges for model used and bias correction
                    model_used = str(model_info.get("model_type", "unknown")).lower()
                    bias_corrected = "Yes" if model_used in ["patchtst", "ensemble"] else "No"
                    
                    badge_css = (
                        "<span style='display:inline-block;padding:6px 10px;border-radius:12px;"
                        "background:#eef2ff;color:#3b82f6;margin-right:8px;font-weight:600'>"
                        f"Model used: {model_used.upper()}"
                        "</span>"
                        "<span style='display:inline-block;padding:6px 10px;border-radius:12px;"
                        f"background:{'#dcfce7' if bias_corrected=='Yes' else '#fee2e2'};"
                        f"color:{'#16a34a' if bias_corrected=='Yes' else '#dc2626'};font-weight:600'>"
                        f"Bias corrected: {bias_corrected}"
                        "</span>"
                    )
                    st.markdown(badge_css, unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.json({
                            "Model Type": model_info["model_type"],
                            "Horizon": f"{model_info['horizon_hours']} hours",
                            "Uncertainty Level": f"{model_info['uncertainty_level']*100:.0f}%"
                        })
                    
                    with col2:
                        st.json({
                            "Stations": model_info["stations_count"],
                            "Request ID": forecast_response["request_id"],
                            "Timestamp": forecast_response["timestamp"]
                        })
                
                else:
                    st.error("No forecast data received for the selected station")
            else:
                st.error("❌ Failed to generate forecast. Please check the API connection.")


def render_explainability_tab(ui: AirAwareUI, config: Dict[str, Any]):
    """Render the explainability tab"""
    st.header("🔍 Feature Importance Analysis")
    
    # Method selection
    st.subheader("📊 Analysis Methods")
    col1, col2 = st.columns(2)
    
    with col1:
        use_permutation = st.checkbox("Permutation Importance", value=True)
        use_tree = st.checkbox("Tree-based Importance", value=True)
    
    with col2:
        use_correlation = st.checkbox("Correlation Analysis", value=False)
        use_mutual_info = st.checkbox("Mutual Information", value=False)
    
    methods = []
    if use_permutation:
        methods.append("permutation")
    if use_tree:
        methods.append("tree")
    if use_correlation:
        methods.append("correlation")
    if use_mutual_info:
        methods.append("mutual_information")
    
    if not methods:
        st.warning("Please select at least one analysis method.")
        return
    
    # Analyze Button
    if st.button("🔍 Analyze Feature Importance", type="primary"):
        with st.spinner("Analyzing feature importance..."):
            # Prepare request
            request_data = {
                "station_id": config["station_id"],
                "horizon_hours": config["horizon_hours"],
                "model_type": config["model_type"],
                "methods": methods,
                "language": config["language"]
            }
            
            # Analyze explainability
            explainability_response = ui.analyze_explainability(request_data)
            
            if explainability_response:
                st.success("✅ Feature importance analysis completed!")
                
                # Display feature importance
                feature_importance = explainability_response["feature_importance"]
                
                if feature_importance:
                    # Create DataFrame
                    importance_df = pd.DataFrame([
                        {
                            "Feature": f["feature_name"],
                            "Importance": f["importance_score"],
                            "Method": f["method"]
                        }
                        for f in feature_importance
                    ])
                    
                    # Group by method and create plots
                    methods_used = importance_df["Method"].unique()
                    
                    for method in methods_used:
                        method_df = importance_df[importance_df["Method"] == method].copy()
                        method_df = method_df.sort_values("Importance", ascending=True)
                        
                        # Create horizontal bar chart
                        fig = px.bar(
                            method_df,
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            title=f"Feature Importance - {method.title()} Method",
                            color="Importance",
                            color_continuous_scale="Blues"
                        )
                        
                        fig.update_layout(
                            height=max(400, len(method_df) * 30),
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                    
                    # Display summary table
                    st.subheader("📋 Feature Importance Summary")
                    st.dataframe(
                        importance_df.pivot_table(
                            index="Feature",
                            columns="Method",
                            values="Importance",
                            aggfunc="first"
                        ).round(4),
                        width='stretch'
                    )
                    
                    # Processing information
                    st.subheader("ℹ️ Analysis Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.json({
                            "Station ID": explainability_response["station_id"],
                            "Methods Used": methods,
                            "Features Analyzed": len(importance_df["Feature"].unique())
                        })
                    
                    with col2:
                        st.json({
                            "Processing Time": f"{explainability_response['processing_time_ms']:.0f} ms",
                            "Request ID": explainability_response["request_id"],
                            "Timestamp": explainability_response["timestamp"]
                        })
                
                else:
                    st.warning("No feature importance data received.")
            else:
                st.error("❌ Failed to analyze feature importance. Please check the API connection.")


def render_whatif_tab(ui: AirAwareUI, config: Dict[str, Any]):
    """Render the what-if analysis tab"""
    st.header("🎯 What-If Scenario Analysis")
    
    st.markdown("""
    <div class="info-box">
        <strong>What-If Analysis:</strong> Explore how different meteorological conditions 
        might affect PM₂.₅ levels. Create custom scenarios to understand the impact of 
        various factors on air quality.
    </div>
    """, unsafe_allow_html=True)
    
    # Scenario creation
    st.subheader("📝 Create Scenarios")
    
    # Default scenarios
    default_scenarios = [
        {
            "name": "High Wind Speed",
            "description": "Wind speed increased by 50%",
            "changes": {"wind_speed": 1.5}
        },
        {
            "name": "Low Temperature",
            "description": "Temperature decreased by 5°C",
            "changes": {"t2m_celsius": -5}
        },
        {
            "name": "High Humidity",
            "description": "Humidity increased by 30%",
            "changes": {"humidity": 1.3}
        }
    ]
    
    # Scenario selection
    scenario_option = st.radio(
        "Choose scenario type:",
        ["Use default scenarios", "Create custom scenario"]
    )
    
    if scenario_option == "Use default scenarios":
        selected_scenarios = default_scenarios
        st.info("Using predefined scenarios for demonstration.")
    else:
        # Custom scenario creation
        st.subheader("🔧 Custom Scenario")
        
        scenario_name = st.text_input("Scenario Name:", value="Custom Scenario")
        scenario_description = st.text_area("Description:", value="Custom meteorological scenario")
        
        # Parameter changes
        st.subheader("📊 Parameter Changes")
        col1, col2 = st.columns(2)
        
        with col1:
            wind_multiplier = st.slider("Wind Speed Multiplier", 0.5, 2.0, 1.0, 0.1)
            temp_change = st.slider("Temperature Change (°C)", -10.0, 10.0, 0.0, 0.5)
        
        with col2:
            humidity_multiplier = st.slider("Humidity Multiplier", 0.5, 2.0, 1.0, 0.1)
            pressure_change = st.slider("Pressure Change (hPa)", -20.0, 20.0, 0.0, 1.0)
        
        # Create custom scenario
        changes = {}
        if wind_multiplier != 1.0:
            changes["wind_speed"] = wind_multiplier
        if temp_change != 0.0:
            changes["t2m_celsius"] = temp_change
        if humidity_multiplier != 1.0:
            changes["humidity"] = humidity_multiplier
        if pressure_change != 0.0:
            changes["pressure"] = pressure_change
        
        selected_scenarios = [{
            "name": scenario_name,
            "description": scenario_description,
            "changes": changes
        }]
    
    # Analyze Button
    if st.button("🎯 Analyze What-If Scenarios", type="primary"):
        with st.spinner("Analyzing what-if scenarios..."):
            # Prepare request
            request_data = {
                "station_id": config["station_id"],
                "horizon_hours": config["horizon_hours"],
                "model_type": config["model_type"],
                "scenarios": selected_scenarios,
                "language": config["language"]
            }
            
            # Analyze what-if
            whatif_response = ui.analyze_what_if(request_data)
            
            if whatif_response:
                st.success("✅ What-if analysis completed!")
                
                # Display scenarios
                scenarios = whatif_response["scenarios"]
                
                for i, scenario in enumerate(scenarios):
                    st.subheader(f"📊 Scenario {i+1}: {scenario['scenario_name']}")
                    
                    # Scenario description
                    st.markdown(f"**Description:** {scenario['scenario_description']}")
                    
                    # Impact analysis
                    impact_analysis = scenario["impact_analysis"]
                    
                    if impact_analysis:
                        # Create impact metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Baseline PM₂.₅",
                                f"{impact_analysis.get('baseline_mean', 0):.1f} μg/m³"
                            )
                        
                        with col2:
                            st.metric(
                                "Scenario PM₂.₅",
                                f"{impact_analysis.get('scenario_mean', 0):.1f} μg/m³"
                            )
                        
                        with col3:
                            change = impact_analysis.get('scenario_mean', 0) - impact_analysis.get('baseline_mean', 0)
                            st.metric(
                                "Change",
                                f"{change:+.1f} μg/m³",
                                delta=f"{change:+.1f}"
                            )
                        
                        with col4:
                            percent_change = (change / impact_analysis.get('baseline_mean', 1)) * 100
                            st.metric(
                                "Percent Change",
                                f"{percent_change:+.1f}%",
                                delta=f"{percent_change:+.1f}%"
                            )
                    
                    # Display detailed results if available
                    if "detailed_results" in impact_analysis:
                        st.json(impact_analysis["detailed_results"])
                    
                    st.divider()
                
                # Analysis Summary
                st.subheader("📋 Analysis Summary")
                
                # Key insights
                st.markdown("### 🔍 Key Insights")
                
                # Calculate overall impact
                total_scenarios = len(scenarios)
                avg_change = 0
                if scenarios:
                    changes = []
                    for scenario in scenarios:
                        impact = scenario.get("impact_analysis", {})
                        baseline = impact.get('baseline_mean', 0)
                        scenario_val = impact.get('scenario_mean', 0)
                        if baseline > 0:
                            change = scenario_val - baseline
                            changes.append(change)
                    
                    if changes:
                        avg_change = sum(changes) / len(changes)
                
                # Display insights
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Scenarios Analyzed",
                        total_scenarios,
                        help="Number of what-if scenarios evaluated"
                    )
                
                with col2:
                    st.metric(
                        "Average PM₂.₅ Change",
                        f"{avg_change:+.1f} μg/m³",
                        delta=f"{avg_change:+.1f}",
                        help="Average change in PM₂.₅ levels across all scenarios"
                    )
                
                with col3:
                    processing_time = whatif_response['processing_time_ms']
                    st.metric(
                        "Analysis Speed",
                        f"{processing_time:.0f} ms",
                        help="Time taken to complete the analysis"
                    )
                
                # Health impact assessment
                if avg_change != 0:
                    st.markdown("### 🏥 Health Impact Assessment")
                    
                    if avg_change > 5:
                        st.warning("⚠️ **Significant Increase**: This scenario would lead to substantially higher air pollution levels, potentially affecting sensitive groups.")
                    elif avg_change > 2:
                        st.info("ℹ️ **Moderate Increase**: Air quality would worsen, with some impact on air quality-sensitive individuals.")
                    elif avg_change < -5:
                        st.success("✅ **Significant Improvement**: This scenario would substantially improve air quality and benefit public health.")
                    elif avg_change < -2:
                        st.success("✅ **Moderate Improvement**: Air quality would improve, benefiting most people.")
                    else:
                        st.info("ℹ️ **Minimal Impact**: This scenario would have little effect on air quality levels.")
                
                # Technical details (collapsible)
                with st.expander("🔧 Technical Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Analysis Configuration:**")
                        st.write(f"• **Station**: {whatif_response['station_id']}")
                        st.write(f"• **Model**: {config['model_type'].title()}")
                        st.write(f"• **Forecast Horizon**: {config['horizon_hours']} hours")
                    
                    with col2:
                        st.markdown("**Processing Information:**")
                        st.write(f"• **Processing Time**: {processing_time:.0f} ms")
                        st.write(f"• **Request ID**: {whatif_response['request_id'][:8]}...")
                        st.write(f"• **Timestamp**: {whatif_response['timestamp']}")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                if avg_change > 0:
                    st.markdown("""
                    **To mitigate the negative impact:**
                    - Consider implementing pollution control measures
                    - Monitor air quality more frequently during these conditions
                    - Advise sensitive groups to limit outdoor activities
                    - Explore alternative scenarios with lower pollution impact
                    """)
                elif avg_change < 0:
                    st.markdown("""
                    **To maximize the positive impact:**
                    - This scenario shows promising air quality improvements
                    - Consider implementing similar conditions more frequently
                    - Monitor the effectiveness of these changes
                    - Share insights with relevant stakeholders
                    """)
                else:
                    st.markdown("""
                    **To gain more insights:**
                    - Try more extreme parameter changes
                    - Analyze different time periods
                    - Compare with other meteorological scenarios
                    - Consider seasonal variations
                    """)
            
            else:
                st.error("❌ Failed to analyze what-if scenarios. Please check the API connection.")


def render_agents_tab(ui: AirAwareUI, config: Dict[str, Any]):
    """Render the intelligent agents tab"""
    st.header("🤖 Intelligent Agents")
    
    # Agent selection
    st.subheader("🎯 Select Agent")
    agent_type = st.selectbox(
        "Choose an agent:",
        ["health", "forecast", "data", "notification", "orchestrator"],
        format_func=lambda x: {
            "health": "🏥 Health Advisory Agent",
            "forecast": "🔮 Forecast Optimization Agent", 
            "data": "📊 Data Quality Monitoring Agent",
            "notification": "📢 Notification Agent",
            "orchestrator": "🎭 Agent Orchestrator"
        }[x]
    )
    
    # User ID input
    user_id = st.text_input("User ID (optional):", value="demo_user")
    
    # Agent execution
    if st.button(f"🚀 Execute {agent_type.title()} Agent", type="primary"):
        with st.spinner(f"Executing {agent_type} agent..."):
            # Prepare request
            request_data = {
                "agent_type": agent_type,
                "station_id": config["station_id"],
                "user_id": user_id,
                "context": {
                    "pm25_data": {
                        "current": 45.0,  # Demo data
                        "forecast": [40.0, 42.0, 38.0, 35.0]
                    },
                    "forecast_horizon": config["horizon_hours"],
                    "location": "kathmandu"
                },
                "language": "en"
            }
            
            # Execute agent
            agent_response = ui.execute_agent(request_data)
            
            if agent_response:
                st.success(f"✅ {agent_type.title()} agent executed successfully!")
                
                # Display results
                result = agent_response.get("result", {})
                
                if agent_type == "health":
                    st.subheader("🏥 Health Recommendations")
                    health_summary = result.get("health_summary", {})
                    recommendations = result.get("recommendations", [])
                    emergency_alerts = result.get("emergency_alerts", [])
                    
                    # Health summary
                    if health_summary:
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Current Risk Level:</strong> {health_summary.get('current_risk_level', 'Unknown').title()}<br>
                            <strong>Risk Description:</strong> {health_summary.get('risk_description', 'No description available')}<br>
                            <strong>Current PM₂.₅:</strong> {health_summary.get('current_pm25', 0):.1f} μg/m³
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Recommendations
                    if recommendations:
                        st.subheader("💡 Health Recommendations")
                        for rec in recommendations[:5]:  # Show top 5
                            priority_color = {
                                "critical": "🔴",
                                "high": "🟠", 
                                "medium": "🟡",
                                "low": "🟢"
                            }.get(rec.get("priority", "medium"), "🟡")
                            
                            st.markdown(f"""
                            <div class="prevention-tip">
                                {priority_color} <strong>{rec.get('title', 'Recommendation')}</strong><br>
                                {rec.get('description', 'No description')}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Emergency alerts
                    if emergency_alerts:
                        st.subheader("🚨 Emergency Alerts")
                        for alert in emergency_alerts:
                            st.error(f"**{alert.get('title', 'Alert')}**: {alert.get('message', 'No message')}")
                
                elif agent_type == "forecast":
                    st.subheader("🔮 Forecast Optimization")
                    model_recommendations = result.get("model_recommendations", {})
                    performance_insights = result.get("performance_insights", {})
                    
                    # Model recommendations
                    if model_recommendations:
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Recommended Model:</strong> {model_recommendations.get('recommended_model', 'Unknown')}<br>
                            <strong>Confidence:</strong> {model_recommendations.get('confidence', 0):.1%}<br>
                            <strong>Reasoning:</strong> {model_recommendations.get('reasoning', 'No reasoning provided')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Performance insights
                    if performance_insights:
                        st.subheader("📊 Performance Insights")
                        insights = performance_insights.get("insights", [])
                        for insight in insights[:3]:  # Show top 3
                            st.markdown(f"""
                            **{insight.get('model_id', 'Model')}**: 
                            MAE: {insight.get('performance', {}).get('mae', 0):.2f} μg/m³, 
                            Samples: {insight.get('samples', 0)}
                            """)
                
                elif agent_type == "data":
                    st.subheader("📊 Data Quality Assessment")
                    quality_report = result.get("quality_report", {})
                    
                    if quality_report:
                        overall_score = quality_report.get("overall_quality_score", 0)
                        status = quality_report.get("status", "unknown")
                        
                        # Overall quality
                        status_color = {
                            "good": "🟢",
                            "warning": "🟡", 
                            "critical": "🔴"
                        }.get(status, "⚪")
                        
                        st.markdown(f"""
                        <div class="info-box">
                            {status_color} <strong>Overall Quality Score:</strong> {overall_score:.1%}<br>
                            <strong>Status:</strong> {status.title()}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Quality metrics
                        metrics = quality_report.get("quality_metrics", [])
                        for metric in metrics:
                            metric_status = metric.get("status", "unknown")
                            metric_color = {
                                "good": "🟢",
                                "warning": "🟡",
                                "critical": "🔴"
                            }.get(metric_status, "⚪")
                            
                            st.markdown(f"""
                            {metric_color} **{metric.get('metric_name', 'Metric').title()}**: 
                            {metric.get('value', 0):.1%} (Threshold: {metric.get('threshold', 0):.1%})
                            """)
                
                elif agent_type == "notification":
                    st.subheader("📢 Notification Results")
                    alerts_generated = result.get("alerts_generated", 0)
                    notifications_sent = result.get("notifications_sent", 0)
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>Alerts Generated:</strong> {alerts_generated}<br>
                        <strong>Notifications Sent:</strong> {notifications_sent}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show alerts
                    alerts = result.get("alerts", [])
                    if alerts:
                        st.subheader("🚨 Generated Alerts")
                        for alert in alerts[:3]:  # Show top 3
                            severity_color = {
                                "critical": "🔴",
                                "high": "🟠",
                                "medium": "🟡", 
                                "low": "🟢"
                            }.get(alert.get("severity", "medium"), "🟡")
                            
                            st.markdown(f"""
                            <div class="prevention-tip">
                                {severity_color} <strong>{alert.get('title', 'Alert')}</strong><br>
                                {alert.get('message', 'No message')}
                            </div>
                            """, unsafe_allow_html=True)
                
                elif agent_type == "orchestrator":
                    st.subheader("🎭 Workflow Orchestration")
                    workflow_result = result.get("workflow_result", {})
                    system_insights = result.get("system_insights", [])
                    system_health = result.get("system_health", {})
                    
                    # Workflow result
                    if workflow_result:
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Workflow Status:</strong> {workflow_result.get('status', 'Unknown')}<br>
                            <strong>Completed Steps:</strong> {workflow_result.get('completed_steps', 0)}/{workflow_result.get('total_steps', 0)}<br>
                            <strong>Duration:</strong> {workflow_result.get('total_duration', 0):.2f} seconds
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # System insights
                    if system_insights:
                        st.subheader("💡 System Insights")
                        for insight in system_insights[:3]:  # Show top 3
                            severity_color = {
                                "critical": "🔴",
                                "high": "🟠",
                                "medium": "🟡",
                                "low": "🟢"
                            }.get(insight.get("severity", "medium"), "🟡")
                            
                            st.markdown(f"""
                            <div class="prevention-tip">
                                {severity_color} <strong>{insight.get('title', 'Insight')}</strong><br>
                                {insight.get('description', 'No description')}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Processing time
                processing_time = agent_response.get("processing_time_ms", 0)
                st.info(f"⏱️ Processing time: {processing_time:.0f} ms")
                
            else:
                st.error("❌ Failed to execute agent. Please check the API connection.")
    
    # System status
    st.subheader("📊 System Status")
    if st.button("🔄 Refresh System Status"):
        with st.spinner("Getting system status..."):
            system_status = ui.get_system_status()
            
            if system_status:
                st.success("✅ System status retrieved successfully!")
                
                # Overall status
                overall_status = system_status.get("status", "unknown")
                system_health = system_status.get("system_health", "unknown")
                
                status_color = {
                    "healthy": "🟢",
                    "degraded": "🟡",
                    "unhealthy": "🔴"
                }.get(overall_status, "⚪")
                
                st.markdown(f"""
                <div class="info-box">
                    {status_color} <strong>Overall Status:</strong> {overall_status.title()}<br>
                    <strong>System Health:</strong> {system_health.title()}<br>
                    <strong>Uptime:</strong> {system_status.get('uptime_seconds', 0):.0f} seconds
                </div>
                """, unsafe_allow_html=True)
                
                # Agent status
                agents = system_status.get("agents", {})
                if agents:
                    st.subheader("🤖 Agent Status")
                    for agent_id, agent_info in agents.items():
                        agent_status = agent_info.get("status", "unknown")
                        uptime = agent_info.get("uptime_percentage", 0)
                        
                        status_color = {
                            "running": "🟢",
                            "idle": "🟡",
                            "error": "🔴",
                            "stopped": "⚫"
                        }.get(agent_status, "⚪")
                        
                        st.markdown(f"""
                        **{agent_info.get('agent_name', agent_id)}**: 
                        {status_color} {agent_status.title()} 
                        (Uptime: {uptime:.1f}%)
                        """)
                
                # Workflows
                workflows = system_status.get("workflows", {})
                if workflows:
                    st.subheader("🔄 Workflow Status")
                    st.markdown(f"""
                    **Active Workflows:** {workflows.get('active', 0)}<br>
                    **Total History:** {workflows.get('total_history', 0)}<br>
                    **Recent Success Rate:** {workflows.get('recent_success_rate', 0):.1%}
                    """)
                
                # Insights
                insights = system_status.get("insights", {})
                if insights:
                    st.subheader("💡 System Insights")
                    st.markdown(f"""
                    **Total Insights:** {insights.get('total', 0)}<br>
                    **Unresolved:** {insights.get('unresolved', 0)}<br>
                    **Critical:** {insights.get('critical', 0)}
                    """)
                
            else:
                st.error("❌ Failed to get system status. Please check the API connection.")


def render_cross_station_tab(ui: AirAwareUI, config: Dict[str, Any]):
    """Render the Cross-Station Analysis tab."""
    
    st.header("🌍 Cross-Station Air Quality Analysis")
    
    st.markdown("""
    Analyze air quality patterns using data from external monitoring stations 
    across India, China, and the United States to improve local predictions.
    """)
    
    # External stations info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Available External Stations")
        
        # Get external stations info
        try:
            response = requests.get(f"{ui.api_base_url}/cross-station/external-stations")
            if response.status_code == 200:
                stations_data = response.json()
                
                stations_dict = stations_data.get("stations", {})
                for country, stations in stations_dict.items():
                    st.markdown(f"**{country.title()}** ({len(stations)} stations)")
                    for station in stations[:3]:  # Show first 3 stations
                        st.markdown(f"• {station['name']} ({station['id']})")
                    if len(stations) > 3:
                        st.markdown(f"• ... and {len(stations) - 3} more")
                    st.markdown("")
            else:
                st.error("Failed to load external stations information")
        except Exception as e:
            st.error(f"Error loading external stations: {e}")
    
    with col2:
        st.subheader("🔬 Spatial Features")
        
        try:
            response = requests.get(f"{ui.api_base_url}/cross-station/spatial-features")
            if response.status_code == 200:
                features_data = response.json()
                
                feature_categories = features_data.get("feature_categories", {})
                st.markdown("**Feature Categories:**")
                for category, features in feature_categories.items():
                    st.markdown(f"• **{category.replace('_', ' ').title()}**: {len(features)} features")
                
                st.markdown("**Sample Features:**")
                sample_features = []
                for features in feature_categories.values():
                    sample_features.extend(features[:2])
                
                for feature in sample_features[:6]:
                    st.markdown(f"• {feature}")
            else:
                st.error("Failed to load spatial features information")
        except Exception as e:
            st.error(f"Error loading spatial features: {e}")
    
    # Cross-station forecast
    st.markdown("---")
    st.subheader("🚀 Generate Cross-Station Forecast")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        station_id = st.selectbox(
            "Select Station",
            options=config.get("available_stations", []),
            format_func=lambda x: f"{x} - {config.get('station_names', {}).get(x, 'Unknown Location')}"
        )
    
    with col2:
        horizon_hours = st.selectbox(
            "Forecast Horizon",
            options=[6, 12, 24, 48],
            index=2
        )
    
    # External stations selection
    external_stations = st.multiselect(
        "Select External Countries",
        options=["india", "china", "usa"],
        default=["india", "china"],
        help="Select which countries' data to include in the analysis"
    )

    # Consistency option: allow using the same Prophet model as Forecast tab for comparison
    use_prophet_comparison = st.checkbox(
        "Use station-only Prophet for comparison (enforce consistency)",
        value=True,
        help="Fetch Prophet forecast for the same station and align timestamps for an apples-to-apples comparison."
    )
    
    if st.button("Generate Cross-Station Forecast", type="primary"):
        if not external_stations:
            st.warning("Please select at least one external country.")
            return
        
        with st.spinner("Generating cross-station forecast..."):
            try:
                # Make API request
                response = requests.post(
                    f"{ui.api_base_url}/cross-station/forecast",
                    params={
                        "station_id": station_id,
                        "horizon_hours": horizon_hours,
                        "include_external_stations": external_stations
                    }
                )
                
                if response.status_code == 200:
                    forecast_data = response.json()
                    
                    # Display forecast results
                    st.success("Cross-station forecast generated successfully!")
                    
                    # Show model info with badges
                    model_info = forecast_data.get("model_info", {})
                    model_used = str(model_info.get("model_type", "unknown")).lower()
                    bias_corrected = "Yes" if model_used in ["ensemble", "patchtst"] else "No"
                    badge_css = (
                        "<span style='display:inline-block;padding:6px 10px;border-radius:12px;"
                        "background:#eef2ff;color:#3b82f6;margin-right:8px;font-weight:600'>"
                        f"Model used: {model_used.upper()}"
                        "</span>"
                        "<span style='display:inline-block;padding:6px 10px;border-radius:12px;"
                        f"background:{'#dcfce7' if bias_corrected=='Yes' else '#fee2e2'};"
                        f"color:{'#16a34a' if bias_corrected=='Yes' else '#dc2626'};font-weight:600'>"
                        f"Bias corrected: {bias_corrected}"
                        "</span>"
                    )
                    st.markdown(badge_css, unsafe_allow_html=True)
                    if model_info.get('external_stations_used'):
                        st.info(f"External stations: {', '.join(model_info.get('external_stations_used', []))}")
                    
                    # Plot forecast
                    if "forecasts" in forecast_data:
                        forecasts = forecast_data["forecasts"]
                        
                        # Create DataFrame for plotting
                        df_forecast = pd.DataFrame(forecasts)
                        df_forecast["timestamp"] = pd.to_datetime(df_forecast["timestamp"])
                        
                        # Plot
                        fig = go.Figure()
                        
                        # Add PM2.5 forecast
                        fig.add_trace(go.Scatter(
                            x=df_forecast["timestamp"],
                            y=df_forecast["pm25_mean"],
                            mode="lines+markers",
                            name="PM₂.₅ Forecast",
                            line=dict(color="#1f77b4", width=3),
                            marker=dict(size=6)
                        ))
                        
                        # Add confidence intervals
                        if "pm25_lower" in df_forecast.columns and "pm25_upper" in df_forecast.columns:
                            fig.add_trace(go.Scatter(
                                x=df_forecast["timestamp"],
                                y=df_forecast["pm25_upper"],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip"
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=df_forecast["timestamp"],
                                y=df_forecast["pm25_lower"],
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor="rgba(31, 119, 180, 0.2)",
                                name="Confidence Interval",
                                hoverinfo="skip"
                            ))
                        
                        # Update layout
                        fig.update_layout(
                            title="Cross-Station PM₂.₅ Forecast",
                            xaxis_title="Time",
                            yaxis_title="PM₂.₅ (µg/m³)",
                            hovermode="x unified",
                            height=500
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                        
                        # Add current value and key metrics display
                        if len(df_forecast) > 0:
                            st.markdown("### 📊 Current Values & Summary")
                            
                            # Get current PM2.5 (first forecast value)
                            current_pm25 = df_forecast["pm25_mean"].iloc[0]
                            max_pm25 = df_forecast["pm25_mean"].max()
                            min_pm25 = df_forecast["pm25_mean"].min()
                            avg_pm25 = df_forecast["pm25_mean"].mean()
                            
                            # Air quality classification
                            air_quality_info = get_air_quality_info(current_pm25)
                            
                            # Display metrics in columns
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Current PM₂.₅",
                                    f"{current_pm25:.1f} μg/m³",
                                    help=f"Air Quality: {air_quality_info['level']}"
                                )
                            
                            with col2:
                                st.metric(
                                    "Peak Forecast",
                                    f"{max_pm25:.1f} μg/m³",
                                    delta=f"{max_pm25 - current_pm25:+.1f}",
                                    help="Highest predicted value"
                                )
                            
                            with col3:
                                st.metric(
                                    "Average Forecast",
                                    f"{avg_pm25:.1f} μg/m³",
                                    delta=f"{avg_pm25 - current_pm25:+.1f}",
                                    help="Average over forecast period"
                                )
                            
                            with col4:
                                st.metric(
                                    "Forecast Range",
                                    f"│{max_pm25 - min_pm25:.1f}│ μg/m³",
                                    help="Volatility in predictions"
                                )
                            
                            # Display current air quality status
                            st.markdown(f"""
                            <div class="{air_quality_info['css_class']}">
                                <h4>{air_quality_info['icon']} Current Air Quality: {air_quality_info['level']}</h4>
                                <p><strong>Current PM₂.₅:</strong> {current_pm25:.1f} μg/m³</p>
                                <p><strong>Description:</strong> {air_quality_info['description']}</p>
                                <p><strong>Health Impact:</strong> {air_quality_info.get('health_impact', 'Monitor air quality for sensitive groups.')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add prevention tips if PM2.5 is moderate or worse
                            if current_pm25 > 35:  # WHO guideline threshold
                                prevention_tips = get_prevention_tips(current_pm25)
                                st.subheader("🛡️ Health Recommendations")
                                for tip in prevention_tips:
                                    st.markdown(f'<div class="prevention moderation">• {tip}</div>', unsafe_allow_html=True)

                        # Consistency and precision: compare with Prophet and latest observed
                        if use_prophet_comparison and len(df_forecast) > 0:
                            try:
                                comp_resp = requests.post(
                                    f"{ui.api_base_url}/forecast",
                                    json={
                                        "station_ids": [station_id],
                                        "horizon_hours": int(horizon_hours),
                                        "model_type": "prophet",
                                        "include_uncertainty": True
                                    },
                                    timeout=30
                                )
                                if comp_resp.status_code == 200:
                                    comp_json = comp_resp.json()
                                    station_series = comp_json.get("station_forecasts", {}).get(station_id, [])
                                    if station_series:
                                        df_prophet = pd.DataFrame(station_series)
                                        df_prophet["timestamp"] = pd.to_datetime(df_prophet["timestamp"]).dt.tz_convert("UTC")
                                        # Align on first cross-station timestamp
                                        first_ts = df_forecast["timestamp"].iloc[0]
                                        match = df_prophet[df_prophet["timestamp"] == first_ts]
                                        if match.empty:
                                            # fallback to the nearest future timestamp
                                            match = df_prophet.iloc[[0]]
                                        prophet_current = float(match["pm25_mean"].iloc[0])

                                        # Latest observed from cross-station (if provided)
                                        latest_obs = forecast_data.get("latest_observed")
                                        latest_val = None
                                        if latest_obs and isinstance(latest_obs, dict) and latest_obs.get("pm25") is not None:
                                            try:
                                                latest_val = float(latest_obs.get("pm25"))
                                            except Exception:
                                                latest_val = None

                                        # Decide which is closer to latest observed
                                        chosen_model = "cross_station"
                                        if latest_val is not None:
                                            cross_val = float(df_forecast["pm25_mean"].iloc[0])
                                            cross_err = abs(cross_val - latest_val)
                                            prophet_err = abs(prophet_current - latest_val)
                                            chosen_model = "prophet" if prophet_err <= cross_err else "cross_station"

                                        st.markdown("### 🔍 Consistency Check (Same timestamp)")
                                        c1, c2, c3 = st.columns(3)
                                        with c1:
                                            st.metric("Cross-station (current)", f"{df_forecast['pm25_mean'].iloc[0]:.1f} μg/m³")
                                        with c2:
                                            st.metric("Prophet (station-only)", f"{prophet_current:.1f} μg/m³")
                                        with c3:
                                            if latest_val is not None:
                                                st.metric("Latest observed", f"{latest_val:.1f} μg/m³")
                                            else:
                                                st.metric("Latest observed", "n/a")

                                        note = "Prophet" if chosen_model == "prophet" else "Cross‑station"
                                        st.info(f"Recommended for precision: {note} (closest to latest observed value)")
                                else:
                                    st.warning("Prophet comparison unavailable right now.")
                            except Exception as e:
                                st.warning(f"Prophet comparison failed: {e}")
                    
                    # Show feature importance
                    feature_importance = forecast_data.get("feature_importance", {})
                    if feature_importance:
                        st.subheader("🔍 Feature Importance")
                        
                        # Create feature importance DataFrame
                        importance_data = []
                        for feature, importance in feature_importance.items():
                            importance_data.append({
                                "Feature": feature,
                                "Importance": importance
                            })
                        
                        if importance_data:
                            df_importance = pd.DataFrame(importance_data)
                            df_importance = df_importance.sort_values("Importance", ascending=False)
                            
                            # Plot feature importance
                            fig_importance = px.bar(
                                df_importance.head(10),
                                x="Importance",
                                y="Feature",
                                orientation="h",
                                title="Top 10 Most Important Features"
                            )
                            fig_importance.update_layout(height=400)
                            st.plotly_chart(fig_importance, width='stretch')
                
                else:
                    st.error(f"Failed to generate forecast: {response.text}")
                    
            except Exception as e:
                st.error(f"Error generating cross-station forecast: {e}")
    
    # Model performance
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    if st.button("Get Model Performance"):
        try:
            response = requests.get(f"{ui.api_base_url}/cross-station/model-performance")
            if response.status_code == 200:
                performance_data = response.json()
                
                # Display performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Train MAE", f"{performance_data.get('train_mae', 0):.2f}")
                
                with col2:
                    st.metric("Test MAE", f"{performance_data.get('test_mae', 0):.2f}")
                
                with col3:
                    st.metric("Train RMSE", f"{performance_data.get('train_rmse', 0):.2f}")
                
                with col4:
                    st.metric("Test RMSE", f"{performance_data.get('test_rmse', 0):.2f}")
                
                # Show additional info
                st.info(f"Features: {performance_data.get('n_features', 0)} | "
                       f"Samples: {performance_data.get('n_samples', 0)}")
                
            else:
                st.error("Failed to get model performance")
                
        except Exception as e:
            st.error(f"Error getting model performance: {e}")


def render_about_tab():
    """Render the about tab with comprehensive information"""
    st.header("ℹ️ About AirAware")
    
    # Project Overview
    st.subheader("🌍 Project Overview")
    st.markdown("""
    **AirAware** is a production-grade PM₂.₅ nowcasting system designed to provide accurate air quality 
    forecasts for urban areas. The system combines advanced machine learning models with real-time 
    meteorological data to deliver reliable predictions with uncertainty quantification.
    """)
    
    # Key Features
    st.subheader("✨ Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔮 Forecasting Capabilities:**
        - 6, 12, and 24-hour PM₂.₅ predictions
        - Multiple model types (Prophet, PatchTST, TFT)
        - Calibrated uncertainty bands
        - Real-time model performance monitoring
        
        **📊 Data & Analysis:**
        - Feature importance analysis
        - What-if scenario modeling
        - Historical trend analysis
        - Model explainability
        """)
    
    with col2:
        st.markdown("""
        **🛡️ Health & Safety:**
        - Air quality level categorization
        - Personalized health recommendations
        - Prevention and protection tips
        - Emergency guidance for hazardous conditions
        
        **🌐 User Experience:**
        - Interactive web interface
        - Real-time data visualization
        - Cross-station analysis with global data
        - Mobile-responsive design
        """)
    
    # Technical Architecture
    st.subheader("🏗️ Technical Architecture")
    st.markdown("""
    **Backend Technologies:**
    - **FastAPI**: High-performance API framework
    - **PyTorch**: Deep learning model training and inference
    - **Scikit-learn**: Traditional ML models and preprocessing
    - **Pandas/NumPy**: Data manipulation and analysis
    
    **Frontend Technologies:**
    - **Streamlit**: Interactive web application framework
    - **Plotly**: Advanced data visualization
    - **Custom CSS**: Responsive and accessible design
    
    **Data Sources:**
    - **OpenAQ API**: Real-time air quality measurements
    - **ERA5 Reanalysis**: Meteorological data from Copernicus
    - **US EPA AirNow**: US air quality data
    - **Local monitoring stations**: High-resolution local data
    - **Cross-station integration**: Global air quality networks
    """)
    
    # Model Performance
    st.subheader("📈 Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Baseline Models:**
        - **Prophet**: MAE 7.14 μg/m³ (65.6% improvement over seasonal naive)
        - **ARIMA**: MAE 11.13-15.86 μg/m³ (varies by horizon)
        - **Ensemble**: MAE 12.88-14.51 μg/m³
        
        **Deep Learning Models:**
        - **Simple TFT**: MAE 14.52 μg/m³ (with uncertainty quantification)
        - **PatchTST**: MAE 16.67 μg/m³ (transformer-based)
        """)
    
    with col2:
        st.markdown("""
        **Calibration Quality:**
        - **Coverage**: 88-89% (target: 90%)
        - **Calibration Error**: <0.05 (excellent)
        - **Interval Width**: ~31.5 μg/m³
        - **Winkler Score**: ~41.5
        
        **System Performance:**
        - **Forecast Latency**: <2 seconds
        - **API Response Time**: <100ms
        - **Model Loading**: <5 seconds
        - **Data Processing**: Real-time
        """)
    
    # Health Impact
    st.subheader("🏥 Health Impact & Guidelines")
    st.markdown("""
    **Understanding PM₂.₅ Health Effects:**
    
    PM₂.₅ (fine particulate matter) particles are small enough to penetrate deep into the lungs and 
    enter the bloodstream, causing various health problems:
    
    - **Short-term effects**: Eye, nose, throat irritation, coughing, sneezing, runny nose, shortness of breath
    - **Long-term effects**: Reduced lung function, chronic bronchitis, aggravated asthma, heart disease, lung cancer
    - **Vulnerable groups**: Children, elderly, people with heart or lung disease, pregnant women
    
    **Protective Measures:**
    - Monitor air quality regularly
    - Stay indoors during high pollution periods
    - Use air purifiers with HEPA filters
    - Wear N95 masks when necessary
    - Avoid outdoor exercise during poor air quality
    - Keep windows and doors closed during high pollution
    """)
    
    # Future Roadmap
    st.subheader("🚀 Future Roadmap")
    st.markdown("""
    **Phase 1 (Current):**
    - ✅ Kathmandu Valley coverage
    - ✅ Core forecasting models
    - ✅ Basic web interface
    - ✅ Health recommendations
    
    **Phase 2 (Coming Soon):**
    - 🌍 Multi-country expansion (India, China, Bangladesh)
    - 📱 Mobile application
    - 🔔 Push notifications and alerts
    - 📊 Advanced analytics dashboard
    
    **Phase 3 (Future):**
    - 🤖 AI-powered health recommendations
    - 🌐 Global coverage
    - 🔗 Integration with health apps
    - 📈 Long-term trend analysis
    """)
    
    # Contact & Support
    st.subheader("📞 Contact & Support")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Technical Support:**
        - GitHub Issues: [Report bugs and feature requests]
        - Documentation: [Comprehensive user guides]
        - API Documentation: [Developer resources]
        
        **Data Sources:**
        - OpenAQ: [openaq.org]
        - Copernicus Climate Data Store: [cds.climate.copernicus.eu]
        - Local monitoring networks
        """)
    
    with col2:
        st.markdown("""
        **Research & Development:**
        - Model improvements and validation
        - New feature development
        - Performance optimization
        - User experience enhancements
        
        **Community:**
        - Open source contributions welcome
        - Research collaborations
        - Educational partnerships
        - Public health initiatives
        """)
    
    # Disclaimer
    st.subheader("⚠️ Disclaimer")
    st.markdown("""
    **Important Notice:**
    
    This system provides air quality forecasts for informational purposes only. While we strive for 
    accuracy, forecasts should not be the sole basis for health decisions. Always consult with 
    healthcare professionals for medical advice, especially if you have respiratory or cardiovascular 
    conditions.
    
    **Data Accuracy:**
    - Forecasts are based on statistical models and historical data
    - Actual conditions may vary due to unforeseen factors
    - Real-time measurements may differ from forecasts
    - Use multiple sources for critical health decisions
    
    **Limitations:**
    - Models are trained on historical data and may not capture all future scenarios
    - Local conditions and microclimates may affect accuracy
    - System availability depends on data source reliability
    - Forecast accuracy decreases with longer time horizons
    """)


def main():
    """Main Streamlit application"""
    # Initialize UI
    ui = AirAwareUI()
    
    # Render header
    render_header()
    
    # Render sidebar and get configuration
    config = render_sidebar(ui)
    
    if config is None:
        st.stop()
    
    # Add cross-station configuration to sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("🌍 Cross-Station Analysis")
        
        # External station selection
        external_stations = st.multiselect(
            "Select External Stations",
            options=["India", "China", "USA"],
            default=["India", "China"],
            help="Select external countries to include in cross-station analysis"
        )
        
        # Cross-station forecast toggle
        use_cross_station = st.checkbox(
            "Enable Cross-Station Forecasting",
            value=False,
            help="Use data from external stations to improve predictions"
        )
        
        config.update({
            "external_stations": [country.lower() for country in external_stations],
            "use_cross_station": use_cross_station
        })
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Forecast", "🔍 Explainability", "🎯 What-If Analysis", "🤖 Intelligent Agents", "🌍 Cross-Station", "ℹ️ About"])
    
    with tab1:
        render_forecast_tab(ui, config)
    
    with tab2:
        render_explainability_tab(ui, config)
    
    with tab3:
        render_whatif_tab(ui, config)
    
    with tab4:
        render_agents_tab(ui, config)
    
    with tab5:
        render_cross_station_tab(ui, config)
    
    with tab6:
        render_about_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🌍 AirAware - Production-grade PM₂.₅ nowcasting for Kathmandu Valley<br>
        Built with ❤️ using FastAPI, Streamlit, and advanced ML models
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
