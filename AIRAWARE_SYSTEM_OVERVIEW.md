# AirAware System Overview - With Intelligent Agents

## 🌟 What is AirAware?

**AirAware** is a production-grade PM₂.₅ air quality nowcasting system that provides 6-24 hour forecasts for Kathmandu Valley. The system now includes **intelligent agents** that provide automated health advisory, forecast optimization, data quality monitoring, and smart notifications.

## 🎯 What Does the Product Do?

### Core Capabilities

1. **PM₂.₅ Forecasting**
   - 6, 12, and 24-hour predictions
   - Multiple model types (Prophet, PatchTST, Simple TFT, Ensemble)
   - Calibrated uncertainty quantification
   - Real-time nowcasting

2. **Intelligent Agents** ⭐ **NEW**
   - **Health Advisory Agent**: Personalized health recommendations based on air quality
   - **Forecast Optimization Agent**: Automatic model selection and performance optimization
   - **Data Quality Monitoring Agent**: Real-time data validation and anomaly detection
   - **Notification Agent**: Smart alerts and multi-channel notifications
   - **Agent Orchestrator**: Coordinated workflows and system insights

3. **Explainable AI**
   - Feature importance analysis
   - SHAP explanations
   - What-if scenario analysis
   - Model interpretability

4. **Health Guidance**
   - Air quality level categorization (Good, Moderate, Unhealthy, Hazardous)
   - Health recommendations by risk level
   - Prevention tips and protective measures
   - Emergency guidance for high pollution days

## 🚀 How Can End Users Access This Project?

### 1. Web Interface (Streamlit UI)

**URL**: `http://localhost:8501` (when running locally)

**Features Available to Users**:

#### 📈 Forecast Tab
- **Real-time PM₂.₅ predictions** with uncertainty bands
- **Current air quality status** with color-coded indicators
- **Health recommendations** based on current conditions
- **Prevention tips** and protective measures
- **Interactive charts** with AQI thresholds
- **Detailed forecast tables** with timestamps

#### 🔍 Explainability Tab
- **Feature importance analysis** showing which factors most influence predictions
- **Model transparency** with multiple explanation methods
- **Interactive visualizations** of feature contributions

#### 🎯 What-If Analysis Tab
- **Scenario analysis** - "What if temperature increases by 5°C?"
- **Sensitivity analysis** for understanding model behavior
- **Counterfactual explanations** for decision support

#### 🤖 Intelligent Agents Tab ⭐ **NEW**
- **Health Advisory Agent**: Get personalized health recommendations
- **Forecast Optimization**: See which models perform best
- **Data Quality Monitoring**: Check data reliability
- **System Status**: Monitor all agent health and performance
- **Real-time agent execution** with detailed results

#### ℹ️ About Tab
- **System documentation** and technical details
- **Performance metrics** and model information
- **Contact information** and support resources

### 2. API Access (FastAPI Backend)

**Base URL**: `http://localhost:8000` (when running locally)

**Available Endpoints**:

#### Core Forecasting
- `POST /forecast` - Generate PM₂.₅ forecasts
- `GET /models` - Available model information
- `GET /stations` - Available monitoring stations

#### Explainability
- `POST /explainability` - Feature importance analysis
- `POST /what-if` - What-if scenario analysis

#### Intelligent Agents ⭐ **NEW**
- `POST /agents/execute` - Execute specific agents
- `POST /workflows/execute` - Run coordinated workflows
- `GET /system/status` - Comprehensive system status

#### System Health
- `GET /health` - API health check
- `GET /` - API documentation and endpoints

### 3. Command Line Interface

**Available CLI Tools**:
- `python scripts/deep_models_cli.py` - Train and evaluate deep learning models
- `python scripts/calibration_cli.py` - Calibration methods
- `python scripts/explainability_cli.py` - Explainability analysis
- `python scripts/agents_cli.py` - Agent management and testing ⭐ **NEW**

## 📊 What Information Will Users Get?

### Air Quality Information
1. **Current PM₂.₅ levels** in μg/m³
2. **6-24 hour forecasts** with uncertainty bounds
3. **Air Quality Index (AQI) categories**:
   - 🟢 **Good** (0-12 μg/m³): Safe for everyone
   - 🟡 **Moderate** (12.1-35.4 μg/m³): Sensitive groups should limit outdoor activities
   - 🟠 **Unhealthy for Sensitive Groups** (35.5-55.4 μg/m³): Health warnings
   - 🔴 **Unhealthy** (55.5-150.4 μg/m³): Everyone should limit outdoor activities
   - 🟣 **Very Unhealthy** (150.5-250.4 μg/m³): Health alert
   - 🔴 **Hazardous** (250.5+ μg/m³): Emergency conditions

### Health Guidance ⭐ **Enhanced with Agents**
1. **Personalized recommendations** based on user profile (age, health conditions)
2. **Activity suggestions** (outdoor exercise, window opening, etc.)
3. **Protection measures** (mask wearing, air purifiers)
4. **Emergency protocols** for hazardous conditions
5. **Medication reminders** for sensitive individuals
6. **Real-time health alerts** via multiple channels

### Technical Insights
1. **Model performance metrics** (MAE, RMSE, coverage)
2. **Feature importance rankings** showing key pollution drivers
3. **Prediction confidence levels** and uncertainty quantification
4. **Data quality indicators** and reliability scores ⭐ **NEW**
5. **System health status** and agent performance ⭐ **NEW**

### Intelligent Agent Results ⭐ **NEW**
1. **Health Advisory Results**:
   - Risk level assessment
   - Personalized recommendations
   - Emergency alerts if needed
   - Preventive measures

2. **Forecast Optimization Results**:
   - Best performing model recommendations
   - Performance insights and trends
   - Model selection confidence

3. **Data Quality Reports**:
   - Overall quality score
   - Specific metric assessments
   - Anomaly detection results
   - Reliability indicators

4. **System Insights**:
   - Cross-agent analysis
   - Performance optimization suggestions
   - Workflow execution results
   - System health diagnostics

## 🚀 How to Start the System

### Option 1: Start Everything Together
```bash
cd /path/to/AirQuality
python scripts/start_airaware.py
```

### Option 2: Start Components Separately
```bash
# Start API Backend
python scripts/start_api.py

# Start UI Frontend (in another terminal)
python scripts/start_ui.py
```

### Option 3: Manual Startup
```bash
# API Backend
uvicorn airaware.api.app:app --host 0.0.0.0 --port 8000

# UI Frontend
streamlit run airaware/ui/app.py --server.port 8501
```

## 🌍 Future Expansion

The system is designed for scalability:
- **Multi-location support**: Currently Kathmandu Valley, expandable to other cities
- **Multiple countries**: Framework ready for international deployment
- **Additional pollutants**: Extensible to other air quality parameters
- **Enhanced agent capabilities**: More sophisticated AI agents and workflows

## 🔒 Data Privacy & Security

- **Local deployment**: All data processing happens on your infrastructure
- **No external data sharing**: User data stays within your environment
- **Configurable privacy settings**: Control what data is collected and stored
- **Secure API endpoints**: Authentication and rate limiting available

## 📞 Support & Documentation

- **API Documentation**: Available at `http://localhost:8000/docs`
- **System Status**: Monitor at `http://localhost:8000/system/status`
- **Health Check**: Verify at `http://localhost:8000/health`
- **Agent Management**: Access via the UI's "Intelligent Agents" tab

---

**AirAware** provides comprehensive air quality intelligence with the added power of intelligent agents for automated health advisory, system optimization, and proactive monitoring. Users get actionable insights, personalized recommendations, and reliable forecasts to make informed decisions about their health and daily activities.
