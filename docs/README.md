# OpenAQ Air Quality Data Ingestion for Nepal, India & China

A comprehensive Python system for collecting air quality data from the OpenAQ API, specifically designed to gather data from Nepal and surrounding countries (India and China) for air quality prediction modeling.

## 🎯 Project Overview

This system provides:
- **Full-scale data collection** from OpenAQ API v3
- **Multi-country coverage**: Nepal, India, and China
- **Production-ready pipeline** with error handling and logging
- **Efficient data storage** in compressed Parquet format
- **Comprehensive analysis tools** for data quality assessment
- **Prediction-ready datasets** for machine learning models

## 📊 Expected Data Collection

- **Countries**: 3 (Nepal, India, China)
- **Locations**: ~3,000+ monitoring stations
- **Sensors**: ~15,000+ air quality sensors
- **Measurements**: 10-50 million historical records
- **Parameters**: PM2.5, PM10, O3, NO2, SO2, CO, temperature, humidity
- **Time Range**: 2015-present (10+ years of data)
- **Storage**: 5-15 GB compressed data

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install pandas requests pyarrow matplotlib seaborn
```

### 2. Run Data Collection

```bash
# Quick test run (recommended first)
python launch_ingestion.py --test

# Full-scale collection (4-8 hours)
python launch_ingestion.py

# Analyze collected data
python launch_ingestion.py --analyze
```

## 📁 Project Structure

```
openaq_ingestion/
├── launch_ingestion.py          # 🚀 Main launcher script
├── run_full_ingestion.py        # 🏭 Full-scale ingestion engine
├── test_ingestion.py            # 🧪 Test/validation script
├── analyze_results.py           # 📊 Data analysis tools
├── config.py                    # ⚙️ Configuration settings
├── ingest_openaq_data.py        # 📝 Original development script
├── README.md                    # 📖 This documentation
├── venv/                        # 🐍 Python virtual environment
├── test_data/                   # 🧪 Test ingestion outputs
│   ├── test_locations_*.parquet
│   ├── test_sensors_*.parquet
│   └── test_measurements_*.parquet
└── full_data/                   # 🏭 Full ingestion outputs
    ├── all_locations_*.parquet
    ├── all_sensors_*.parquet
    ├── measurements_batch_*.parquet
    ├── ingestion_summary_*.json
    └── data_analysis_report_*.txt
```

## 🛠️ Script Descriptions

### Core Scripts

1. **`launch_ingestion.py`** - Smart launcher with environment checks
   - Validates dependencies and setup
   - Provides resource estimates
   - Offers test, full, and analysis modes
   - User-friendly interface with confirmations

2. **`run_full_ingestion.py`** - Production ingestion engine
   - Multi-threaded data collection
   - Comprehensive error handling and retries
   - Progress tracking and logging
   - Memory-efficient batch processing
   - Automatic data validation

3. **`test_ingestion.py`** - Quick validation script
   - Limited data collection for testing
   - Validates API connectivity
   - Tests data pipeline
   - ~30 minutes runtime

4. **`analyze_results.py`** - Data analysis and quality assessment
   - Comprehensive data statistics
   - Parameter availability analysis
   - Prediction readiness assessment
   - Geographic and temporal coverage
   - Data quality reporting

### Configuration

5. **`config.py`** - Centralized configuration
   - API settings and rate limiting
   - Country selection
   - Performance tuning
   - Output formatting options

## 📊 Data Schema

### Locations Data
```
all_locations_YYYYMMDD_HHMMSS.parquet
├── id                   # Location ID
├── name                 # Location name
├── locality             # City/region
├── country              # Country details (name, code, ID)
├── coordinates          # Latitude, longitude
├── timezone             # Local timezone
├── sensors              # Associated sensors list
├── owner                # Data owner info
├── provider             # Data provider info
├── datetimeFirst        # First measurement date
├── datetimeLast         # Last measurement date
├── isMobile             # Mobile vs fixed station
└── isMonitor            # Monitoring station flag
```

### Sensors Data
```
all_sensors_YYYYMMDD_HHMMSS.parquet
├── sensor_id            # Unique sensor ID
├── sensor_name          # Sensor name
├── parameter            # Parameter details (name, units, display)
├── location_id          # Parent location ID
├── location_name        # Location name
├── country              # Country details
├── coordinates          # Geographic coordinates
├── timezone             # Local timezone
├── owner                # Data owner
├── provider             # Data provider
└── datetime_first/last  # Temporal coverage
```

### Measurements Data
```
measurements_batch_XXXX_YYYYMMDD_HHMMSS.parquet
├── date                 # Measurement timestamp (ISO format)
├── value                # Measured value
├── parameter            # Parameter type (pm25, o3, etc.)
├── units                # Measurement units
├── period               # Averaging period
├── sensor_id            # Source sensor ID
├── location_id          # Source location ID
├── location_name        # Location name
├── country_name         # Country name
├── country_code         # Country code
├── coordinates_lat      # Latitude
├── coordinates_lon      # Longitude
├── timezone             # Local timezone
├── provider_name        # Data provider
└── ingestion_timestamp  # When data was collected
```

## 🔧 Configuration Options

Edit `config.py` to customize:

```python
# Performance settings
REQUEST_DELAY = 0.5          # Seconds between API requests
BATCH_SIZE = 50000           # Measurements per file
MAX_WORKERS = 3              # Concurrent requests

# Date range
DATE_FROM = "2015-01-01T00:00:00Z"
DATE_TO = None               # None = current time

# Countries (add/remove as needed)
TARGET_COUNTRIES = {
    "Nepal": 145,
    "India": 9,
    "China": 10
}
```

## 📈 Usage Examples

### Basic Usage

```bash
# Environment check and test run
python launch_ingestion.py --test

# Full data collection
python launch_ingestion.py

# Analyze existing data
python launch_ingestion.py --analyze
```

### Advanced Usage

```bash
# Run with custom config
python run_full_ingestion.py

# Direct analysis of specific data
python analyze_results.py full_data

# Environment check only
python launch_ingestion.py --skip-check
```

### Programmatic Usage

```python
# Using the ingestion classes directly
from run_full_ingestion import FullScaleOpenAQIngester

api_key = "your_api_key_here"
ingester = FullScaleOpenAQIngester(api_key, "my_output_dir")
ingester.run_full_ingestion()

# Analysis
from analyze_results import OpenAQDataAnalyzer

analyzer = OpenAQDataAnalyzer("my_output_dir")
analyzer.run_complete_analysis()
```

## 🚦 Performance and Monitoring

### Monitoring Progress

The ingestion provides real-time progress updates:
- Countries processed
- Locations and sensors collected
- Measurements ingested
- Estimated completion time
- Current processing status

### Log Files

Comprehensive logging includes:
- API request/response details
- Error tracking and retries
- Performance metrics
- Progress checkpoints
- Final statistics

### Resource Usage

- **CPU**: Light (I/O bound operations)
- **Memory**: 2-4 GB recommended
- **Network**: ~10,000 API requests
- **Storage**: 5-15 GB for full dataset

## 🔍 Data Quality Features

### Built-in Validation
- Parameter standardization
- Geographic coordinate validation
- Timestamp consistency checks
- Unit conversion verification
- Duplicate detection

### Analysis Capabilities
- Data completeness scoring
- Parameter availability matrix
- Temporal coverage analysis
- Geographic distribution mapping
- Prediction readiness assessment

## 🤖 Machine Learning Ready

The collected data is optimized for:

### Time Series Forecasting
- Hourly measurement resolution
- Multi-year historical data
- Consistent temporal indexing
- Missing data flagging

### Spatial Modeling
- Dense sensor networks
- Geographic coordinates
- Administrative boundaries
- Cross-border coverage

### Multi-parameter Prediction
- Correlated pollutants (PM2.5, PM10, O3)
- Meteorological data (temperature, humidity)
- Source attribution parameters
- Air quality indices

## ⚡ API Rate Limiting

The system implements responsible API usage:
- 0.5-1 second delays between requests
- Automatic retry with exponential backoff
- Rate limit detection and handling
- Respectful concurrent request limits

## 🛡️ Error Handling

Robust error management includes:
- Network timeout recovery
- HTTP error handling
- JSON parsing validation
- Partial data preservation
- Graceful degradation

## 📝 Output Files

### Data Files
- **Parquet format**: Efficient, compressed, cross-platform
- **Batch organization**: Memory-efficient processing
- **Metadata preservation**: Full context retention
- **Schema consistency**: Standardized column structure

### Reports
- **Ingestion summary**: JSON metadata
- **Analysis report**: Comprehensive text report
- **Log files**: Detailed operation logs
- **Progress tracking**: Real-time status updates

## 🔮 Future Enhancements

Potential improvements:
- Real-time data streaming
- Automated data updates
- Advanced quality control
- Integration with prediction models
- Web-based monitoring dashboard
- Data visualization tools

## 🐛 Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   # Verify API key in config.py
   # Check OpenAQ account status
   ```

2. **Network Problems**
   ```bash
   # Check internet connectivity
   # Verify firewall settings
   # Test with --test flag first
   ```

3. **Memory Issues**
   ```bash
   # Reduce BATCH_SIZE in config.py
   # Close other applications
   # Monitor with system tools
   ```

4. **Disk Space**
   ```bash
   # Check available space (5-15 GB needed)
   # Clean old data files
   # Use external storage if needed
   ```

### Debug Mode

Enable detailed logging:
```python
# In config.py
LOG_LEVEL = "DEBUG"
SAVE_RAW_RESPONSES = True
```

## 📞 Support

For issues or questions:
1. Check log files for error details
2. Run test ingestion first
3. Verify environment setup
4. Review configuration settings
5. Check OpenAQ API status

## 📄 License

This project is designed for educational and research purposes. Please respect OpenAQ's terms of service and rate limits.

---

**Happy Air Quality Monitoring! 🌍💨📊**  
✅ **File Output**: Parquet files generated correctly  
✅ **Error Handling**: Graceful handling of API timeouts and server errors  

## Usage Instructions

### 1. Setup (Already Done)
```bash
python3 -m venv venv
source venv/bin/activate
pip install requests pandas pyarrow
```

### 2. Run Full Ingestion (Will take several hours)
```bash
source venv/bin/activate
python ingest_openaq_data.py
```

### 3. Run Limited Test (Quick test with Nepal only)
```bash
source venv/bin/activate  
python test_ingestion.py
```

## Output Files

- `locations_TIMESTAMP.parquet`: All location data for Nepal, India, China
- `sensors_TIMESTAMP.parquet`: All sensor data with location context
- `measurements_*.parquet`: All measurement data (may be split into batches)
- `*.log`: Detailed logging of the ingestion process

## API Details

- **Base URL**: https://api.openaq.org/v3
- **Country IDs**: Nepal=145, India=9, China=10
- **Rate Limiting**: 1 second between requests
- **Pagination**: Handled automatically
- **Error Handling**: Robust with retry logic and logging

## Data Structure

1. **Locations**: Geographic points with sensors
2. **Sensors**: Individual measurement devices at locations  
3. **Measurements**: Time-series data from sensors

## Expected Data Volume

Based on test results:
- **Nepal alone**: 14 locations, 66 sensors
- **Test sample**: 31,000 measurements from just 2 sensors
- **Full dataset estimate**: Several million measurement records
- **File sizes**: Gigabytes when complete

## Performance Notes

- **Test ingestion**: ~4 minutes for 2 sensors (31,000 measurements)
- **Full ingestion estimate**: 4-8 hours depending on API performance
- **Memory management**: Batches data to avoid memory issues
- **Error resilience**: Continues processing even if some API calls fail

## Production Recommendations

1. **API Key Security**: Move API key to environment variable
2. **Incremental Updates**: Modify to only fetch new data since last run
3. **Monitoring**: Add alerting for failed ingestions
4. **Storage**: Consider database storage for large datasets
5. **Scheduling**: Set up cron job for regular data updates