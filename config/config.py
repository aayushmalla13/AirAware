# Full-Scale OpenAQ Data Ingestion Configuration

# API Configuration
API_KEY = "9a1e83c1ac07424296a0e90aa95153fe13c7c8b8996d5f4d712c95484d2d1588"
BASE_URL = "https://api.openaq.org/v3"

# Target Countries (Name: ID mapping)
TARGET_COUNTRIES = {
    "Nepal": 145,
    "India": 9,
    "China": 10
}

# Output Configuration
OUTPUT_DIR = "data/full_data"
COMPRESSION = "snappy"  # Options: snappy, gzip, brotli

# Performance Settings
REQUEST_DELAY = 0.3         # Seconds between API requests (0.3 = ~3 requests/sec)
BATCH_SIZE = 75000          # Measurements per parquet file
MAX_WORKERS = 3             # Concurrent API requests
SENSOR_BATCH_SIZE = 75      # Sensors processed in parallel

# Date Range (for measurements)
DATE_FROM = "2024-01-01T00:00:00Z"  # Start collecting from this date (2024 onwards)
DATE_TO = None                       # None = current time, or specify end date

# Retry Configuration
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
RATE_LIMIT_WAIT = 60        # Seconds to wait when rate limited

# Logging Configuration
LOG_LEVEL = "INFO"          # DEBUG, INFO, WARNING, ERROR
PROGRESS_LOG_INTERVAL = 10  # Log progress every N sensors

# Data Quality Settings
INCLUDE_MOBILE_SENSORS = True    # Include mobile monitoring stations
INCLUDE_MONITOR_SENSORS = True   # Include fixed monitoring stations
MIN_MEASUREMENTS_PER_SENSOR = 1  # Skip sensors with fewer measurements

# Memory Management
MEMORY_LIMIT_GB = 8         # Approximately when to save batches
ENABLE_GARBAGE_COLLECTION = True

# Advanced Options
SAVE_RAW_RESPONSES = False  # Save raw API responses for debugging
VALIDATE_DATA_TYPES = True  # Validate data before saving
ENABLE_CHECKPOINTING = True # Save progress to resume interrupted runs