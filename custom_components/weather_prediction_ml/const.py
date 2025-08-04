"""Constants for the Weather Prediction ML integration."""
from datetime import timedelta

DOMAIN = "weather_prediction_ml"

# Configuration
CONF_TEMPERATURE_SENSOR = "temperature_sensor"
CONF_HUMIDITY_SENSOR = "humidity_sensor"
CONF_PRESSURE_SENSOR = "pressure_sensor"
CONF_UPDATE_INTERVAL = "update_interval"
CONF_AUTO_RETRAIN = "auto_retrain"
CONF_RETRAIN_SCHEDULE = "retrain_schedule"

# Defaults
DEFAULT_UPDATE_INTERVAL = 3600  # 1 hour in seconds
DEFAULT_AUTO_RETRAIN = False
DEFAULT_RETRAIN_SCHEDULE = "0 2 * * 0"  # Weekly at 2 AM Sunday

# Sensor types
SENSOR_TYPES = {
    "prediction": {
        "name": "Weather Prediction",
        "icon": "mdi:weather-partly-cloudy",
        "unit": None,
    },
    "trend": {
        "name": "Weather Trend",
        "icon": "mdi:trending-up",
        "unit": None,
    },
    "confidence": {
        "name": "Prediction Confidence",
        "icon": "mdi:gauge",
        "unit": "%",
    },
    "increase_probability": {
        "name": "Temperature Increase Probability",
        "icon": "mdi:thermometer-chevron-up",
        "unit": "%",
    },
    "decrease_probability": {
        "name": "Temperature Decrease Probability",
        "icon": "mdi:thermometer-chevron-down",
        "unit": "%",
    },
    "stable_probability": {
        "name": "Temperature Stable Probability",
        "icon": "mdi:thermometer",
        "unit": "%",
    },
}

# Model configuration
MODEL_ACCURACY = 82.2
MODEL_TYPE = "Ensemble (XGBoost + Random Forest)"
SEQUENCE_LENGTH = 48  # Hours of historical data needed

# Update coordinator
SCAN_INTERVAL = timedelta(seconds=DEFAULT_UPDATE_INTERVAL)

# Services
SERVICE_PREDICT = "predict"
SERVICE_RETRAIN = "retrain_model"
SERVICE_GET_DIAGNOSTICS = "get_diagnostics"

# Attributes
ATTR_CONFIDENCE = "confidence"
ATTR_PROBABILITIES = "probabilities"
ATTR_LAST_UPDATE = "last_update"
ATTR_MODEL_TYPE = "model_type"
ATTR_MODEL_ACCURACY = "model_accuracy"
ATTR_DESCRIPTION = "description"
ATTR_EXPECTED_CHANGE = "expected_change"