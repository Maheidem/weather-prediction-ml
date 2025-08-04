"""Sensor platform for Weather Prediction ML."""
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
    UpdateFailed,
)
from homeassistant.util import dt

from .const import (
    DOMAIN,
    CONF_TEMPERATURE_SENSOR,
    CONF_HUMIDITY_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_UPDATE_INTERVAL,
    DEFAULT_UPDATE_INTERVAL,
    SENSOR_TYPES,
    MODEL_ACCURACY,
    MODEL_TYPE,
    SEQUENCE_LENGTH,
    ATTR_CONFIDENCE,
    ATTR_PROBABILITIES,
    ATTR_LAST_UPDATE,
    ATTR_MODEL_TYPE,
    ATTR_MODEL_ACCURACY,
    ATTR_DESCRIPTION,
    ATTR_EXPECTED_CHANGE,
)

_LOGGER = logging.getLogger(__name__)


class WeatherPredictionCoordinator(DataUpdateCoordinator):
    """Coordinator to manage weather prediction updates."""
    
    def __init__(self, hass: HomeAssistant, config: dict):
        """Initialize the coordinator."""
        self.config = config
        self.models_loaded = False
        self.xgboost_model = None
        self.rf_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.ensemble_weights = None
        
        # Load models
        self._load_models()
        
        update_interval = config.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=update_interval),
        )
    
    def _load_models(self):
        """Load ML models from files."""
        try:
            models_path = Path(__file__).parent / "models"
            
            # Load model config
            with open(models_path / "final_model_config.json", "r") as f:
                config = json.load(f)
            
            self.feature_columns = config["feature_columns"]
            self.ensemble_weights = config["ensemble_weights"]
            
            # Load models
            with open(models_path / "final_xgboost_model.pkl", "rb") as f:
                self.xgboost_model = pickle.load(f)
            
            with open(models_path / "final_rf_model.pkl", "rb") as f:
                self.rf_model = pickle.load(f)
            
            # Load preprocessors
            with open(models_path / "final_scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            
            with open(models_path / "final_label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)
            
            self.models_loaded = True
            _LOGGER.info("ML models loaded successfully")
            
        except Exception as e:
            _LOGGER.error(f"Failed to load models: {e}")
            self.models_loaded = False
    
    async def _async_update_data(self):
        """Fetch data and make prediction."""
        if not self.models_loaded:
            raise UpdateFailed("Models not loaded")
        
        try:
            # Get sensor data
            temp_sensor = self.config[CONF_TEMPERATURE_SENSOR]
            humidity_sensor = self.config[CONF_HUMIDITY_SENSOR]
            pressure_sensor = self.config[CONF_PRESSURE_SENSOR]
            
            # Fetch historical data from recorder
            history_data = await self._get_sensor_history()
            
            if len(history_data) < SEQUENCE_LENGTH:
                _LOGGER.warning(
                    f"Insufficient data: {len(history_data)} records, need {SEQUENCE_LENGTH}"
                )
                return None
            
            # Prepare features
            features_df = await self.hass.async_add_executor_job(
                self._prepare_features, history_data
            )
            
            # Make prediction
            prediction = await self.hass.async_add_executor_job(
                self._make_prediction, features_df
            )
            
            return prediction
            
        except Exception as e:
            _LOGGER.error(f"Error updating weather prediction: {e}")
            raise UpdateFailed(f"Error updating weather prediction: {e}")
    
    async def _get_sensor_history(self):
        """Get sensor history from recorder."""
        from homeassistant.components.recorder import get_instance, history
        
        end_time = dt.utcnow()
        start_time = end_time - timedelta(hours=SEQUENCE_LENGTH + 24)  # Extra buffer
        
        temp_sensor = self.config[CONF_TEMPERATURE_SENSOR]
        humidity_sensor = self.config[CONF_HUMIDITY_SENSOR]
        pressure_sensor = self.config[CONF_PRESSURE_SENSOR]
        
        # Get history for all sensors
        history_data = await get_instance(self.hass).async_add_executor_job(
            history.get_significant_states,
            self.hass,
            start_time,
            end_time,
            [temp_sensor, humidity_sensor, pressure_sensor],
        )
        
        # Process history data
        processed_data = []
        
        # Align timestamps and create records
        for timestamp in pd.date_range(start=start_time, end=end_time, freq='H'):
            record = {'timestamp': timestamp}
            
            # Get values for each sensor at this timestamp
            for sensor_id, sensor_key in [
                (temp_sensor, 'temperature'),
                (humidity_sensor, 'humidity'),
                (pressure_sensor, 'pressure'),
            ]:
                if sensor_id in history_data:
                    # Find closest value to timestamp
                    value = self._get_sensor_value_at_time(
                        history_data[sensor_id], timestamp
                    )
                    record[sensor_key] = value
            
            if all(k in record for k in ['temperature', 'humidity', 'pressure']):
                processed_data.append(record)
        
        return processed_data
    
    def _get_sensor_value_at_time(self, states, target_time):
        """Get sensor value at or before target time."""
        for state in reversed(states):
            if state.last_changed <= target_time:
                try:
                    return float(state.state)
                except (ValueError, TypeError):
                    continue
        return None
    
    def _prepare_features(self, history_data):
        """Prepare features for prediction."""
        # Convert to DataFrame
        df = pd.DataFrame(history_data)
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        # Add time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Create lag features
        for col in ['temperature', 'humidity', 'pressure']:
            for lag in [1, 2, 3, 6, 12, 24, 48]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling statistics
        for col in ['temperature', 'humidity', 'pressure']:
            for window in [6, 12, 24, 48]:
                df[f'{col}_mean_{window}h'] = df[col].rolling(window).mean()
                df[f'{col}_std_{window}h'] = df[col].rolling(window).std()
                if col == 'temperature':
                    df[f'{col}_min_{window}h'] = df[col].rolling(window).min()
                    df[f'{col}_max_{window}h'] = df[col].rolling(window).max()
                    df[f'{col}_range_{window}h'] = (
                        df[f'{col}_max_{window}h'] - df[f'{col}_min_{window}h']
                    )
        
        # Change features
        for col in ['temperature', 'humidity', 'pressure']:
            for period in [1, 3, 6, 12, 24]:
                df[f'{col}_change_{period}h'] = df[col].diff(period)
                if col == 'temperature':
                    df[f'{col}_pct_change_{period}h'] = df[col].pct_change(period)
        
        # EMA features
        for col in ['temperature', 'humidity', 'pressure']:
            for span in [6, 12, 24]:
                df[f'{col}_ema_{span}'] = df[col].ewm(span=span).mean()
        
        # Interaction features
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
        df['temp_pressure_interaction'] = df['temperature'] * df['pressure']
        df['humidity_pressure_interaction'] = df['humidity'] * df['pressure']
        
        # Derived features
        df['dewpoint'] = df['temperature'] - ((100 - df['humidity']) / 5)
        df['heat_index'] = df['temperature'] + 0.5555 * (
            6.11 * np.exp(5417.7530 * (1/273.16 - 1/(273.15 + df['dewpoint']))) - 10
        )
        
        # Volatility
        df['temp_volatility_6h'] = df['temperature'].rolling(6).std()
        df['temp_volatility_24h'] = df['temperature'].rolling(24).std()
        
        # Trend
        df['temp_trend_6h'] = (
            df['temperature'].rolling(6).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
            )
        )
        df['temp_trend_24h'] = (
            df['temperature'].rolling(24).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
            )
        )
        
        # Add missing features with default values
        for col in ['temperature_mean', 'temperature_min', 'temperature_max',
                    'humidity_mean', 'humidity_min', 'humidity_max',
                    'pressure_mean', 'pressure_min', 'pressure_max',
                    'temperature_range', 'humidity_range',
                    'solar_azimuth', 'solar_elevation', 'atmosphere_pressure']:
            if col not in df.columns:
                if 'mean' in col:
                    base_col = col.replace('_mean', '')
                    if base_col in df.columns:
                        df[col] = df[base_col].rolling(24).mean()
                elif 'min' in col:
                    base_col = col.replace('_min', '')
                    if base_col in df.columns:
                        df[col] = df[base_col].rolling(24).min()
                elif 'max' in col:
                    base_col = col.replace('_max', '')
                    if base_col in df.columns:
                        df[col] = df[base_col].rolling(24).max()
                elif 'range' in col:
                    base_col = col.replace('_range', '')
                    if base_col in df.columns:
                        df[col] = df[base_col].rolling(24).max() - df[base_col].rolling(24).min()
                else:
                    df[col] = 0  # Default value
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select only required features
        available_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        
        if missing_features:
            _LOGGER.warning(f"Missing features: {missing_features}")
            # Add missing features with zeros
            for col in missing_features:
                df[col] = 0
        
        return df[self.feature_columns]
    
    def _make_prediction(self, features_df):
        """Make ensemble prediction."""
        # Use last row for prediction
        X = features_df.iloc[-1:].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        xgb_pred = self.xgboost_model.predict(X_scaled)[0]
        xgb_proba = self.xgboost_model.predict_proba(X_scaled)[0]
        
        rf_pred = self.rf_model.predict(X_scaled)[0]
        rf_proba = self.rf_model.predict_proba(X_scaled)[0]
        
        # Weighted ensemble
        ensemble_proba = (
            self.ensemble_weights['xgboost'] * xgb_proba +
            self.ensemble_weights['random_forest'] * rf_proba
        )
        
        # Get final prediction
        pred_idx = np.argmax(ensemble_proba)
        pred_class = self.label_encoder.classes_[pred_idx]
        confidence = float(ensemble_proba[pred_idx] * 100)
        
        # Create probability dict
        probabilities = {
            cls: float(prob * 100)
            for cls, prob in zip(self.label_encoder.classes_, ensemble_proba)
        }
        
        return {
            'prediction': pred_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'timestamp': datetime.now(),
        }


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up weather prediction sensors from a config entry."""
    config = config_entry.data
    
    # Create coordinator
    coordinator = WeatherPredictionCoordinator(hass, config)
    
    # Store coordinator
    hass.data[DOMAIN][config_entry.entry_id]["coordinator"] = coordinator
    
    # Fetch initial data
    await coordinator.async_config_entry_first_refresh()
    
    # Create sensors
    sensors = []
    for sensor_type in SENSOR_TYPES:
        sensors.append(
            WeatherPredictionSensor(coordinator, sensor_type, config_entry.entry_id)
        )
    
    async_add_entities(sensors, True)


class WeatherPredictionSensor(CoordinatorEntity, SensorEntity):
    """Weather Prediction sensor."""
    
    def __init__(self, coordinator, sensor_type, entry_id):
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._sensor_type = sensor_type
        self._entry_id = entry_id
        self._attr_unique_id = f"{DOMAIN}_{entry_id}_{sensor_type}"
        self._attr_has_entity_name = True
        
        # Set sensor attributes from SENSOR_TYPES
        sensor_config = SENSOR_TYPES[sensor_type]
        self._attr_name = sensor_config["name"]
        self._attr_icon = sensor_config["icon"]
        self._attr_native_unit_of_measurement = sensor_config["unit"]
        
        if sensor_type in ["confidence", "increase_probability", 
                          "decrease_probability", "stable_probability"]:
            self._attr_state_class = SensorStateClass.MEASUREMENT
    
    @property
    def device_info(self) -> DeviceInfo:
        """Return device info."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry_id)},
            name="Weather Prediction ML",
            manufacturer="Custom",
            model=MODEL_TYPE,
            sw_version="1.0.0",
        )
    
    @property
    def native_value(self):
        """Return the state of the sensor."""
        if self.coordinator.data is None:
            return None
        
        data = self.coordinator.data
        
        if self._sensor_type == "prediction":
            return data.get('prediction', 'unknown')
        elif self._sensor_type == "trend":
            prediction = data.get('prediction', 'unknown')
            trend_map = {
                'increase': '↑',
                'decrease': '↓',
                'stable': '→'
            }
            return trend_map.get(prediction, '?')
        elif self._sensor_type == "confidence":
            return round(data.get('confidence', 0), 1)
        elif self._sensor_type == "increase_probability":
            return round(data.get('probabilities', {}).get('increase', 0), 1)
        elif self._sensor_type == "decrease_probability":
            return round(data.get('probabilities', {}).get('decrease', 0), 1)
        elif self._sensor_type == "stable_probability":
            return round(data.get('probabilities', {}).get('stable', 0), 1)
        
        return None
    
    @property
    def extra_state_attributes(self):
        """Return extra state attributes."""
        if self.coordinator.data is None:
            return {}
        
        data = self.coordinator.data
        base_attrs = {
            ATTR_MODEL_TYPE: MODEL_TYPE,
            ATTR_MODEL_ACCURACY: MODEL_ACCURACY,
            ATTR_LAST_UPDATE: data.get('timestamp', '').isoformat() if data.get('timestamp') else None,
        }
        
        if self._sensor_type == "prediction":
            base_attrs.update({
                ATTR_CONFIDENCE: data.get('confidence'),
                ATTR_PROBABILITIES: data.get('probabilities'),
            })
        elif self._sensor_type == "trend":
            prediction = data.get('prediction', 'unknown')
            descriptions = {
                'increase': 'Temperature expected to increase',
                'decrease': 'Temperature expected to decrease',
                'stable': 'Temperature expected to remain stable'
            }
            base_attrs.update({
                ATTR_DESCRIPTION: descriptions.get(prediction, 'Unknown trend'),
                ATTR_EXPECTED_CHANGE: prediction,
            })
        
        return base_attrs
    
    @property
    def icon(self):
        """Return dynamic icon based on state."""
        if self._sensor_type == "trend" and self.coordinator.data:
            prediction = self.coordinator.data.get('prediction', 'unknown')
            icon_map = {
                'increase': 'mdi:trending-up',
                'decrease': 'mdi:trending-down',
                'stable': 'mdi:trending-neutral'
            }
            return icon_map.get(prediction, self._attr_icon)
        return self._attr_icon