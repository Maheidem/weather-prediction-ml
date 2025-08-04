# Weather Prediction ML Custom Component

A Home Assistant custom component that provides machine learning-based weather predictions using ensemble models (XGBoost + Random Forest) with 82.2% accuracy.

## Features

- **Machine Learning Predictions**: Uses ensemble model combining XGBoost and Random Forest
- **Multiple Sensors**: Main prediction, trend indicator, confidence level, and individual probabilities
- **UI Configuration**: Easy setup through Home Assistant UI
- **Real-time Updates**: Configurable update intervals
- **High Accuracy**: 82.2% prediction accuracy for temperature trends

## Installation

### Manual Installation

1. Copy the `weather_prediction_ml` folder to your Home Assistant's `custom_components` directory:
   ```bash
   cp -r custom_components/weather_prediction_ml /config/custom_components/
   ```

2. Restart Home Assistant

3. Go to Settings → Devices & Services → Add Integration → Search for "Weather Prediction ML"

### HACS Installation (Coming Soon)

This component will be available through HACS in the future.

## Configuration

### UI Configuration (Recommended)

1. Navigate to Settings → Devices & Services
2. Click "Add Integration"
3. Search for "Weather Prediction ML"
4. Select your sensor entities:
   - Temperature sensor
   - Humidity sensor
   - Pressure sensor
5. Configure update interval (default: 3600 seconds)
6. Optionally enable automatic model retraining

### YAML Configuration (Legacy)

```yaml
weather_prediction_ml:
  temperature_sensor: sensor.outdoor_temperature
  humidity_sensor: sensor.outdoor_humidity
  pressure_sensor: sensor.outdoor_pressure
  update_interval: 3600
  auto_retrain: false
  retrain_schedule: "0 2 * * 0"  # Weekly at 2 AM Sunday
```

## Sensors

The integration provides the following sensors:

### Main Prediction Sensor
- **Entity**: `sensor.weather_prediction_ml`
- **State**: Current prediction (increase/decrease/stable)
- **Attributes**:
  - `confidence`: Prediction confidence percentage
  - `probabilities`: All class probabilities
  - `last_update`: Timestamp of last prediction
  - `model_type`: "Ensemble (XGBoost + Random Forest)"
  - `model_accuracy`: 82.2%

### Trend Sensor
- **Entity**: `sensor.weather_prediction_trend`
- **State**: Simplified trend (↑/↓/→)
- **Icon**: Dynamic based on trend
- **Attributes**:
  - `description`: Human-readable trend description
  - `expected_change`: Temperature change expectation

### Confidence Sensor
- **Entity**: `sensor.weather_prediction_confidence`
- **State**: Confidence percentage
- **Unit**: %

### Probability Sensors
- `sensor.weather_prediction_increase_probability`
- `sensor.weather_prediction_decrease_probability`
- `sensor.weather_prediction_stable_probability`

## Services

### weather_prediction_ml.predict
Manually trigger a weather prediction update.

```yaml
service: weather_prediction_ml.predict
```

### weather_prediction_ml.retrain_model
Retrain the model with recent data.

```yaml
service: weather_prediction_ml.retrain_model
data:
  days_back: 30  # Optional, default: 30
```

### weather_prediction_ml.get_diagnostics
Get diagnostic information about the model.

```yaml
service: weather_prediction_ml.get_diagnostics
```

## Example Automations

### Alert on High Confidence Temperature Increase
```yaml
automation:
  - alias: "Alert Temperature Increase"
    trigger:
      - platform: state
        entity_id: sensor.weather_prediction_ml
        to: "increase"
    condition:
      - condition: numeric_state
        entity_id: sensor.weather_prediction_confidence
        above: 80
    action:
      - service: notify.mobile_app
        data:
          title: "Temperature Alert"
          message: "Temperature expected to increase with {{ states('sensor.weather_prediction_confidence') }}% confidence"
```

### Dashboard Card Example
```yaml
type: entities
title: Weather Prediction
entities:
  - entity: sensor.weather_prediction_ml
    name: Prediction
  - entity: sensor.weather_prediction_trend
    name: Trend
  - entity: sensor.weather_prediction_confidence
    name: Confidence
  - type: divider
  - entity: sensor.weather_prediction_increase_probability
    name: Increase Probability
  - entity: sensor.weather_prediction_decrease_probability
    name: Decrease Probability
  - entity: sensor.weather_prediction_stable_probability
    name: Stable Probability
```

## Requirements

- Home Assistant 2023.1.0 or newer
- Python 3.9+
- At least 48 hours of historical sensor data
- Sensors for temperature, humidity, and pressure

## Model Information

- **Type**: Ensemble (XGBoost 50.9% + Random Forest 49.1%)
- **Accuracy**: 82.2% (146.9% improvement over baseline)
- **Features**: 119 engineered features including:
  - Temporal features (cyclical encoding)
  - Lag features (1-48 hours)
  - Rolling statistics (6-48 hour windows)
  - Change and trend features
  - Interaction features
  - Derived features (dewpoint, heat index)

## Troubleshooting

### Insufficient Data Error
The component requires at least 48 hours of historical data. Ensure your sensors have been running for at least 2 days.

### Model Loading Error
Check Home Assistant logs for specific error messages. Ensure all model files are present in the `models` directory.

### Prediction Not Updating
1. Check the update interval in configuration
2. Manually trigger prediction using the service
3. Check logs for any errors

## Development

### Project Structure
```
custom_components/weather_prediction_ml/
├── __init__.py          # Integration setup
├── manifest.json        # Component metadata
├── const.py            # Constants
├── sensor.py           # Sensor implementation
├── config_flow.py      # UI configuration
├── services.yaml       # Service definitions
├── strings.json        # UI translations
└── models/             # ML models
    ├── final_xgboost_model.pkl
    ├── final_rf_model.pkl
    ├── final_scaler.pkl
    ├── final_label_encoder.pkl
    └── final_model_config.json
```

## License

This project is licensed under the MIT License.

## Support

For issues and feature requests, please open an issue on GitHub.