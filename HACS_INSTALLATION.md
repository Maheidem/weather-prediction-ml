# Installing Weather Prediction ML via HACS

Currently, this component is **not yet available** in the default HACS repository. Here are your options:

## Option 1: Add as Custom Repository (Recommended for Testing)

1. **Open HACS** in Home Assistant
   - Navigate to HACS in your sidebar

2. **Add Custom Repository**
   - Click the 3 dots menu (top right)
   - Select "Custom repositories"
   - Add:
     - Repository: `https://github.com/maheidem/weather-prediction-ml` (or your repo URL)
     - Category: `Integration`
   - Click "Add"

3. **Install the Integration**
   - Go to "Integrations" tab in HACS
   - Search for "Weather Prediction ML"
   - Click "Download"
   - Restart Home Assistant

4. **Configure**
   - Go to Settings → Devices & Services
   - Click "Add Integration"
   - Search for "Weather Prediction ML"
   - Follow the configuration steps

## Option 2: Manual Installation (Current Method)

Since the component isn't published to a GitHub repository yet, use the manual installation:

```bash
# Copy the deployment package to your HA server
scp weather_prediction_ml_v1.0.0.tar.gz hassio@192.168.31.114:/home/hassio/

# SSH into HA
ssh hassio@192.168.31.114

# Extract and install
cd /home/hassio
tar -xzf weather_prediction_ml_v1.0.0.tar.gz
cd /config
/home/hassio/weather_prediction_ml_deployment/install.sh

# Restart HA
ha core restart
```

## Publishing to HACS (For Repository Owners)

To make this component available via HACS, you need to:

1. **Create GitHub Repository**
   ```
   weather-prediction-ml/
   ├── custom_components/
   │   └── weather_prediction_ml/
   │       ├── __init__.py
   │       ├── manifest.json
   │       ├── sensor.py
   │       ├── config_flow.py
   │       ├── const.py
   │       ├── services.yaml
   │       ├── strings.json
   │       └── models/
   ├── hacs.json
   ├── info.md
   ├── README.md
   └── LICENSE
   ```

2. **Add Required Files**
   - ✅ `hacs.json` - HACS configuration
   - ✅ `info.md` - HACS info page
   - ✅ `README.md` - Documentation
   - ⚠️ `LICENSE` - Required (suggest MIT)

3. **Create GitHub Release**
   - Tag your release (e.g., v1.0.0)
   - HACS will use releases for versioning

4. **Submit to HACS Default Repository** (Optional)
   - Fork: https://github.com/hacs/default
   - Add your repository to `integration` list
   - Create pull request
   - Wait for review and approval

## Why Use HACS?

- **Easy Updates**: Get notifications and one-click updates
- **Version Management**: Rollback to previous versions if needed
- **Community**: Join the HACS ecosystem
- **Discovery**: Users can find your integration easily

## Current Status

This component is ready for HACS but needs to be:
1. Published to a GitHub repository
2. Tagged with a release version
3. Optionally submitted to HACS default repository

For now, please use the manual installation method.