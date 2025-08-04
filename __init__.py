"""The Weather Prediction ML integration."""
import logging
from datetime import timedelta
from pathlib import Path

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .const import (
    DOMAIN,
    CONF_TEMPERATURE_SENSOR,
    CONF_HUMIDITY_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_UPDATE_INTERVAL,
    CONF_AUTO_RETRAIN,
    CONF_RETRAIN_SCHEDULE,
    DEFAULT_UPDATE_INTERVAL,
    DEFAULT_AUTO_RETRAIN,
    DEFAULT_RETRAIN_SCHEDULE,
    SERVICE_PREDICT,
    SERVICE_RETRAIN,
    SERVICE_GET_DIAGNOSTICS,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR]

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {
                vol.Required(CONF_TEMPERATURE_SENSOR): cv.entity_id,
                vol.Required(CONF_HUMIDITY_SENSOR): cv.entity_id,
                vol.Required(CONF_PRESSURE_SENSOR): cv.entity_id,
                vol.Optional(
                    CONF_UPDATE_INTERVAL, default=DEFAULT_UPDATE_INTERVAL
                ): cv.positive_int,
                vol.Optional(
                    CONF_AUTO_RETRAIN, default=DEFAULT_AUTO_RETRAIN
                ): cv.boolean,
                vol.Optional(
                    CONF_RETRAIN_SCHEDULE, default=DEFAULT_RETRAIN_SCHEDULE
                ): cv.string,
            }
        )
    },
    extra=vol.ALLOW_EXTRA,
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Weather Prediction ML integration."""
    hass.data.setdefault(DOMAIN, {})
    
    # Register services
    async def handle_predict(call: ServiceCall) -> None:
        """Handle the predict service call."""
        _LOGGER.info("Manual prediction triggered")
        # Trigger prediction on all configured instances
        for entry_id in hass.data[DOMAIN]:
            if coordinator := hass.data[DOMAIN][entry_id].get("coordinator"):
                await coordinator.async_request_refresh()
    
    async def handle_retrain(call: ServiceCall) -> None:
        """Handle the retrain model service call."""
        days_back = call.data.get("days_back", 30)
        _LOGGER.info(f"Model retraining triggered with {days_back} days of data")
        # TODO: Implement model retraining logic
    
    async def handle_get_diagnostics(call: ServiceCall) -> None:
        """Handle the get diagnostics service call."""
        _LOGGER.info("Diagnostics requested")
        # TODO: Implement diagnostics logic
    
    hass.services.async_register(DOMAIN, SERVICE_PREDICT, handle_predict)
    hass.services.async_register(
        DOMAIN,
        SERVICE_RETRAIN,
        handle_retrain,
        schema=vol.Schema({
            vol.Optional("days_back", default=30): cv.positive_int,
        }),
    )
    hass.services.async_register(DOMAIN, SERVICE_GET_DIAGNOSTICS, handle_get_diagnostics)
    
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Weather Prediction ML from a config entry."""
    hass.data[DOMAIN][entry.entry_id] = {}
    
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
    
    return unload_ok