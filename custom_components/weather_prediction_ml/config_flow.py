"""Config flow for Weather Prediction ML integration."""
import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import config_validation as cv, entity_registry as er
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
)

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
)

_LOGGER = logging.getLogger(__name__)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect."""
    errors = {}
    
    # Check if sensors exist
    entity_registry = er.async_get(hass)
    
    for sensor_key, sensor_name in [
        (CONF_TEMPERATURE_SENSOR, "Temperature"),
        (CONF_HUMIDITY_SENSOR, "Humidity"),
        (CONF_PRESSURE_SENSOR, "Pressure"),
    ]:
        sensor_id = data.get(sensor_key)
        if sensor_id:
            entity = entity_registry.async_get(sensor_id)
            if not entity:
                # Check if state exists (might not be in registry)
                state = hass.states.get(sensor_id)
                if not state:
                    errors[sensor_key] = f"{sensor_name} sensor not found"
    
    if errors:
        raise ValueError(errors)
    
    # Return info that you want to store in the config entry
    return {"title": "Weather Prediction ML"}


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Weather Prediction ML."""
    
    VERSION = 1
    
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors = {}
        
        if user_input is not None:
            try:
                info = await validate_input(self.hass, user_input)
                
                # Create unique ID based on sensor combination
                unique_id = f"{user_input[CONF_TEMPERATURE_SENSOR]}_{user_input[CONF_HUMIDITY_SENSOR]}_{user_input[CONF_PRESSURE_SENSOR]}"
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()
                
                return self.async_create_entry(title=info["title"], data=user_input)
            except ValueError as err:
                errors = err.args[0]
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
        
        # Show form
        data_schema = vol.Schema(
            {
                vol.Required(CONF_TEMPERATURE_SENSOR): EntitySelector(
                    EntitySelectorConfig(domain="sensor", device_class="temperature")
                ),
                vol.Required(CONF_HUMIDITY_SENSOR): EntitySelector(
                    EntitySelectorConfig(domain="sensor", device_class="humidity")
                ),
                vol.Required(CONF_PRESSURE_SENSOR): EntitySelector(
                    EntitySelectorConfig(domain="sensor", device_class="pressure")
                ),
                vol.Optional(
                    CONF_UPDATE_INTERVAL, default=DEFAULT_UPDATE_INTERVAL
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=60,
                        max=86400,
                        step=60,
                        mode=NumberSelectorMode.BOX,
                        unit_of_measurement="seconds"
                    )
                ),
                vol.Optional(
                    CONF_AUTO_RETRAIN, default=DEFAULT_AUTO_RETRAIN
                ): bool,
                vol.Optional(
                    CONF_RETRAIN_SCHEDULE, default=DEFAULT_RETRAIN_SCHEDULE
                ): str,
            }
        )
        
        return self.async_show_form(
            step_id="user", data_schema=data_schema, errors=errors
        )
    
    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Get the options flow for this handler."""
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for Weather Prediction ML."""
    
    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
    
    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        
        options = self.config_entry.options
        data = self.config_entry.data
        
        # Merge data and options
        current_config = {**data, **options}
        
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_UPDATE_INTERVAL,
                        default=current_config.get(
                            CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=60,
                            max=86400,
                            step=60,
                            mode=NumberSelectorMode.BOX,
                            unit_of_measurement="seconds"
                        )
                    ),
                    vol.Optional(
                        CONF_AUTO_RETRAIN,
                        default=current_config.get(
                            CONF_AUTO_RETRAIN, DEFAULT_AUTO_RETRAIN
                        ),
                    ): bool,
                    vol.Optional(
                        CONF_RETRAIN_SCHEDULE,
                        default=current_config.get(
                            CONF_RETRAIN_SCHEDULE, DEFAULT_RETRAIN_SCHEDULE
                        ),
                    ): str,
                }
            ),
        )