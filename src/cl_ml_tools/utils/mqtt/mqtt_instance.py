from loguru import logger

from .mqtt_impl import MQTTBroadcaster, NoOpBroadcaster


_broadcaster_instance: MQTTBroadcaster | NoOpBroadcaster | None = None
_current_config: tuple[str | None] | None = None


def get_broadcaster(
    mqtt_url: str | None = None,
) -> MQTTBroadcaster | NoOpBroadcaster:
    """Get or create broadcaster singleton.

    NULL SEMANTICS:
    - mqtt_url=None → Returns NoOpBroadcaster
    - mqtt_url="" → Raises ValueError
    - mqtt_url="mqtt://host:port" → Returns MQTTBroadcaster

    Args:
        mqtt_url: MQTT URL or None to disable MQTT

    Returns:
        Broadcaster instance (singleton)

    Raises:
        ValueError: If mqtt_url is provided but invalid
    """
    global _broadcaster_instance, _current_config

    # Determine broadcast type from url
    broadcast_type = "noop" if mqtt_url is None else "mqtt"

    # Create new config tuple for comparison
    new_config = (mqtt_url,)

    # Check if reconfiguration needed
    if _broadcaster_instance is not None and _current_config != new_config:
        logger.info("Broadcaster config changed, disconnecting old instance")
        if hasattr(_broadcaster_instance, "disconnect"):
            _broadcaster_instance.disconnect()
        _broadcaster_instance = None

    # Create broadcaster if needed
    if _broadcaster_instance is None:
        _current_config = new_config

        if broadcast_type == "mqtt":
            # MQTTBroadcaster will validate the URL
            _broadcaster_instance = MQTTBroadcaster(mqtt_url=mqtt_url)
            _broadcaster_instance.connect()
        else:
            _broadcaster_instance = NoOpBroadcaster()

    return _broadcaster_instance


def shutdown_broadcaster() -> None:
    """Shutdown and cleanup broadcaster singleton."""
    global _broadcaster_instance, _current_config

    if _broadcaster_instance is not None:
        if hasattr(_broadcaster_instance, "disconnect"):
            _broadcaster_instance.disconnect()
        _broadcaster_instance = None
        _current_config = None
