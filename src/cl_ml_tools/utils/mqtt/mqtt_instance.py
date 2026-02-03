from typing import TypedDict

from loguru import logger

from .mqtt_impl import MQTTBroadcaster, NoOpBroadcaster


class BroadcasterConfig(TypedDict):
    """Configuration for broadcaster instance."""

    url: str | None


_broadcaster: MQTTBroadcaster | NoOpBroadcaster | None = None
_broadcaster_config: BroadcasterConfig | None = None


def get_broadcaster(url: str | None = None) -> MQTTBroadcaster | NoOpBroadcaster | None:
    """Get or create global broadcaster instance based on config.

    Args:
        url: MQTT broker URL (e.g., mqtt://<host_ip>:<port>).
             If None, returns NoOpBroadcaster.

    Returns:
        MQTTBroadcaster if url is provided, NoOpBroadcaster if url is None.
    """
    global _broadcaster, _broadcaster_config

    desired_config: BroadcasterConfig = {
        "url": url,
    }

    # Check if existing singleton is compatible
    if _broadcaster is not None and _broadcaster_config == desired_config:
        return _broadcaster

    # Config mismatch — shutdown old broadcaster if needed
    if _broadcaster is not None:
        try:
            _broadcaster.disconnect()
            _broadcaster_config = None
        except Exception:
            pass  # Best effort cleanup

    try:
        # Recreate with new config
        if url is None:
            _broadcaster = NoOpBroadcaster(url)
        else:
            _broadcaster = MQTTBroadcaster(url)
        _ = _broadcaster.connect()

        _broadcaster_config = desired_config
    except Exception as e:
        logger.error(f"Error creating broadcaster: {e}")
    return _broadcaster


def shutdown_broadcaster():
    """Shutdown global broadcaster."""
    global _broadcaster
    if _broadcaster:
        _broadcaster.disconnect()
        _broadcaster = None
