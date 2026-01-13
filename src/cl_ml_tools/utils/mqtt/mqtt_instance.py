from typing import TypedDict

from loguru import logger

from .mqtt_impl import MQTTBroadcaster, NoOpBroadcaster


class BroadcasterConfig(TypedDict):
    """Configuration for broadcaster instance."""

    broadcast_type: str
    broker: str | None
    port: int | None


_broadcaster: MQTTBroadcaster | NoOpBroadcaster | None = None
_broadcaster_config: BroadcasterConfig | None = None


def get_broadcaster(
    broadcast_type: str, broker: str | None = None, port: int | None = None
) -> MQTTBroadcaster | NoOpBroadcaster | None:
    """Get or create global broadcaster instance based on config."""
    global _broadcaster, _broadcaster_config

    desired_config: BroadcasterConfig = {
        "broadcast_type": broadcast_type,
        "broker": broker,
        "port": port,
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
        if broadcast_type == "mqtt":
            _broadcaster = MQTTBroadcaster(broker, port)
        else:
            _broadcaster = NoOpBroadcaster(broker, port)
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
