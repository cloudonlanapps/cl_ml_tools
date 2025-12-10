"""MQTT broadcaster for job events and worker capabilities."""

from abc import ABC
import json
import logging
import time
from typing import Optional, Protocol
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

logger = logging.getLogger(__name__)


# NoOpBroadcaster must not require configuration. So protocol
# should not enforce broker nad port mandatory
class BroadcasterBase(Protocol):
    connected: bool

    def __init__(self, broker: Optional[str] = None, port: Optional[int] = None):
        self.connected = False

    def connect(self) -> bool:
        return False

    def disconnect(self):
        pass

    def publish_event(self, *, topic: str, payload: str, qos: int = 1) -> bool:
        return False

    def set_will(
        self, *, topic: str, payload: str, qos: int = 1, retain: bool = False
    ) -> bool:
        return False

    def publish_retained(self, *, topic: str, payload: str, qos: int = 1) -> bool:
        return False

    def clear_retained(self, topic: str, qos: int = 1) -> bool:
        return False


class MQTTBroadcaster(BroadcasterBase):
    """MQTT event broadcaster using modern MQTT v5 protocol."""

    def __init__(self, broker: Optional[str] = None, port: Optional[int] = None):
        if not broker or not port:
            raise Exception(
                "MQTT broadcaster must be provided with borker and its port"
            )
        self.broker = broker
        self.port = port
        self.client: Optional[mqtt.Client] = None
        self.connected = False

    def connect(self) -> bool:
        if not self.broker or not self.port:
            logger.warning(
                f"Failed to configure to MQTT broker:{self.broker}:{self.port}"
            )
            return False
        try:
            self.client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                protocol=mqtt.MQTTv5,
            )

            # Assign v5 callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect

            # Optional: robust reconnection
            self.client.reconnect_delay_set(min_delay=1, max_delay=30)

            self.client.connect_async(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.warning(
                f"Failed to connect to MQTT broker:{self.broker}:{self.port} {e}"
            )
            self.connected = False
            return False

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False

    def publish_event(self, *, topic: str, payload: str, qos: int = 1) -> bool:
        if not self.connected or not self.client:
            return False

        try:

            # v5: publish returns MQTTMessageInfo with rc result code
            result = self.client.publish(topic, payload, qos=1, retain=False)
            print(f"result is : {result.rc} :: {result.rc == mqtt.MQTT_ERR_SUCCESS}")
            logger.error(
                f"result is : {result.rc} :: {result.rc == mqtt.MQTT_ERR_SUCCESS}"
            )
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return False

    def set_will(
        self, *, topic: str, payload: str, qos: int = 1, retain: bool = True
    ) -> bool:
        """Set MQTT Last Will and Testament message."""
        if not self.client:
            return False

        try:
            # Works the same for MQTT v5
            self.client.will_set(topic, payload, qos=qos, retain=retain)
            return True
        except Exception as e:
            logger.error(f"Error setting LWT: {e}")
            return False

    def publish_retained(self, *, topic: str, payload: str, qos: int = 1) -> bool:
        if not self.connected or not self.client:
            return False
        try:
            result = self.client.publish(topic, payload, qos=qos, retain=True)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"Error publishing retained message: {e}")
            return False

    def clear_retained(self, topic: str, qos: int = 1) -> bool:
        """Clear a retained MQTT message by publishing empty payload."""
        return self.publish_retained(topic=topic, payload="", qos=qos)

    #
    # MQTT v5 Callback APIs
    #
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self.connected = reason_code == 0
        if self.connected:
            logger.info("MQTT connected using v5")
        else:
            logger.error(f"MQTT failed: reason={reason_code}, props={properties}")

    def _on_disconnect(
        self, client, userdata, disconnect_flags, reason_code, properties
    ):
        self.connected = False
        logger.warning(f"MQTT disconnected: {reason_code}")


class NoOpBroadcaster(BroadcasterBase):
    """No-operation broadcaster for when MQTT is disabled or unavailable."""

    connected: bool

    def __init__(self, broker: Optional[str] = None, port: Optional[int] = None):
        self.connected = True

    def connect(self) -> bool:
        return True

    def disconnect(self):
        pass

    def publish_event(
        self,
        *,
        topic: str,
        payload: str,
        qos: int = 1,
    ) -> bool:
        return True

    def set_will(
        self, *, topic: str, payload: str, qos: int = 1, retain: bool = True
    ) -> bool:
        return True

    def publish_retained(self, *, topic: str, payload: str, qos: int = 1) -> bool:
        return True

    def clear_retained(self, topic: str, qos: int = 1) -> bool:
        return True
