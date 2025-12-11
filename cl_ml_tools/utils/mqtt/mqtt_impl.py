"""MQTT broadcaster for job events and worker capabilities."""

from abc import ABC
import json
import logging
import time
from typing import Callable, Dict, Optional, Protocol, Tuple
from uuid import uuid4
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

    def subscribe(
        self,
        *,
        topic: str,
        callback: Callable[[str, str], None],
        qos: int = 1,
    ) -> Optional[str]:
        return None

    def unsubscribe(self, subscription_id: str) -> bool:
        return False


class MQTTBroadcaster(BroadcasterBase):
    """MQTT event broadcaster using modern MQTT v5 protocol."""

    def __init__(self, broker: Optional[str] = None, port: Optional[int] = None):
        if not broker or not port:
            raise Exception(
                "MQTT broadcaster must be provided with broker and its port"
            )
        self.broker = broker
        self.port = port
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self._subscriptions: Dict[str, Tuple[str, Callable[[str, str], None]]] = {}

    def connect(self) -> bool:
        try:
            self.client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                protocol=mqtt.MQTTv5,
            )

            # Assign v5 callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message

            # Optional: robust reconnection
            self.client.reconnect_delay_set(min_delay=1, max_delay=30)

            self.client.loop_start()
            self.client.connect(self.broker, self.port, keepalive=60)

            # Wait for connection to be established (up to 5 seconds)
            timeout = 5
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            return self.connected
        except Exception as e:
            logger.warning(
                f"Failed to connect to MQTT broker:{self.broker}:{self.port} {e}"
            )
            self.connected = False
            return False

    def disconnect(self):
        self._subscriptions.clear()
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

    def subscribe(
        self, *, topic: str, callback: Callable[[str, str], None], qos: int = 1
    ) -> Optional[str]:
        """Subscribe to MQTT topic and register callback.

        Args:
            topic: MQTT topic to subscribe to (supports wildcards: +, #)
            callback: Function called when message received (topic, payload)
            qos: Quality of service level (0, 1, or 2)

        Returns:
            Subscription ID (string) if successful, None otherwise
            Use this ID with unsubscribe() to remove this specific subscription

        Note:
            - Callback is invoked from MQTT client thread. Ensure thread-safety.
            - Multiple subscriptions to the same topic are supported.
            - Each subscription gets a unique ID and independent callback.
        """
        if not self.connected or not self.client:
            return None

        try:
            # Check if we already have a subscription to this exact topic
            # If not, subscribe at MQTT level
            already_subscribed = any(t == topic for t, _ in self._subscriptions.values())

            if not already_subscribed:
                result, mid = self.client.subscribe(topic, qos=qos)
                if result != mqtt.MQTT_ERR_SUCCESS:
                    logger.error(f"Failed to subscribe to {topic}: error code {result}")
                    return None

            # Generate unique subscription ID and store callback
            subscription_id = str(uuid4())
            self._subscriptions[subscription_id] = (topic, callback)
            logger.info(f"Subscribed to topic: {topic} (subscription_id: {subscription_id})")
            return subscription_id

        except Exception as e:
            logger.error(f"Error subscribing to topic {topic}: {e}")
            return None

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from MQTT topic using subscription ID.

        Args:
            subscription_id: The unique subscription ID returned by subscribe()

        Returns:
            True if unsubscription successful, False otherwise
        """
        if not self.connected or not self.client:
            return False

        try:
            # Get subscription info
            if subscription_id not in self._subscriptions:
                logger.warning(f"Subscription ID not found: {subscription_id}")
                return False

            topic, _ = self._subscriptions[subscription_id]

            # Remove from our tracking
            del self._subscriptions[subscription_id]

            # Check if any other subscriptions exist for this topic
            still_subscribed = any(t == topic for t, _ in self._subscriptions.values())

            # Only unsubscribe at MQTT level if no more subscriptions to this topic
            if not still_subscribed:
                result, mid = self.client.unsubscribe(topic)
                if result != mqtt.MQTT_ERR_SUCCESS:
                    logger.error(f"Failed to unsubscribe from {topic}: error code {result}")
                    return False

            logger.info(f"Unsubscribed: {subscription_id} from topic: {topic}")
            return True

        except Exception as e:
            logger.error(f"Error unsubscribing {subscription_id}: {e}")
            return False

    #
    # MQTT v5 Callback APIs
    #
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self.connected = reason_code == 0
        if self.connected:
            logger.info("MQTT connected using v5")
        else:
            logger.warning(f"MQTT connection failed: reason={reason_code}, props={properties}")

    def _on_disconnect(
        self, client, userdata, disconnect_flags, reason_code, properties
    ):
        self.connected = False
        logger.warning(f"MQTT disconnected: {reason_code}")

    def _on_message(self, client, userdata, message):
        """Handle incoming MQTT messages (VERSION2 callback)."""
        try:
            received_topic = message.topic
            payload = message.payload.decode("utf-8")

            # Invoke all callbacks for matching subscriptions
            for subscription_id, (subscribed_topic, callback) in self._subscriptions.items():
                # paho-mqtt handles wildcard matching internally, so we get messages
                # only for topics we're subscribed to. Just check if topics match.
                if received_topic == subscribed_topic or self._topic_matches(
                    subscribed_topic, received_topic
                ):
                    try:
                        callback(received_topic, payload)
                    except Exception as callback_error:
                        logger.error(
                            f"Error in callback for subscription {subscription_id}: {callback_error}"
                        )
        except Exception as e:
            logger.error(f"Error in message callback for topic {message.topic}: {e}")

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if topic matches subscription pattern (handles wildcards).

        MQTT wildcards:
        - '+' matches single level
        - '#' matches multiple levels (must be last character)
        """
        # This is a simplified check - paho-mqtt already does filtering,
        # but we need this for multiple subscriptions to same/overlapping topics
        if pattern == topic:
            return True

        # Simple wildcard matching
        if "#" in pattern:
            prefix = pattern.rstrip("#")
            return topic.startswith(prefix)

        if "+" not in pattern:
            return False

        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        if len(pattern_parts) != len(topic_parts):
            return False

        for p, t in zip(pattern_parts, topic_parts):
            if p != "+" and p != t:
                return False

        return True


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

    def subscribe(
        self,
        *,
        topic: str,
        callback: Callable[[str, str], None],
        qos: int = 1,
    ) -> Optional[str]:
        """NoOp subscribe - accepts but returns None to indicate failure."""
        return None

    def unsubscribe(self, subscription_id: str) -> bool:
        """NoOp unsubscribe - accepts but returns False to indicate failure."""
        return False
