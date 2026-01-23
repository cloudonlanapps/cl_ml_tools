"""MQTT broadcaster for job events and worker capabilities."""

import time
from typing import Callable, Protocol, override
from uuid import uuid4

import paho.mqtt.client as mqtt
from loguru import logger
from paho.mqtt.client import ConnectFlags, DisconnectFlags, MQTTMessage
from paho.mqtt.enums import CallbackAPIVersion
from paho.mqtt.properties import Properties
from paho.mqtt.reasoncodes import ReasonCode


# NoOpBroadcaster must not require configuration. So protocol
# should not enforce broker nad port mandatory
class BroadcasterBase(Protocol):
    connected: bool
    client: mqtt.Client | None = None

    def __init__(self, broker: str | None = None, port: int | None = None):
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
    ) -> str | None:
        return None

    def unsubscribe(self, subscription_id: str) -> bool:
        return False


class MQTTBroadcaster(BroadcasterBase):
    """MQTT event broadcaster using modern MQTT v5 protocol."""

    def __init__(self, broker: str | None = None, port: int | None = None):
        super().__init__(broker, port)
        if not broker or not port:
            raise Exception(
                "MQTT broadcaster must be provided with broker and its port"
            )
        self.broker: str = broker
        self.port: int = port
        self.client: mqtt.Client | None = None
        self.connected: bool = False
        self.subscriptions: dict[str, tuple[str, Callable[[str, str], None]]] = {}

    @override
    def connect(self) -> bool:
        try:
            logger.info(
                f"Assuming MQTT client configuration: broker={self.broker}:{self.port}"
            )
            self.client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                protocol=mqtt.MQTTv5,
            )

            # Assign v5 callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message

            # Optional: robust reconnection
            _ = self.client.reconnect_delay_set(min_delay=1, max_delay=30)

            _ = self.client.loop_start()
            _ = self.client.connect(
                self.broker, self.port, keepalive=60, clean_start=True
            )

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

    @override
    def disconnect(self):
        self.subscriptions.clear()
        if self.client:
            _ = self.client.loop_stop()
            _ = self.client.disconnect()
            self.connected = False

    @override
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

    @override
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

    @override
    def publish_retained(self, *, topic: str, payload: str, qos: int = 1) -> bool:
        if not self.connected or not self.client:
            return False
        try:
            result = self.client.publish(topic, payload, qos=qos, retain=True)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"Error publishing retained message: {e}")
            return False

    @override
    def clear_retained(self, topic: str, qos: int = 1) -> bool:
        """Clear a retained MQTT message by publishing empty payload."""
        return self.publish_retained(topic=topic, payload="", qos=qos)

    @override
    def subscribe(
        self, *, topic: str, callback: Callable[[str, str], None], qos: int = 1
    ) -> str | None:
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
            already_subscribed = any(t == topic for t, _ in self.subscriptions.values())

            if not already_subscribed:
                result, _mid = self.client.subscribe(topic, qos=qos)
                if result != mqtt.MQTT_ERR_SUCCESS:
                    logger.error(f"Failed to subscribe to {topic}: error code {result}")
                    return None

            # Generate unique subscription ID and store callback
            subscription_id = str(uuid4())
            self.subscriptions[subscription_id] = (topic, callback)
            logger.info(
                f"Subscribed to topic: {topic} (subscription_id: {subscription_id})"
            )
            return subscription_id

        except Exception as e:
            logger.error(f"Error subscribing to topic {topic}: {e}")
            return None

    @override
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
            if subscription_id not in self.subscriptions:
                logger.warning(f"Subscription ID not found: {subscription_id}")
                return False

            topic, _ = self.subscriptions[subscription_id]

            # Remove from our tracking
            del self.subscriptions[subscription_id]

            # Check if any other subscriptions exist for this topic
            still_subscribed = any(t == topic for t, _ in self.subscriptions.values())

            # Only unsubscribe at MQTT level if no more subscriptions to this topic
            if not still_subscribed:
                result, _mid = self.client.unsubscribe(topic)
                if result != mqtt.MQTT_ERR_SUCCESS:
                    logger.error(
                        f"Failed to unsubscribe from {topic}: error code {result}"
                    )
                    return False

            logger.info(f"Unsubscribed: {subscription_id} from topic: {topic}")
            return True

        except Exception as e:
            logger.error(f"Error unsubscribing {subscription_id}: {e}")
            return False

    #
    # MQTT v5 Callback APIs
    #
    def _on_connect(
        self,
        _client: mqtt.Client,
        _userdata: object,
        _flags: ConnectFlags,
        reason_code: ReasonCode,
        properties: Properties | None,
    ) -> None:
        self.connected = reason_code == 0
        if self.connected:
            logger.info("MQTT connected using v5")
        else:
            logger.warning(
                f"MQTT connection failed: reason={reason_code}, props={properties}"
            )

    def _on_disconnect(
        self,
        _client: mqtt.Client,
        _userdata: object,
        _disconnect_flags: DisconnectFlags,
        reason_code: ReasonCode,
        _properties: Properties | None,
    ) -> None:
        self.connected = False
        logger.warning(f"MQTT disconnected: {reason_code}")

    def _on_message(
        self, _client: mqtt.Client, _userdata: object, message: MQTTMessage
    ) -> None:
        """Handle incoming MQTT messages (VERSION2 callback)."""
        try:
            received_topic = message.topic
            payload = message.payload.decode("utf-8")

            # Invoke all callbacks for matching subscriptions
            for subscription_id, (
                subscribed_topic,
                callback,
            ) in self.subscriptions.items():
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
    client: None = None

    def __init__(self, broker: str | None = None, port: int | None = None):
        super().__init__(broker, port)
        self.connected = True

    @override
    def connect(self) -> bool:
        return True

    @override
    def disconnect(self):
        pass

    @override
    def publish_event(
        self,
        *,
        topic: str,
        payload: str,
        qos: int = 1,
    ) -> bool:
        return True

    @override
    def set_will(
        self, *, topic: str, payload: str, qos: int = 1, retain: bool = True
    ) -> bool:
        return True

    @override
    def publish_retained(self, *, topic: str, payload: str, qos: int = 1) -> bool:
        return True

    @override
    def clear_retained(self, topic: str, qos: int = 1) -> bool:
        return True

    @override
    def subscribe(
        self,
        *,
        topic: str,
        callback: Callable[[str, str], None],
        qos: int = 1,
    ) -> str | None:
        """NoOp subscribe - accepts but returns None to indicate failure."""
        return None

    @override
    def unsubscribe(self, subscription_id: str) -> bool:
        """NoOp unsubscribe - accepts but returns False to indicate failure."""
        return False
