"""Advanced unit tests for MQTT broadcaster.

Targets error handling, connection failures, and edge cases in MQTTBroadcaster and NoOpBroadcaster.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from paho.mqtt.client import MQTTMessage

from cl_ml_tools.utils.mqtt.mqtt_impl import BroadcasterBase, MQTTBroadcaster, NoOpBroadcaster

# ============================================================================
# BroadcasterBase Tests
# ============================================================================


class ConcreteBroadcaster(BroadcasterBase):
    """Concrete subclass for testing BroadcasterBase defaults."""


def test_broadcaster_base_defaults():
    """Test default implementations in BroadcasterBase."""
    base = ConcreteBroadcaster()
    assert base.connect() is False
    assert base.publish_event(topic="t", payload="p") is False
    assert base.set_will(topic="t", payload="p") is False
    assert base.publish_retained(topic="t", payload="p") is False
    assert base.clear_retained("topic") is False
    assert base.subscribe(topic="t", callback=lambda x, y: None) is None
    assert base.unsubscribe("id") is False
    base.disconnect()  # Should not raise


# ============================================================================
# MQTTBroadcaster Tests
# ============================================================================


def test_mqtt_broadcaster_init_error():
    """Test MQTTBroadcaster raises exception on missing config."""
    with pytest.raises(ValueError, match="MQTT URL cannot be None"):
        MQTTBroadcaster(mqtt_url=None)


def test_mqtt_broadcaster_connect_exception(mqtt_url: str):
    """Test connect handles client instantiation/loop errors."""
    broadcaster = MQTTBroadcaster(mqtt_url=mqtt_url)
    with patch("paho.mqtt.client.Client", side_effect=Exception("oops")):
        assert broadcaster.connect() is False
        assert broadcaster.connected is False


def test_mqtt_broadcaster_publish_errors(mqtt_url: str):
    """Test publish methods handle disconnection and exceptions."""
    broadcaster = MQTTBroadcaster(mqtt_url=mqtt_url)

    # 1. Not connected
    assert broadcaster.publish_event(topic="t", payload="p") is False
    assert broadcaster.publish_retained(topic="t", payload="p") is False

    # 2. Connected but publish fails
    broadcaster.connected = True
    broadcaster.client = MagicMock()

    mock_info = MagicMock()
    mock_info.rc = 1  # Error
    broadcaster.client.publish.return_value = mock_info
    assert broadcaster.publish_event(topic="t", payload="p") is False

    # 3. Exception during publish
    broadcaster.client.publish.side_effect = Exception("fail")
    assert broadcaster.publish_event(topic="t", payload="p") is False
    assert broadcaster.publish_retained(topic="t", payload="p") is False


def test_mqtt_broadcaster_set_will_errors(mqtt_url: str):
    """Test set_will handle missing client or exceptions."""
    broadcaster = MQTTBroadcaster(mqtt_url=mqtt_url)
    assert broadcaster.set_will(topic="t", payload="p") is False

    broadcaster.client = MagicMock()
    broadcaster.client.will_set.side_effect = Exception("fail")
    assert broadcaster.set_will(topic="t", payload="p") is False


def test_mqtt_broadcaster_subscribe_errors(mqtt_url: str):
    """Test subscribe failure modes."""
    broadcaster = MQTTBroadcaster(mqtt_url=mqtt_url)

    # Not connected
    assert broadcaster.subscribe(topic="t", callback=lambda x, y: None) is None

    broadcaster.connected = True
    broadcaster.client = MagicMock()

    # MQTT library returns error
    broadcaster.client.subscribe.return_value = (1, 1)  # (rc, mid)
    assert broadcaster.subscribe(topic="t", callback=lambda x, y: None) is None

    # Connection established but subscription raises exception
    broadcaster.client.subscribe.side_effect = Exception("error")
    assert broadcaster.subscribe(topic="t", callback=lambda x, y: None) is None


def test_mqtt_broadcaster_unsubscribe_errors(mqtt_url: str):
    """Test unsubscribe failure modes."""
    broadcaster = MQTTBroadcaster(mqtt_url=mqtt_url)
    assert broadcaster.unsubscribe("some-id") is False

    broadcaster.connected = True
    broadcaster.client = MagicMock()

    # ID not found
    assert broadcaster.unsubscribe("missing-id") is False

    # Subscribed to topic
    broadcaster.subscriptions["test-id"] = ("topic", lambda x, y: None)

    # MQTT library returns error
    broadcaster.client.unsubscribe.return_value = (1, 1)
    assert broadcaster.unsubscribe("test-id") is False

    # Exception
    broadcaster.subscriptions["test-id"] = ("topic", lambda x, y: None)
    broadcaster.client.unsubscribe.side_effect = Exception("fail")
    assert broadcaster.unsubscribe("test-id") is False


def test_mqtt_broadcaster_on_connect_fail(mqtt_url: str):
    """Test _on_connect with non-zero reason code."""
    broadcaster = MQTTBroadcaster(mqtt_url=mqtt_url)
    broadcaster._on_connect(None, None, None, 5, None)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert broadcaster.connected is False


def test_mqtt_broadcaster_on_message_callback_error(mqtt_url: str):
    """Test callback failures don't crash the message loop."""
    broadcaster = MQTTBroadcaster(mqtt_url=mqtt_url)

    def failing_callback(t: str, p: Any):  # pyright: ignore[reportUnusedParameter]
        raise Exception("callback explosion")

    broadcaster.subscriptions["sub1"] = ("test/topic", failing_callback)

    msg = MQTTMessage()
    msg.topic = b"test/topic"
    msg.payload = b"test payload"

    # Should log error but not raise
    broadcaster._on_message(None, None, msg)  # pyright: ignore[reportPrivateUsage, reportArgumentType]


def test_mqtt_broadcaster_topic_matches_edge_cases(mqtt_url: str):
    """Test _topic_matches with complex patterns."""
    broadcaster = MQTTBroadcaster(mqtt_url=mqtt_url)

    # Hash wildcard
    assert broadcaster._topic_matches("home/#", "home/livingroom/temp") is True
    assert broadcaster._topic_matches("home/#", "work/desk") is False

    # Plus wildcard
    assert broadcaster._topic_matches("home/+/temp", "home/kitchen/temp") is True
    assert broadcaster._topic_matches("home/+/temp", "home/kitchen/humidity") is False
    assert (
        broadcaster._topic_matches("home/+/temp", "home/living/dining/temp") is False
    )  # Multi level

    # No wildcard mismatch
    assert broadcaster._topic_matches("fixed/topic", "different/topic") is False  # pyright: ignore[reportPrivateUsage]


# ============================================================================
# NoOpBroadcaster Tests
# ============================================================================


def test_noop_broadcaster_more():
    """Test NoOpBroadcaster failure returns."""
    broadcaster = NoOpBroadcaster()
    assert broadcaster.subscribe(topic="t", callback=lambda x, y: None) is None
    assert broadcaster.unsubscribe("any-id") is False
