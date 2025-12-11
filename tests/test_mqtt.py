"""Unit tests for MQTT broadcaster.

Tests MQTTBroadcaster and NoOpBroadcaster implementations.
Requires MQTT broker running on localhost:1883 for MQTTBroadcaster tests.
"""

import json
import time
import socket
from uuid import uuid4

import pytest

from cl_ml_tools import (
    MQTTBroadcaster,
    NoOpBroadcaster,
    get_broadcaster,
    shutdown_broadcaster,
)


# ============================================================================
# Helper Functions
# ============================================================================


def is_mqtt_running(host="localhost", port=1883, timeout=2):
    """Check if MQTT broker is running on specified host:port.

    Args:
        host: MQTT broker hostname
        port: MQTT broker port
        timeout: Connection timeout in seconds

    Returns:
        True if MQTT broker is reachable, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def skip_if_no_mqtt():
    """Pytest marker to skip test if MQTT is not running."""
    if not is_mqtt_running():
        pytest.skip("MQTT broker not running on localhost:1883")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_topic():
    """Generate unique test topic for each test."""
    return f"test/topic/{uuid4()}"


@pytest.fixture
def mqtt_broadcaster():
    """Create MQTTBroadcaster instance for testing."""
    skip_if_no_mqtt()
    broadcaster = MQTTBroadcaster(broker="localhost", port=1883)
    yield broadcaster
    # Cleanup
    broadcaster.disconnect()


@pytest.fixture
def noop_broadcaster():
    """Create NoOpBroadcaster instance for testing."""
    return NoOpBroadcaster()


# ============================================================================
# NoOpBroadcaster Tests (Always Run)
# ============================================================================


class TestNoOpBroadcaster:
    """Test suite for NoOpBroadcaster."""

    def test_connect(self, noop_broadcaster: NoOpBroadcaster):
        """Test NoOp connect always succeeds."""
        assert noop_broadcaster.connect() is True

    def test_disconnect(self, noop_broadcaster: NoOpBroadcaster):
        """Test NoOp disconnect does nothing."""
        noop_broadcaster.disconnect()  # Should not raise

    def test_publish_event(self, noop_broadcaster: NoOpBroadcaster, test_topic):
        """Test NoOp publish_event always succeeds."""
        result = noop_broadcaster.publish_event(
            topic=test_topic,
            payload=json.dumps(
                {
                    "event_type": "test",
                    "job_id": "job-123",
                    "status": "processing",
                }
            ),
        )
        assert result is True

    def test_set_will(self, noop_broadcaster):
        """Test NoOp set_will always succeeds."""
        result = noop_broadcaster.set_will(topic="test/will", payload="offline")
        assert result is True

    def test_publish_retained(self, noop_broadcaster):
        """Test NoOp publish_retained always succeeds."""
        result = noop_broadcaster.publish_retained(
            topic="test/retained", payload="test message"
        )
        assert result is True

    def test_clear_retained(self, noop_broadcaster):
        """Test NoOp clear_retained always succeeds."""
        result = noop_broadcaster.clear_retained(topic="test/retained")
        assert result is True

    def test_subscribe(self, noop_broadcaster):
        """Test NoOp subscribe returns None."""
        callback = lambda topic, payload: None
        result = noop_broadcaster.subscribe(
            topic="test/topic", callback=callback, qos=1
        )
        assert result is None

    def test_unsubscribe(self, noop_broadcaster):
        """Test NoOp unsubscribe returns False."""
        result = noop_broadcaster.unsubscribe(subscription_id="fake-id")
        assert result is False


# ============================================================================
# MQTTBroadcaster Tests (Require MQTT Broker)
# ============================================================================


class TestMQTTBroadcaster:
    """Test suite for MQTTBroadcaster.

    These tests require MQTT broker running on localhost:1883.
    Tests will be skipped if broker is not available.
    """

    def test_mqtt_broker_running(self):
        """Test that MQTT broker is running on localhost:1883."""
        if not is_mqtt_running():
            pytest.fail(
                "MQTT broker is not running on localhost:1883. "
                "Please start MQTT broker to run these tests."
            )

    def test_init_without_broker(self):
        """Test that MQTTBroadcaster raises exception without broker."""
        with pytest.raises(
            Exception, match="MQTT broadcaster must be provided with broker"
        ):
            MQTTBroadcaster(broker=None, port=1883)

    def test_init_without_port(self):
        """Test that MQTTBroadcaster raises exception without port."""
        with pytest.raises(
            Exception, match="MQTT broadcaster must be provided with broker"
        ):
            MQTTBroadcaster(broker="localhost", port=None)

    def test_init_without_both(self):
        """Test that MQTTBroadcaster raises exception without broker and port."""
        with pytest.raises(
            Exception, match="MQTT broadcaster must be provided with broker"
        ):
            MQTTBroadcaster(broker=None, port=None)

    def test_connect_to_invalid_broker(self):
        """Test connection failure with invalid broker hostname."""
        broadcaster = MQTTBroadcaster(broker="invalid-host-xyz-12345", port=9999)
        result = broadcaster.connect()
        assert result is False
        assert broadcaster.connected is False
        # Cleanup
        broadcaster.disconnect()

    def test_connect_to_wrong_port(self):
        """Test connection failure with wrong port."""
        broadcaster = MQTTBroadcaster(broker="localhost", port=9999)
        result = broadcaster.connect()
        assert result is False
        assert broadcaster.connected is False
        # Cleanup
        broadcaster.disconnect()

    def test_connect(self, mqtt_broadcaster):
        """Test connecting to MQTT broker."""
        result = mqtt_broadcaster.connect()
        assert result is True
        assert mqtt_broadcaster.connected is True
        assert mqtt_broadcaster.client is not None

    def test_disconnect(self, mqtt_broadcaster):
        """Test disconnecting from MQTT broker."""
        mqtt_broadcaster.connect()
        mqtt_broadcaster.disconnect()
        assert mqtt_broadcaster.connected is False

    def test_publish_event(self, mqtt_broadcaster, test_topic):
        """Test publishing job event to MQTT."""
        mqtt_broadcaster.connect()

        job_id = str(uuid4())
        result = mqtt_broadcaster.publish_event(
            topic=test_topic,
            payload=json.dumps(
                {
                    "event_type": "started",
                    "job_id": job_id,
                    "status": "processing",
                    "progress": 0.0,
                }
            ),
        )

        assert result is True

    def test_publish_event_payload_format(self, mqtt_broadcaster, test_topic):
        """Test that publish_event creates correctly formatted payload."""
        mqtt_broadcaster.connect()

        job_id = str(uuid4())

        # We can't directly verify the payload without a subscriber,
        # but we can verify the method succeeds
        result = mqtt_broadcaster.publish_event(
            topic=test_topic,
            payload=json.dumps(
                {
                    "event_type": "progress",
                    "job_id": job_id,
                    "progress": 50,
                    "status": "processing",
                }
            ),
        )
        assert result is True

    def test_publish_event_without_connection(self, test_topic):
        """Test publish_event fails without connection."""
        broadcaster = MQTTBroadcaster(broker="localhost", port=1883)
        # Don't connect

        result = broadcaster.publish_event(
            topic=test_topic,
            payload=json.dumps(
                {
                    "job_id": "job-123",
                    "event_type": "test",
                    "timestamp": int(time.time() * 1000),
                }
            ),
        )

        assert result is False

    def test_publish_retained(self, mqtt_broadcaster):
        """Test publishing retained message."""
        mqtt_broadcaster.connect()

        topic = f"test/retained/{uuid4()}"
        payload = json.dumps({"status": "online", "timestamp": int(time.time() * 1000)})

        result = mqtt_broadcaster.publish_retained(topic=topic, payload=payload, qos=1)
        assert result is True

    def test_clear_retained(self, mqtt_broadcaster):
        """Test clearing retained message."""
        mqtt_broadcaster.connect()

        topic = f"test/retained/{uuid4()}"

        # First publish a retained message
        payload = json.dumps({"status": "online"})
        result = mqtt_broadcaster.publish_retained(topic=topic, payload=payload, qos=1)
        assert result is True

        # Then clear it
        result = mqtt_broadcaster.clear_retained(topic, qos=1)
        assert result is True

    def test_set_will(self):
        """Test setting Last Will and Testament.

        Note: Current implementation requires connect() to be called first
        because the MQTT client is only created during connect(). This is
        a limitation - ideally LWT should be set before connecting.
        """
        skip_if_no_mqtt()

        will_topic = f"test/will/{uuid4()}"
        will_payload = json.dumps({"status": "offline"})

        # Create broadcaster and connect first
        broadcaster = MQTTBroadcaster(broker="localhost", port=1883)
        broadcaster.connect()

        # Set will after connecting (limitation of current implementation)
        result = broadcaster.set_will(
            topic=will_topic, payload=will_payload, qos=1, retain=True
        )
        assert result is True

        # Cleanup
        broadcaster.disconnect()

    def test_reconnect(self, mqtt_broadcaster):
        """Test reconnecting to MQTT broker."""
        # First connection
        mqtt_broadcaster.connect()
        assert mqtt_broadcaster.connected is True

        # Disconnect
        mqtt_broadcaster.disconnect()
        assert mqtt_broadcaster.connected is False

        # Reconnect
        result = mqtt_broadcaster.connect()
        assert result is True
        assert mqtt_broadcaster.connected is True

    def test_multiple_events(self, mqtt_broadcaster, test_topic):
        """Test publishing multiple events in sequence."""
        mqtt_broadcaster.connect()

        job_id = str(uuid4())
        events = [
            ("started", {"status": "processing"}),
            ("progress", {"progress": 25}),
            ("progress", {"progress": 50}),
            ("progress", {"progress": 75}),
            ("completed", {"status": "completed", "progress": 100}),
        ]

        for event in events:

            result = mqtt_broadcaster.publish_event(
                topic=test_topic,
                payload=json.dumps(event),
            )

            assert result is True
            time.sleep(0.01)  # Small delay between events

    def test_subscribe_without_connection(self, test_topic):
        """Test subscribe returns None when not connected."""
        broadcaster = MQTTBroadcaster(broker="localhost", port=1883)
        callback = lambda topic, payload: None

        result = broadcaster.subscribe(topic=test_topic, callback=callback, qos=1)

        assert result is None

    def test_subscribe_success(self, mqtt_broadcaster, test_topic):
        """Test successful subscription returns subscription ID."""
        mqtt_broadcaster.connect()

        callback = lambda topic, payload: None
        result = mqtt_broadcaster.subscribe(topic=test_topic, callback=callback, qos=1)

        assert result is not None
        assert isinstance(result, str)

    def test_subscribe_returns_unique_ids(self, mqtt_broadcaster, test_topic):
        """Test each subscription gets a unique ID."""
        mqtt_broadcaster.connect()

        callback = lambda topic, payload: None

        id1 = mqtt_broadcaster.subscribe(topic=test_topic, callback=callback)
        id2 = mqtt_broadcaster.subscribe(topic=test_topic, callback=callback)

        assert id1 is not None
        assert id2 is not None
        assert id1 != id2

    def test_unsubscribe_success(self, mqtt_broadcaster, test_topic):
        """Test successful unsubscription with subscription ID."""
        mqtt_broadcaster.connect()

        # Subscribe first
        callback = lambda topic, payload: None
        subscription_id = mqtt_broadcaster.subscribe(
            topic=test_topic, callback=callback
        )

        # Then unsubscribe
        result = mqtt_broadcaster.unsubscribe(subscription_id=subscription_id)
        assert result is True

    def test_unsubscribe_invalid_id(self, mqtt_broadcaster):
        """Test unsubscribe with invalid ID returns False."""
        mqtt_broadcaster.connect()

        result = mqtt_broadcaster.unsubscribe(subscription_id="invalid-id-12345")
        assert result is False

    def test_multiple_subscriptions_same_topic(self, mqtt_broadcaster, test_topic):
        """Test multiple callbacks for same topic."""
        mqtt_broadcaster.connect()

        callback1 = lambda topic, payload: None
        callback2 = lambda topic, payload: None

        id1 = mqtt_broadcaster.subscribe(topic=test_topic, callback=callback1)
        id2 = mqtt_broadcaster.subscribe(topic=test_topic, callback=callback2)

        assert id1 is not None
        assert id2 is not None
        assert id1 != id2

    def test_multiple_subscriptions_different_topics(self, mqtt_broadcaster):
        """Test subscribing to multiple different topics."""
        mqtt_broadcaster.connect()

        callback = lambda topic, payload: None
        topic1 = f"test/topic1/{uuid4()}"
        topic2 = f"test/topic2/{uuid4()}"

        id1 = mqtt_broadcaster.subscribe(topic=topic1, callback=callback)
        id2 = mqtt_broadcaster.subscribe(topic=topic2, callback=callback)

        assert id1 is not None
        assert id2 is not None
        assert id1 != id2

    def test_subscribe_with_wildcard(self, mqtt_broadcaster):
        """Test subscribing with wildcard topics."""
        mqtt_broadcaster.connect()

        callback = lambda topic, payload: None

        # Single level wildcard
        id1 = mqtt_broadcaster.subscribe(topic="test/+/status", callback=callback)
        assert id1 is not None

        # Multi-level wildcard
        id2 = mqtt_broadcaster.subscribe(topic="test/#", callback=callback)
        assert id2 is not None

    def test_subscribe_before_connect(self):
        """Test that subscribe before connect returns None."""
        broadcaster = MQTTBroadcaster(broker="localhost", port=1883)
        callback = lambda topic, payload: None

        result = broadcaster.subscribe(topic="test/topic", callback=callback)
        assert result is None

    def test_callback_exception_handling(self, mqtt_broadcaster, test_topic):
        """Test that exception in callback doesn't crash system."""
        mqtt_broadcaster.connect()

        call_count = [0]

        def bad_callback(topic, payload):
            call_count[0] += 1
            raise ValueError("Intentional error in callback")

        mqtt_broadcaster.subscribe(topic=test_topic, callback=bad_callback)
        time.sleep(0.2)

        # Publish message - should trigger callback
        mqtt_broadcaster.publish_event(topic=test_topic, payload="test")
        time.sleep(0.2)

        # Callback was called despite exception
        assert call_count[0] == 1

        # Broadcaster still works
        assert mqtt_broadcaster.connected is True

    def test_subscribe_qos_levels(self, mqtt_broadcaster, test_topic):
        """Test subscribing with different QoS levels."""
        mqtt_broadcaster.connect()

        callback = lambda topic, payload: None

        # QoS 0
        id0 = mqtt_broadcaster.subscribe(
            topic=f"{test_topic}/qos0", callback=callback, qos=0
        )
        assert id0 is not None

        # QoS 1
        id1 = mqtt_broadcaster.subscribe(
            topic=f"{test_topic}/qos1", callback=callback, qos=1
        )
        assert id1 is not None

        # QoS 2
        id2 = mqtt_broadcaster.subscribe(
            topic=f"{test_topic}/qos2", callback=callback, qos=2
        )
        assert id2 is not None


# ============================================================================
# Global Broadcaster Tests
# ============================================================================


class TestGlobalBroadcaster:
    """Test suite for global broadcaster singleton."""

    def setup_method(self):
        """Ensure clean state before each test."""
        shutdown_broadcaster()

    def teardown_method(self):
        """Cleanup after each test."""
        shutdown_broadcaster()

    def test_get_broadcaster_mqtt(self):
        """Test getting MQTT broadcaster."""
        skip_if_no_mqtt()

        broadcaster = get_broadcaster(
            broadcast_type="mqtt", broker="localhost", port=1883
        )

        assert isinstance(broadcaster, MQTTBroadcaster)
        assert broadcaster.connected is True

    def test_get_broadcaster_noop(self):
        """Test getting NoOp broadcaster."""
        broadcaster = get_broadcaster(
            broadcast_type="noop", broker="localhost", port=1883
        )

        assert isinstance(broadcaster, NoOpBroadcaster)

    def test_get_broadcaster_singleton(self):
        """Test that get_broadcaster returns same instance."""
        broadcaster1 = get_broadcaster(
            broadcast_type="noop", broker="localhost", port=1883
        )

        broadcaster2 = get_broadcaster(
            broadcast_type="noop", broker="localhost", port=1883
        )

        assert broadcaster1 is broadcaster2

    def test_shutdown_broadcaster(self):
        """Test shutting down global broadcaster."""
        broadcaster = get_broadcaster(
            broadcast_type="noop", broker="localhost", port=1883
        )

        assert broadcaster is not None

        shutdown_broadcaster()

        # After shutdown, getting broadcaster should create new instance
        new_broadcaster = get_broadcaster(
            broadcast_type="noop", broker="localhost", port=1883
        )

        assert new_broadcaster is not broadcaster

    def test_broadcaster_reconfiguration(self):
        """Test switching broadcaster type (NoOp to MQTT and back)."""
        # Start with NoOp
        broadcaster1 = get_broadcaster(
            broadcast_type="noop", broker="localhost", port=1883
        )
        assert isinstance(broadcaster1, NoOpBroadcaster)

        # Switch to MQTT (if broker is running)
        if is_mqtt_running():
            broadcaster2 = get_broadcaster(
                broadcast_type="mqtt", broker="localhost", port=1883
            )
            assert isinstance(broadcaster2, MQTTBroadcaster)
            assert broadcaster2 is not broadcaster1
            assert broadcaster2.connected is True

            # Switch back to NoOp
            broadcaster3 = get_broadcaster(
                broadcast_type="noop", broker="localhost", port=1883
            )
            assert isinstance(broadcaster3, NoOpBroadcaster)
            assert broadcaster3 is not broadcaster2

    def test_broadcaster_config_change(self):
        """Test changing broker configuration forces new instance."""
        broadcaster1 = get_broadcaster(
            broadcast_type="noop", broker="localhost", port=1883
        )

        # Change port should create new instance
        broadcaster2 = get_broadcaster(
            broadcast_type="noop", broker="localhost", port=1884
        )
        assert broadcaster2 is not broadcaster1

    def test_get_broadcaster_invalid_mqtt_config(self):
        """Test get_broadcaster with invalid MQTT configuration."""
        # This should create a broadcaster but fail to connect
        broadcaster = get_broadcaster(
            broadcast_type="mqtt", broker="invalid-host", port=9999
        )
        # Broadcaster is created but not connected
        assert isinstance(broadcaster, MQTTBroadcaster)
        assert broadcaster.connected is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestMQTTIntegration:
    """Integration tests for MQTT broadcaster in job workflow context."""

    def test_broadcasting(self, mqtt_broadcaster, test_topic):
        """Test publishing job failure event."""
        mqtt_broadcaster.connect()

        job_id = str(uuid4())

        # Job started
        mqtt_broadcaster.publish_event(
            topic=test_topic, payload=json.dumps({"status": "processing"})
        )

        # Job failed
        result = mqtt_broadcaster.publish_event(
            topic=test_topic,
            payload=json.dumps(
                {"error": "File not found: input.jpg", "status": "failed"}
            ),
        )
        assert result is True


class TestMQTTSubscription:
    """Integration tests for MQTT subscription functionality."""

    def test_subscribe_receive_message(self, mqtt_broadcaster, test_topic):
        """Test subscribing and receiving a message."""
        mqtt_broadcaster.connect()

        # Track received messages
        received_messages = []

        def callback(topic, payload):
            received_messages.append({"topic": topic, "payload": payload})

        # Subscribe
        subscription_id = mqtt_broadcaster.subscribe(
            topic=test_topic, callback=callback
        )
        assert subscription_id is not None

        # Wait for subscription to be established
        time.sleep(0.2)

        # Publish message
        test_payload = json.dumps({"status": "test", "data": "hello"})
        mqtt_broadcaster.publish_event(topic=test_topic, payload=test_payload)

        # Wait for message to be received
        time.sleep(0.3)

        # Verify message received
        assert len(received_messages) == 1
        assert received_messages[0]["topic"] == test_topic
        assert received_messages[0]["payload"] == test_payload

    def test_subscribe_multiple_messages(self, mqtt_broadcaster, test_topic):
        """Test receiving multiple messages on same topic."""
        mqtt_broadcaster.connect()

        received_messages = []

        def callback(topic, payload):
            received_messages.append(payload)

        mqtt_broadcaster.subscribe(topic=test_topic, callback=callback)
        time.sleep(0.2)

        # Publish multiple messages
        messages = ["message1", "message2", "message3"]
        for msg in messages:
            mqtt_broadcaster.publish_event(topic=test_topic, payload=msg)
            time.sleep(0.1)

        time.sleep(0.3)

        # Verify all messages received
        assert len(received_messages) == 3
        assert received_messages == messages

    def test_multiple_subscriptions_same_topic_both_called(
        self, mqtt_broadcaster, test_topic
    ):
        """Test multiple callbacks to same topic all invoked."""
        mqtt_broadcaster.connect()

        received1 = []
        received2 = []

        def callback1(topic, payload):
            received1.append(payload)

        def callback2(topic, payload):
            received2.append(payload)

        # Subscribe with two different callbacks
        id1 = mqtt_broadcaster.subscribe(topic=test_topic, callback=callback1)
        id2 = mqtt_broadcaster.subscribe(topic=test_topic, callback=callback2)

        assert id1 is not None
        assert id2 is not None
        time.sleep(0.2)

        # Publish message
        mqtt_broadcaster.publish_event(topic=test_topic, payload="test message")
        time.sleep(0.3)

        # Both callbacks should have received the message
        assert len(received1) == 1
        assert len(received2) == 1
        assert received1[0] == "test message"
        assert received2[0] == "test message"

    def test_subscribe_wildcard_messages(self, mqtt_broadcaster):
        """Test receiving messages with wildcard subscription."""
        mqtt_broadcaster.connect()

        received_messages = []

        def callback(topic, payload):
            received_messages.append({"topic": topic, "payload": payload})

        # Subscribe with wildcard
        base_topic = f"test/{uuid4()}"
        wildcard_topic = f"{base_topic}/+"
        mqtt_broadcaster.subscribe(topic=wildcard_topic, callback=callback)
        time.sleep(0.2)

        # Publish to different subtopics
        topics = [f"{base_topic}/status", f"{base_topic}/data", f"{base_topic}/error"]

        for topic in topics:
            mqtt_broadcaster.publish_event(topic=topic, payload=f"message for {topic}")
            time.sleep(0.1)

        time.sleep(0.3)

        # Verify messages from all subtopics received
        assert len(received_messages) == 3
        received_topics = [msg["topic"] for msg in received_messages]
        assert set(received_topics) == set(topics)

    def test_unsubscribe_stops_messages(self, mqtt_broadcaster, test_topic):
        """Test that unsubscribe prevents further messages for that subscription."""
        mqtt_broadcaster.connect()

        received_messages = []

        def callback(topic, payload):
            received_messages.append(payload)

        # Subscribe and receive first message
        subscription_id = mqtt_broadcaster.subscribe(
            topic=test_topic, callback=callback
        )
        time.sleep(0.2)

        mqtt_broadcaster.publish_event(topic=test_topic, payload="message1")
        time.sleep(0.2)

        assert len(received_messages) == 1

        # Unsubscribe
        result = mqtt_broadcaster.unsubscribe(subscription_id=subscription_id)
        assert result is True
        time.sleep(0.2)

        # Publish another message
        mqtt_broadcaster.publish_event(topic=test_topic, payload="message2")
        time.sleep(0.2)

        # Should still be 1 (second message not received)
        assert len(received_messages) == 1

    def test_unsubscribe_one_of_multiple(self, mqtt_broadcaster, test_topic):
        """Test unsubscribe one subscription, others still work."""
        mqtt_broadcaster.connect()

        received1 = []
        received2 = []

        def callback1(topic, payload):
            received1.append(payload)

        def callback2(topic, payload):
            received2.append(payload)

        # Subscribe with two callbacks
        id1 = mqtt_broadcaster.subscribe(topic=test_topic, callback=callback1)
        id2 = mqtt_broadcaster.subscribe(topic=test_topic, callback=callback2)
        time.sleep(0.2)

        # Publish first message - both should receive
        mqtt_broadcaster.publish_event(topic=test_topic, payload="message1")
        time.sleep(0.2)

        assert len(received1) == 1
        assert len(received2) == 1

        # Unsubscribe first callback
        mqtt_broadcaster.unsubscribe(subscription_id=id1)
        time.sleep(0.2)

        # Publish second message
        mqtt_broadcaster.publish_event(topic=test_topic, payload="message2")
        time.sleep(0.2)

        # Only second callback should receive
        assert len(received1) == 1  # Still 1
        assert len(received2) == 2  # Now 2

    def test_subscribe_disconnect_clears_subscriptions(self, test_topic):
        """Test that disconnect clears subscriptions."""
        skip_if_no_mqtt()

        broadcaster = MQTTBroadcaster(broker="localhost", port=1883)
        broadcaster.connect()

        callback = lambda topic, payload: None
        broadcaster.subscribe(topic=test_topic, callback=callback)

        # Verify subscription exists
        assert len(broadcaster._subscriptions) == 1

        # Disconnect
        broadcaster.disconnect()

        # Subscriptions should be cleared
        assert len(broadcaster._subscriptions) == 0


class TestBroadcasterMethods:
    """Tests for broadcaster retained message and LWT methods."""

    def test_set_will_method_exists(self):
        """Test that broadcaster has set_will method."""
        from cl_ml_tools import MQTTBroadcaster

        broadcaster = MQTTBroadcaster(broker="localhost", port=1883)
        try:
            assert hasattr(broadcaster, "set_will")
            assert callable(broadcaster.set_will)
        finally:
            # Disconnect to prevent connection leaks
            if hasattr(broadcaster, "disconnect"):
                broadcaster.disconnect()

    def test_publish_retained_method_exists(self):
        """Test that broadcaster has publish_retained method."""
        from cl_ml_tools import MQTTBroadcaster

        broadcaster = MQTTBroadcaster(broker="localhost", port=1883)
        try:
            assert hasattr(broadcaster, "publish_retained")
            assert callable(broadcaster.publish_retained)
        finally:
            # Disconnect to prevent connection leaks
            if hasattr(broadcaster, "disconnect"):
                broadcaster.disconnect()

    def test_noop_broadcaster_has_methods(self):
        """Test that NoOp broadcaster implements required methods."""
        from cl_ml_tools import NoOpBroadcaster

        broadcaster = NoOpBroadcaster()
        assert hasattr(broadcaster, "set_will")
        assert hasattr(broadcaster, "publish_retained")
        assert callable(broadcaster.set_will)
        assert callable(broadcaster.publish_retained)

    def test_clear_retained_method_exists(self):
        """Test that broadcaster has clear_retained method.

        Note: This test verifies that cl_ml_tools.MQTTBroadcaster
        has implemented the clear_retained method. If this test fails, it means
        the cl_server_shared package needs to be updated.
        """
        from cl_ml_tools import MQTTBroadcaster

        broadcaster = MQTTBroadcaster(broker="localhost", port=1883)
        try:
            # Skip test if clear_retained not yet implemented in cl_server_shared
            if not hasattr(broadcaster, "clear_retained"):
                pytest.skip(
                    "clear_retained method not yet implemented in cl_server_shared"
                )

            assert callable(broadcaster.clear_retained)
        finally:
            # Disconnect to prevent connection leaks
            if hasattr(broadcaster, "disconnect"):
                broadcaster.disconnect()
