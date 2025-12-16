"""Comprehensive test suite for timestamp utility."""
from datetime import datetime, timedelta, timezone

from cl_ml_tools.utils.timestamp import fromTimeStamp, toTimeStamp

# ============================================================================
# Test Class 1: toTimeStamp Conversion
# ============================================================================


class TestToTimeStamp:
    """Test toTimeStamp function (datetime -> millisecond timestamp)."""

    def test_utc_datetime_to_timestamp(self) -> None:
        """Test conversion of UTC datetime to timestamp."""
        # 2024-01-01 00:00:00 UTC = 1704067200000 ms
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        timestamp = toTimeStamp(dt)

        assert timestamp == 1704067200000

    def test_aware_datetime_to_timestamp(self) -> None:
        """Test conversion of timezone-aware datetime."""
        # 2024-01-01 12:00:00 UTC
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        timestamp = toTimeStamp(dt)

        # 12 hours = 43200 seconds = 43200000 ms
        assert timestamp == 1704110400000

    def test_naive_datetime_to_timestamp(self) -> None:
        """Test conversion of naive datetime (assumes local timezone)."""
        # Naive datetime - will be interpreted as local time
        dt = datetime(2024, 1, 1, 0, 0, 0)
        timestamp = toTimeStamp(dt)

        # Should return a valid timestamp (exact value depends on system timezone)
        assert isinstance(timestamp, int)
        assert timestamp > 0

    def test_with_offset_timezone(self) -> None:
        """Test conversion with custom timezone offset."""
        # Create datetime with +05:30 offset (IST)
        offset = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2024, 1, 1, 5, 30, 0, tzinfo=offset)

        # When converted to UTC, this should be 2024-01-01 00:00:00
        timestamp = toTimeStamp(dt)
        assert timestamp == 1704067200000

    def test_with_negative_offset_timezone(self) -> None:
        """Test conversion with negative timezone offset."""
        # Create datetime with -05:00 offset (EST)
        offset = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=offset)

        # When converted to UTC, this should be 2024-01-01 05:00:00
        # 5 hours = 18000 seconds = 18000000 ms
        timestamp = toTimeStamp(dt)
        assert timestamp == 1704085200000  # 1704067200000 + 18000000

    def test_millisecond_precision(self) -> None:
        """Test that conversion preserves millisecond precision."""
        dt = datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)  # 123.456 ms
        timestamp = toTimeStamp(dt)

        # Should include milliseconds
        assert timestamp == 1704067200123

    def test_epoch_timestamp(self) -> None:
        """Test conversion of Unix epoch (1970-01-01 00:00:00 UTC)."""
        dt = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        timestamp = toTimeStamp(dt)

        assert timestamp == 0

    def test_far_future_timestamp(self) -> None:
        """Test conversion of far future datetime."""
        dt = datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        timestamp = toTimeStamp(dt)

        # Should be a large positive integer
        assert timestamp > 4000000000000  # Year 2099 is far in the future


# ============================================================================
# Test Class 2: fromTimeStamp Conversion
# ============================================================================


class TestFromTimeStamp:
    """Test fromTimeStamp function (millisecond timestamp -> datetime)."""

    def test_timestamp_to_utc_datetime(self) -> None:
        """Test conversion of timestamp to datetime."""
        # 1704067200000 ms = 2024-01-01 00:00:00 UTC
        timestamp = 1704067200000
        dt = fromTimeStamp(timestamp)

        # Convert to UTC for comparison
        dt_utc = dt.astimezone(timezone.utc)

        assert dt_utc.year == 2024
        assert dt_utc.month == 1
        assert dt_utc.day == 1
        assert dt_utc.hour == 0
        assert dt_utc.minute == 0
        assert dt_utc.second == 0

    def test_timestamp_with_milliseconds(self) -> None:
        """Test conversion of timestamp with milliseconds."""
        # 1704067200123 ms = 2024-01-01 00:00:00.123 UTC
        timestamp = 1704067200123
        dt = fromTimeStamp(timestamp)

        dt_utc = dt.astimezone(timezone.utc)

        assert dt_utc.year == 2024
        assert dt_utc.microsecond == 123000  # 123 ms = 123000 microseconds

    def test_epoch_timestamp_conversion(self) -> None:
        """Test conversion of Unix epoch timestamp."""
        timestamp = 0
        dt = fromTimeStamp(timestamp)

        dt_utc = dt.astimezone(timezone.utc)

        assert dt_utc.year == 1970
        assert dt_utc.month == 1
        assert dt_utc.day == 1
        assert dt_utc.hour == 0
        assert dt_utc.minute == 0
        assert dt_utc.second == 0

    def test_returned_datetime_is_aware(self) -> None:
        """Test that returned datetime is timezone-aware."""
        timestamp = 1704067200000
        dt = fromTimeStamp(timestamp)

        # Should be timezone-aware (local timezone)
        assert dt.tzinfo is not None
        assert dt.tzinfo.utcoffset(dt) is not None

    def test_far_future_timestamp_conversion(self) -> None:
        """Test conversion of far future timestamp."""
        # 4102444799999 ms = 2099-12-31 23:59:59.999 UTC (approximately)
        timestamp = 4102444799999
        dt = fromTimeStamp(timestamp)

        dt_utc = dt.astimezone(timezone.utc)

        assert dt_utc.year == 2099
        assert dt_utc.month == 12
        assert dt_utc.day == 31


# ============================================================================
# Test Class 3: Round-Trip Conversion
# ============================================================================


class TestRoundTripConversion:
    """Test round-trip conversions (datetime -> timestamp -> datetime)."""

    def test_utc_datetime_round_trip(self) -> None:
        """Test round-trip conversion preserves UTC datetime."""
        original = datetime(2024, 6, 15, 14, 30, 45, 123456, tzinfo=timezone.utc)

        # Convert to timestamp and back
        timestamp = toTimeStamp(original)
        restored = fromTimeStamp(timestamp)

        # Convert both to UTC for comparison
        restored_utc = restored.astimezone(timezone.utc)

        assert original.year == restored_utc.year
        assert original.month == restored_utc.month
        assert original.day == restored_utc.day
        assert original.hour == restored_utc.hour
        assert original.minute == restored_utc.minute
        assert original.second == restored_utc.second
        # Microsecond precision may be slightly off due to millisecond conversion
        assert abs(original.microsecond - restored_utc.microsecond) < 1000

    def test_multiple_round_trips(self) -> None:
        """Test that multiple round-trips don't accumulate errors."""
        dt = datetime(2024, 3, 15, 10, 20, 30, tzinfo=timezone.utc)

        # Multiple round trips
        for _ in range(5):
            timestamp = toTimeStamp(dt)
            dt = fromTimeStamp(timestamp)

        # Should still match original (within microsecond precision)
        dt_utc = dt.astimezone(timezone.utc)
        assert dt_utc.year == 2024
        assert dt_utc.month == 3
        assert dt_utc.day == 15
        assert dt_utc.hour == 10
        assert dt_utc.minute == 20
        assert dt_utc.second == 30

    def test_timestamp_round_trip(self) -> None:
        """Test round-trip conversion preserves timestamp."""
        original_timestamp = 1704110400000  # 2024-01-01 12:00:00 UTC

        # Convert to datetime and back
        dt = fromTimeStamp(original_timestamp)
        restored_timestamp = toTimeStamp(dt)

        # Should match exactly
        assert restored_timestamp == original_timestamp


# ============================================================================
# Test Class 4: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_timestamp(self) -> None:
        """Test handling of zero timestamp (Unix epoch)."""
        timestamp = 0
        dt = fromTimeStamp(timestamp)

        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None

        # Round-trip should preserve zero
        assert toTimeStamp(dt) == 0

    def test_negative_timestamp(self) -> None:
        """Test handling of negative timestamp (before 1970)."""
        # -86400000 ms = 1969-12-31 00:00:00 UTC (1 day before epoch)
        timestamp = -86400000
        dt = fromTimeStamp(timestamp)

        dt_utc = dt.astimezone(timezone.utc)

        assert dt_utc.year == 1969
        assert dt_utc.month == 12
        assert dt_utc.day == 31

    def test_very_large_timestamp(self) -> None:
        """Test handling of very large timestamp."""
        # Large but valid timestamp (year 2099 - safely within datetime range)
        timestamp = 4102444799000  # Dec 31, 2099
        dt = fromTimeStamp(timestamp)

        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None

        # Verify it's a far future date
        dt_utc = dt.astimezone(timezone.utc)
        assert dt_utc.year == 2099
        assert dt_utc.month == 12

    def test_datetime_with_microseconds(self) -> None:
        """Test datetime with microseconds (higher precision than milliseconds)."""
        dt = datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)
        timestamp = toTimeStamp(dt)

        # Microseconds should be truncated to milliseconds
        # 123456 microseconds = 123 milliseconds (truncated)
        assert timestamp == 1704067200123

    def test_naive_datetime_consistency(self) -> None:
        """Test that naive datetime conversion is consistent."""
        # Create two naive datetimes with same values
        dt1 = datetime(2024, 1, 1, 12, 0, 0)
        dt2 = datetime(2024, 1, 1, 12, 0, 0)

        # Both should produce the same timestamp
        timestamp1 = toTimeStamp(dt1)
        timestamp2 = toTimeStamp(dt2)

        assert timestamp1 == timestamp2
