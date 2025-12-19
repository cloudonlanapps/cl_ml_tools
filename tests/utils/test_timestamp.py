"""Unit tests for timestamp utilities."""

from datetime import datetime, timedelta, timezone

from cl_ml_tools.utils.timestamp import fromTimeStamp, toTimeStamp


def test_to_timestamp_aware():
    """Test toTimeStamp with timezone-aware datetime."""
    # Create a specific UTC datetime
    dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # 2023-01-01 12:00:00 UTC = 1672574400 seconds = 1672574400000 ms
    expected = 1672574400000
    assert toTimeStamp(dt) == expected


def test_to_timestamp_naive():
    """Test toTimeStamp with naive datetime (assumed local)."""
    dt = datetime(2023, 1, 1, 12, 0, 0)
    ts = toTimeStamp(dt)

    # fromTimeStamp(ts) should return the same local time
    dt_back = fromTimeStamp(ts).replace(tzinfo=None)
    # astimezone() might result in slight differences if DST changes,
    # but for this specific date it should be fine.
    assert dt_back == dt


def test_from_timestamp():
    """Test fromTimeStamp conversion."""
    ts = 1672574400000  # 2023-01-01 12:00:00 UTC
    dt = fromTimeStamp(ts)

    # Convert to UTC for comparison
    dt_utc = dt.astimezone(timezone.utc)
    assert dt_utc.year == 2023
    assert dt_utc.month == 1
    assert dt_utc.day == 1
    assert dt_utc.hour == 12


def test_round_trip():
    """Test round trip conversion."""
    dt = datetime.now(timezone.utc).replace(microsecond=0)
    ts = toTimeStamp(dt)
    dt_back = fromTimeStamp(ts).astimezone(timezone.utc)
    assert dt == dt_back


def test_with_offset():
    """Test with specific timezone offset."""
    offset = timezone(timedelta(hours=5, minutes=30))
    dt = datetime(2023, 5, 20, 10, 0, 0, tzinfo=offset)
    ts = toTimeStamp(dt)

    # 10:00:00 +05:30 = 04:30:00 UTC
    expected_utc = datetime(2023, 5, 20, 4, 30, 0, tzinfo=timezone.utc)
    assert fromTimeStamp(ts).astimezone(timezone.utc) == expected_utc
