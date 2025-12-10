from datetime import datetime, timezone


def toTimeStamp(localTime: datetime, silent: bool = True):
    """
    Converts a local datetime object to a UTC timestamp in milliseconds.
    If the input datetime is naive (lacks timezone info), it's assumed to be
    in the system's local timezone and then converted.

    If the datetime object is naive (no timezone info), assume it's in the
    system's local time and make it timezone-aware.
    """

    if not isinstance(localTime, datetime):
        print("Error: Input 'localTime' must be a datetime object.")
        if silent:
            return None
        else:
            raise Exception("Input 'localTime' must be a datetime object.")

    if localTime.tzinfo is None or localTime.tzinfo.utcoffset(localTime) is None:
        localTime = localTime.astimezone()

    utc_dt = localTime.astimezone(timezone.utc)

    utc_timestamp_ms = int(utc_dt.timestamp() * 1000)
    return utc_timestamp_ms


def fromTimeStamp(utc_timestamp_ms: int) -> datetime:
    """
    Converts a UTC timestamp in milliseconds to a timezone-aware local datetime object.
    """
    
    utc_timestamp_s = utc_timestamp_ms / 1000
    utc_dt = datetime.fromtimestamp(utc_timestamp_s, tz=timezone.utc)
    local_dt = utc_dt.astimezone()
    return local_dt