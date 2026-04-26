def require(condition, message, error_type=ValueError):
    if not bool(condition):
        raise error_type(message)
