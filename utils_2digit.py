def is_valid_2d_number(x):
    try:
        n = int(x)
    except Exception:
        return False
    return 0 <= n <= 99


def to_2d_str(x):
    n = int(x)
    if n < 0 or n > 99:
        raise ValueError("2-digit number must be between 0 and 99")
    return f"{n:02d}"
