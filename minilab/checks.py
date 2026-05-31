import math
from numbers import Integral, Real


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def require_finite_number(value, name):
    require(isinstance(value, Real) and not isinstance(value, bool), f"{name} must be a real number")
    require(math.isfinite(float(value)), f"{name} must be finite")


def require_integer(value, name):
    require(isinstance(value, Integral) and not isinstance(value, bool), f"{name} must be an integer")


def require_finite_fields(obj, names):
    for name in names:
        require_finite_number(getattr(obj, name), name)


def require_integer_fields(obj, names):
    for name in names:
        require_integer(getattr(obj, name), name)
