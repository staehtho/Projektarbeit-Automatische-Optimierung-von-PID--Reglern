from enum import Enum, IntEnum, auto


class AntiWindup(Enum):
    CLAMPING = auto()
    CONDITIONAL = auto()


# IntEnum for Numba (only integer)
class AntiWindupInt(IntEnum):
    CLAMPING = AntiWindup.CLAMPING.value
    CONDITIONAL = AntiWindup.CONDITIONAL.value


class PerformanceIndex(Enum):
    IAE = auto()
    ISE = auto()
    ITAE = auto()
    ITSE = auto()


# IntEnum for Numba (only integer)
class PerformanceIndexInt(IntEnum):
    IAE = PerformanceIndex.IAE.value
    ISE = PerformanceIndex.ISE.value
    ITAE = PerformanceIndex.ITAE.value
    ITSE = PerformanceIndex.ITSE.value


class MySolver(Enum):
    RK4 = auto()


# IntEnum for Numba (only integer)
class MySolverInt(IntEnum):
    RK4 = MySolver.RK4.value


def map_enum_to_int(enum_value: Enum) -> IntEnum:
    """
    Map a regular Enum member to its corresponding IntEnum member.

    This version does not rely on globals() and works even if the Enums
    are imported from another module or have potential name conflicts.

    Args:
        enum_value (Enum): The Enum member to convert.

    Returns:
        IntEnum: The corresponding IntEnum member.

    Raises:
        ValueError: If no matching IntEnum class is found or the member
            name does not exist in the resolved IntEnum class.
    """
    enum_cls = enum_value.__class__
    int_enum_name = enum_cls.__name__ + "Int"

    # Try to get the module where the enum is defined
    module = __import__(enum_cls.__module__, fromlist=[int_enum_name])

    # Get the IntEnum class from that module
    int_enum_cls = getattr(module, int_enum_name, None)
    if int_enum_cls is None:
        raise ValueError(f"Kein IntEnum für {enum_cls.__name__} gefunden in {enum_cls.__module__}")

    # Map using the member name
    return int_enum_cls[enum_value.name]
