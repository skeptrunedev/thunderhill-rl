"""Explicit simulator initial conditions shared by native launch and runtime."""
import math


def scene_id(initial_speed_m_s: float = 0.0) -> str:
    if (type(initial_speed_m_s) not in (int, float)
            or not math.isfinite(initial_speed_m_s)
            or not 0 <= initial_speed_m_s <= 10):
        raise ValueError("Initial speed must be finite and in [0, 10] m/s")
    if initial_speed_m_s == 0:
        return "thunderhill-east-standing"
    return "thunderhill-east-rolling-" + str(float(initial_speed_m_s)) + "mps"
