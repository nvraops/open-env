MIN_OPEN_SCORE = 0.001
MAX_OPEN_SCORE = 0.999


def clamp_open_score(value: float) -> float:
    return min(MAX_OPEN_SCORE, max(MIN_OPEN_SCORE, float(value)))


def finalize_open_score(value: float, digits: int = 4) -> float:
    rounded = round(float(value), digits)
    return clamp_open_score(rounded)


def scale_score_to_band(
    value: float,
    min_score: float,
    max_score: float,
) -> float:
    clamped_value = clamp_open_score(value)
    scaled = min_score + (max_score - min_score) * clamped_value
    return finalize_open_score(scaled)
