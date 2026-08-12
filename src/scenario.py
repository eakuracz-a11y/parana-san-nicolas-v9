def run(current_level, rain):
    """
    Experimental rainfall sensitivity scenario.
    It is intentionally not presented as a calibrated rainfall-runoff model.
    """
    total = sum(float(v) for v in rain.values())
    # Conservative illustrative sensitivity coefficient, clearly labeled
    # experimental in the Streamlit application.
    impact = total * 0.0005
    return {
        "Bajo": float(current_level + impact * 0.5),
        "Central": float(current_level + impact),
        "Alto": float(current_level + impact * 1.5),
    }
