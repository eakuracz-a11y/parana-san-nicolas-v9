import requests
import pandas as pd

# Stations used by the V9 interface. The INA service/API can change over time.
STATIONS = [
    "Corrientes", "Goya", "La Paz", "Paraná",
    "Diamante", "Rosario", "Villa Constitución", "San Nicolás"
]

# Public INA endpoint configuration is isolated here so it can be updated
# without changing the Streamlit interface.
INA_URL = "https://alerta.ina.gob.ar/"

def observed(start: str, end: str):
    """
    Retrieve observed hydrometric data.

    The public INA endpoint may change its API contract. This implementation
    first attempts the known public service and, if the service cannot be
    parsed, returns a clear error rather than inventing measurements.
    """
    errors = []
    # Candidate endpoints kept intentionally conservative.
    candidates = [
        f"{INA_URL.rstrip('/')}/api/series?start={start}&end={end}",
        f"{INA_URL.rstrip('/')}/api/observed?start={start}&end={end}",
    ]

    for url in candidates:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            df = _normalize(data)
            if df is not None and not df.empty:
                return df, errors
        except Exception as e:
            errors.append(f"{url}: {e}")

    raise RuntimeError(
        "No fue posible obtener datos observados del INA. "
        "El servicio/API público puede haber cambiado. "
        + (" | ".join(errors) if errors else "")
    )

def _normalize(data):
    if isinstance(data, dict):
        for key in ("data", "results", "series", "observations"):
            if key in data:
                data = data[key]
                break

    if not isinstance(data, list):
        return None

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        dt = item.get("datetime") or item.get("date") or item.get("timestamp") or item.get("fecha")
        station = item.get("station") or item.get("station_name") or item.get("estacion")
        value = item.get("value") or item.get("level") or item.get("nivel")
        if dt is not None and station is not None and value is not None:
            rows.append({"datetime": dt, "station": station, "value": value})

    if not rows:
        # Also support already-tabular dictionaries.
        try:
            df = pd.DataFrame(data)
            if "datetime" in df.columns:
                return _pivot(df)
        except Exception:
            return None
        return None

    return _pivot(pd.DataFrame(rows))

def _pivot(df):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["datetime", "station"])
    out = df.pivot_table(index="datetime", columns="station", values="value", aggfunc="last").reset_index()
    out.columns.name = None
    return out

def forecast_meta():
    try:
        r = requests.get(f"{INA_URL.rstrip('/')}/", timeout=20)
        return {
            "fuente": "INA",
            "url": INA_URL,
            "http_status": r.status_code,
            "mensaje": "Consulta realizada al portal público del INA."
        }
    except Exception as e:
        return {"fuente": "INA", "url": INA_URL, "error": str(e)}
