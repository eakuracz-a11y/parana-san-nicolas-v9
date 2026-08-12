import requests
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

INA_URL = "https://alerta.ina.gob.ar/pub/datos/datos"

STATIONS = [
    "Corrientes",
    "Goya",
    "La Paz",
    "Paraná",
    "Diamante",
    "Rosario",
    "Villa Constitución",
    "San Nicolás",
]

SAN_NICOLAS_SERIES_ID = 36


# ============================================================
# CONSULTA SIMPLE AL INA
# ============================================================

def get_series(start, end):

    params = {
        "timeStart": str(start),
        "timeEnd": str(end),
        "seriesId": SAN_NICOLAS_SERIES_ID,
        "format": "json",
    }

    response = requests.get(INA_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):

        if "data" in data:
            data = data["data"]

        elif "results" in data:
            data = data["results"]

        else:
            data = []

    df = pd.DataFrame(data)

    if df.empty:
        return df

    # Normalizar columnas
    for col in df.columns:

        name = str(col).lower()

        if name in ["fecha", "datetime", "timestamp"]:
            df = df.rename(columns={col: "datetime"})

        if name in ["valor", "nivel", "altura", "value"]:
            df = df.rename(columns={col: "value"})

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


# ============================================================
# FUNCIÓN QUE USA app.py
# ============================================================

def observed(start, end):

    from datetime import datetime, timedelta
import pandas as pd

def observed(start, end):

    try:
        # Convertir fechas
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # Evitar fechas futuras
        today = pd.Timestamp.today().normalize()

        if end_dt > today:
            end_dt = today

        # Si el rango es inválido, usar últimos 30 días
        if start_dt >= end_dt:
            start_dt = end_dt - pd.Timedelta(days=30)

        df = get_series(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
        )

        if df.empty:
            return (
                pd.DataFrame(),
                "El INA no devolvió datos para el período seleccionado."
            )

        if "datetime" not in df.columns:
            return pd.DataFrame(), "Falta columna de fecha."

        if "value" not in df.columns:
            return pd.DataFrame(), "Falta columna de nivel."

        df = df.dropna(subset=["datetime", "value"])

        if df.empty:
            return pd.DataFrame(), "No hay valores válidos de nivel."

        df = df.sort_values("datetime").reset_index(drop=True)

        return df, None

    except Exception as exc:

        return pd.DataFrame(), str(exc)


# ============================================================
# METADATOS
# ============================================================

def forecast_meta():

    return {
        "fuente": "Instituto Nacional del Agua (INA)",
        "estacion": "San Nicolás",
        "serie": SAN_NICOLAS_SERIES_ID,
        "variable": "Nivel hidrométrico",
        "unidad": "m",
    }
