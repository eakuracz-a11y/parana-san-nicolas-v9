import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURACIÓN
# ============================================================

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HISTORICAL_FORECAST_URL = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)

INA_SERIES_URL = "https://alerta.ina.gob.ar/pub/datos/series"

INA_A5_URL = "https://alerta.ina.gob.ar/a5/getObservaciones"


# ============================================================
# PUNTOS REPRESENTATIVOS DEL CORREDOR DEL PARANÁ
# ============================================================
#
# Se utilizan varios puntos desde Corrientes hasta San Nicolás
# para que la lluvia no represente solamente la precipitación
# local sobre San Nicolás.
#
# Son puntos representativos de las localidades, no una
# discretización hidrológica completa de la cuenca.
# ============================================================

RAIN_POINTS = {
    "Corrientes": (-27.4692, -58.8306),
    "Goya": (-29.1399, -59.2634),
    "La Paz": (-30.7449, -59.6457),
    "Paraná": (-31.7413, -60.5115),
    "Rosario": (-32.9442, -60.6505),
    "San Nicolás": (-33.3358, -60.2252),
}


# ============================================================
# ESTACIONES PREFERIDAS PARA CAUDAL
# ============================================================

CAUDAL_STATION_PRIORITY = [
    "San Nicolás",
    "Rosario",
    "Paraná",
    "La Paz",
    "Goya",
    "Corrientes",
]


# ============================================================
# CACHE SIMPLE DEL CATÁLOGO INA
# ============================================================

_SERIES_CATALOG_CACHE = None


# ============================================================
# UTILIDADES
# ============================================================

def _request_json(url, params=None, timeout=40):

    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={
            "User-Agent": "Parana-San-Nicolas-V9/1.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    response.raise_for_status()

    return response.json()


def _normalize_date(value):

    dt = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(dt):
        return pd.NaT

    return dt.tz_localize(None).normalize()


# ============================================================
# PRECIPITACIÓN HISTÓRICA
# ============================================================

def get_rain_history(start, end):

    frames = []

    for name, (lat, lon) in RAIN_POINTS.items():

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": str(start),
            "end_date": str(end),
            "daily": "precipitation_sum",
            "timezone": "America/Argentina/Buenos_Aires",
        }

        try:

            data = _request_json(
                HISTORICAL_FORECAST_URL,
                params=params,
            )

            daily = data.get(
                "daily",
                {},
            )

            times = daily.get(
                "time",
                [],
            )

            rain = daily.get(
                "precipitation_sum",
                [],
            )

            if not times:
                continue

            frame = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(
                        times,
                        errors="coerce",
                    ),
                    f"rain_{name}": pd.to_numeric(
                        rain,
                        errors="coerce",
                    ),
                }
            )

            frames.append(
                frame
            )

        except Exception:
            continue

    if not frames:

        return pd.DataFrame(
            columns=[
                "datetime",
                "precip_mm",
            ]
        )

    result = frames[0]

    for frame in frames[1:]:

        result = result.merge(
            frame,
            on="datetime",
            how="outer",
        )

    rain_cols = [
        c
        for c in result.columns
        if c.startswith("rain_")
    ]

    result["precip_mm"] = (
        result[rain_cols]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    result = result[
        [
            "datetime",
            "precip_mm",
        ]
    ]

    result = (
        result
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# PRECIPITACIÓN FUTURA
# ============================================================

def get_rain_forecast(days=15):

    days = max(
        1,
        min(
            int(days),
            16,
        ),
    )

    frames = []

    for name, (lat, lon) in RAIN_POINTS.items():

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "forecast_days": days,
            "timezone": "America/Argentina/Buenos_Aires",
        }

        try:

            data = _request_json(
                FORECAST_URL,
                params=params,
            )

            daily = data.get(
                "daily",
                {},
            )

            times = daily.get(
                "time",
                [],
            )

            rain = daily.get(
                "precipitation_sum",
                [],
            )

            if not times:
                continue

            frame = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(
                        times,
                        errors="coerce",
                    ),
                    f"rain_{name}": pd.to_numeric(
                        rain,
                        errors="coerce",
                    ),
                }
            )

            frames.append(
                frame
            )

        except Exception:
            continue

    if not frames:

        return pd.DataFrame(
            columns=[
                "datetime",
                "precip_mm",
            ]
        )

    result = frames[0]

    for frame in frames[1:]:

        result = result.merge(
            frame,
            on="datetime",
            how="outer",
        )

    rain_cols = [
        c
        for c in result.columns
        if c.startswith("rain_")
    ]

    result["precip_mm"] = (
        result[rain_cols]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    result = result[
        [
            "datetime",
            "precip_mm",
        ]
    ]

    result = (
        result
        .sort_values("datetime")
        .head(days)
        .reset_index(drop=True)
    )

    return result


# ============================================================
# CATÁLOGO DE SERIES INA
# ============================================================

def _get_series_catalog():

    global _SERIES_CATALOG_CACHE

    if _SERIES_CATALOG_CACHE is not None:

        return _SERIES_CATALOG_CACHE

    try:

        data = _request_json(
            INA_SERIES_URL
        )

    except Exception:

        _SERIES_CATALOG_CACHE = []

        return []

    if isinstance(data, list):

        catalog = data

    elif isinstance(data, dict):

        catalog = []

        for value in data.values():

            if isinstance(
                value,
                list,
            ):

                catalog = value

                break

    else:

        catalog = []

    _SERIES_CATALOG_CACHE = catalog

    return catalog


# ============================================================
# SELECCIONAR SERIE DE CAUDAL
# ============================================================

def find_best_caudal_series():

    catalog = _get_series_catalog()

    if not catalog:
        return None

    candidates = []

    for row in catalog:

        if not isinstance(
            row,
            dict,
        ):
            continue

        try:

            varid = int(
                row.get(
                    "varid",
                    -1,
                )
            )

        except Exception:

            continue

        # Caudal
        if varid != 4:
            continue

        station = str(
            row.get(
                "estacion_nombre",
                "",
            )
        ).strip()

        if station not in CAUDAL_STATION_PRIORITY:
            continue

        series_id = (
            row.get("seriesid")
            or row.get("seriesId")
        )

        if series_id is None:
            continue

        try:

            series_id = int(
                series_id
            )

        except Exception:
            continue

        procid = row.get(
            "procid"
        )

        try:
            procid = int(
                procid
            )
        except Exception:
            procid = -1

        obs_count = row.get(
            "obs_count"
        )

        try:
            obs_count = int(
                obs_count
            )
        except Exception:
            obs_count = 0

        to_date = pd.to_datetime(
            row.get(
                "to_date"
            ),
            errors="coerce",
            utc=True,
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score = 0

        # Preferir medición directa
        if procid == 1:
            score += 100

        # Curva de gasto
        elif procid == 2:
            score += 80

        if obs_count > 0:
            score += 40

        if pd.notna(
            to_date
        ):

            score += 30

            age_days = (
                pd.Timestamp.now(
                    tz="UTC"
                )
                - to_date
            ).days

            if age_days <= 7:
                score += 50

            elif age_days <= 30:
                score += 35

            elif age_days <= 180:
                score += 20

        try:

            priority = (
                len(
                    CAUDAL_STATION_PRIORITY
                )
                - CAUDAL_STATION_PRIORITY.index(
                    station
                )
            )

            score += priority

        except ValueError:
            pass

        candidates.append(
            {
                "series_id": series_id,
                "station": station,
                "procid": procid,
                "proc_name": row.get(
                    "proc_nombre"
                ),
                "unit": row.get(
                    "unit_nombre"
                ),
                "obs_count": obs_count,
                "to_date": row.get(
                    "to_date"
                ),
                "score": score,
            }
        )

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[0]


# ============================================================
# HISTORIAL DE CAUDAL
# ============================================================

def get_caudal_history(
    start,
    end,
):

    info = find_best_caudal_series()

    if info is None:

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    "caudal_m3s",
                ]
            ),
            None,
        )

    params = {
        "tipo": "puntual",
        "series_id": info[
            "series_id"
        ],
        "timestart": str(start),
        "timeend": str(end),
    }

    try:

        data = _request_json(
            INA_A5_URL,
            params=params,
        )

    except Exception:

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    "caudal_m3s",
                ]
            ),
            info,
        )

    if not isinstance(
        data,
        list,
    ):

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    "caudal_m3s",
                ]
            ),
            info,
        )

    df = pd.DataFrame(
        data
    )

    if df.empty:

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    "caudal_m3s",
                ]
            ),
            info,
        )

    if (
        "timestart" not in df.columns
        or "valor" not in df.columns
    ):

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    "caudal_m3s",
                ]
            ),
            info,
        )

    df["datetime"] = pd.to_datetime(
        df["timestart"],
        errors="coerce",
        utc=True,
    )

    df["datetime"] = (
        df["datetime"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    df["caudal_m3s"] = pd.to_numeric(
        df["valor"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "datetime",
            "caudal_m3s",
        ]
    )

    # Promedio diario si hubiera más de una observación
    df = (
        df.groupby(
            "datetime",
            as_index=False,
        )["caudal_m3s"]
        .mean()
    )

    return (
        df,
        info,
    )


# ============================================================
# PROYECCIÓN EXPERIMENTAL DE CAUDAL
# ============================================================

def project_caudal(
    history,
    future_dates,
):

    future_dates = pd.to_datetime(
        future_dates,
        errors="coerce",
    )

    if (
        history is None
        or history.empty
        or "caudal_m3s" not in history.columns
    ):

        return pd.DataFrame(
            {
                "datetime": future_dates,
                "caudal_m3s": np.nan,
            }
        )

    values = (
        history["caudal_m3s"]
        .dropna()
        .tail(14)
        .to_numpy(
            dtype=float
        )
    )

    if len(values) == 0:

        return pd.DataFrame(
            {
                "datetime": future_dates,
                "caudal_m3s": np.nan,
            }
        )

    base = float(
        values[-1]
    )

    if len(values) >= 4:

        x = np.arange(
            len(values)
        )

        slope = float(
            np.polyfit(
                x,
                values,
                1,
            )[0]
        )

    else:

        slope = 0.0

    # Evitar una extrapolación excesiva
    max_daily_change = max(
        abs(base) * 0.03,
        1.0,
    )

    slope = float(
        np.clip(
            slope,
            -max_daily_change,
            max_daily_change,
        )
    )

    predictions = []

    for h in range(
        1,
        len(future_dates) + 1,
    ):

        # La tendencia pierde peso con el horizonte
        damping = np.exp(
            -h / 10.0
        )

        value = (
            base
            + slope
            * h
            * damping
        )

        value = max(
            value,
            0.0,
        )

        predictions.append(
            value
        )

    return pd.DataFrame(
        {
            "datetime": future_dates,
            "caudal_m3s": predictions,
        }
    )


# ============================================================
# OBTENER VARIABLES EXÓGENAS
# ============================================================

def get_exogenous_data(
    start,
    end,
    forecast_days=15,
):

    rain_history = get_rain_history(
        start,
        end,
    )

    rain_future = get_rain_forecast(
        forecast_days
    )

    caudal_history, caudal_info = (
        get_caudal_history(
            start,
            end,
        )
    )

    # ========================================================
    # HISTÓRICO
    # ========================================================

    history = rain_history.copy()

    if history.empty:

        history = pd.DataFrame(
            columns=[
                "datetime",
                "precip_mm",
            ]
        )

    if not caudal_history.empty:

        history = history.merge(
            caudal_history,
            on="datetime",
            how="outer",
        )

    if "caudal_m3s" not in history.columns:

        history[
            "caudal_m3s"
        ] = np.nan

    history = (
        history
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # ========================================================
    # FUTURO
    # ========================================================

    future = rain_future.copy()

    if future.empty:

        future_dates = pd.date_range(
            pd.Timestamp.today().normalize()
            + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D",
        )

        future = pd.DataFrame(
            {
                "datetime": future_dates,
                "precip_mm": 0.0,
            }
        )

    caudal_future = project_caudal(
        caudal_history,
        future["datetime"],
    )

    future = future.merge(
        caudal_future,
        on="datetime",
        how="left",
    )

    meta = {
        "rain_points": list(
            RAIN_POINTS.keys()
        ),
        "caudal_series": caudal_info,
        "rain_source": "Open-Meteo",
        "caudal_source": "INA",
        "caudal_projection": (
            "Tendencia reciente amortiguada"
        ),
    }

    return (
        history,
        future,
        meta,
    )
