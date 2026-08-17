import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURACIÓN
# ============================================================

FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

HISTORICAL_WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

INA_SERIES_URL = (
    "https://alerta.ina.gob.ar/pub/datos/series"
)

INA_A5_URL = (
    "https://alerta.ina.gob.ar/a5/getObservaciones"
)


# ============================================================
# PARÁMETROS GENERALES
# ============================================================

MAX_FORECAST_DAYS = 30

WEATHER_FORECAST_DAYS = 16

CLIMATOLOGY_LOOKBACK_YEARS = 15

CAUDAL_TREND_WINDOW = 14


# ============================================================
# PUNTOS REPRESENTATIVOS DEL CORREDOR DEL PARANÁ
# ============================================================

RAIN_POINTS = {

    "Corrientes": (
        -27.4692,
        -58.8306,
    ),

    "Goya": (
        -29.1399,
        -59.2634,
    ),

    "La Paz": (
        -30.7449,
        -59.6457,
    ),

    "Paraná": (
        -31.7413,
        -60.5115,
    ),

    "Rosario": (
        -32.9442,
        -60.6505,
    ),

    "San Nicolás": (
        -33.3358,
        -60.2252,
    ),
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
# CACHE
# ============================================================

_SERIES_CATALOG_CACHE = None


# ============================================================
# REQUEST
# ============================================================

def _request_json(
    url,
    params=None,
    timeout=60,
):

    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={
            "User-Agent":
                "Parana-San-Nicolas-V11/1.0",

            "Accept":
                "application/json,text/plain,*/*",
        },
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# UTILIDADES DE FECHA
# ============================================================

def _normalize_date(
    value,
):

    dt = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(
        dt
    ):

        return pd.NaT

    return (
        dt
        .tz_localize(
            None
        )
        .normalize()
    )


def _normalize_series_dates(
    serie,
):

    return (
        pd.to_datetime(
            serie,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(
            None
        )
        .dt
        .normalize()
    )


# ============================================================
# HISTÓRICO DE PRECIPITACIÓN DEL CORREDOR
# ============================================================

def get_rain_history(
    start,
    end,
):

    start_dt = pd.to_datetime(
        start,
        errors="coerce",
    )

    end_dt = pd.to_datetime(
        end,
        errors="coerce",
    )

    today = (
        pd.Timestamp.today()
        .normalize()
    )

    if pd.isna(
        start_dt
    ):

        start_dt = (
            today
            - pd.Timedelta(
                days=365
            )
        )

    if pd.isna(
        end_dt
    ):

        end_dt = today

    end_dt = min(
        end_dt.normalize(),
        today,
    )

    start_dt = min(
        start_dt.normalize(),
        end_dt,
    )

    frames = []

    for (
        name,
        (
            lat,
            lon,
        ),
    ) in RAIN_POINTS.items():

        params = {

            "latitude":
                lat,

            "longitude":
                lon,

            "start_date":
                start_dt.strftime(
                    "%Y-%m-%d"
                ),

            "end_date":
                end_dt.strftime(
                    "%Y-%m-%d"
                ),

            "daily":
                "precipitation_sum",

            "timezone":
                "America/Argentina/Buenos_Aires",
        }

        try:

            data = _request_json(
                HISTORICAL_WEATHER_URL,
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

            if (
                not times
                or not rain
            ):

                continue

            frame = pd.DataFrame(
                {
                    "datetime":
                        pd.to_datetime(
                            times,
                            errors="coerce",
                        ),

                    f"rain_{name}":
                        pd.to_numeric(
                            rain,
                            errors="coerce",
                        ),
                }
            )

            frame[
                f"rain_{name}"
            ] = (
                frame[
                    f"rain_{name}"
                ]
                .clip(
                    lower=0.0
                )
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

    result = frames[
        0
    ]

    for frame in frames[
        1:
    ]:

        result = result.merge(
            frame,
            on="datetime",
            how="outer",
        )

    rain_cols = [
        c
        for c
        in result.columns
        if c.startswith(
            "rain_"
        )
    ]

    result[
        "precip_mm"
    ] = (
        result[
            rain_cols
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    result[
        "precip_mm"
    ] = (
        pd.to_numeric(
            result[
                "precip_mm"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
    )

    result[
        "datetime"
    ] = _normalize_series_dates(
        result[
            "datetime"
        ]
    )

    result = (
        result[
            [
                "datetime",
                "precip_mm",
            ]
        ]
        .dropna(
            subset=[
                "datetime"
            ]
        )
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=[
                "datetime"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# PRECIPITACIÓN FUTURA OPEN-METEO
# ============================================================

def get_rain_forecast(
    days=15,
):

    days = max(
        1,
        min(
            int(
                days
            ),
            WEATHER_FORECAST_DAYS,
        ),
    )

    frames = []

    for (
        name,
        (
            lat,
            lon,
        ),
    ) in RAIN_POINTS.items():

        params = {

            "latitude":
                lat,

            "longitude":
                lon,

            "daily":
                "precipitation_sum",

            "forecast_days":
                days,

            "timezone":
                "America/Argentina/Buenos_Aires",
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

            if (
                not times
                or not rain
            ):

                continue

            frame = pd.DataFrame(
                {
                    "datetime":
                        pd.to_datetime(
                            times,
                            errors="coerce",
                        ),

                    f"rain_{name}":
                        pd.to_numeric(
                            rain,
                            errors="coerce",
                        ),
                }
            )

            frame[
                f"rain_{name}"
            ] = (
                frame[
                    f"rain_{name}"
                ]
                .clip(
                    lower=0.0
                )
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
                "rain_source",
            ]
        )

    result = frames[
        0
    ]

    for frame in frames[
        1:
    ]:

        result = result.merge(
            frame,
            on="datetime",
            how="outer",
        )

    rain_cols = [
        c
        for c
        in result.columns
        if c.startswith(
            "rain_"
        )
    ]

    result[
        "precip_mm"
    ] = (
        result[
            rain_cols
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    result[
        "precip_mm"
    ] = (
        pd.to_numeric(
            result[
                "precip_mm"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
    )

    result[
        "datetime"
    ] = _normalize_series_dates(
        result[
            "datetime"
        ]
    )

    result[
        "rain_source"
    ] = "Open-Meteo forecast"

    return (
        result[
            [
                "datetime",
                "precip_mm",
                "rain_source",
            ]
        ]
        .dropna(
            subset=[
                "datetime"
            ]
        )
        .sort_values(
            "datetime"
        )
        .head(
            days
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CLIMATOLOGÍA DE LLUVIA
# ============================================================

def _build_rain_climatology(
    history,
):

    if (
        history is None
        or not isinstance(
            history,
            pd.DataFrame,
        )
        or history.empty
        or "datetime"
        not in history.columns
        or "precip_mm"
        not in history.columns
    ):

        return pd.DataFrame(
            columns=[
                "month",
                "day",
                "rain_climatology_mm",
            ]
        )

    x = history.copy()

    x[
        "datetime"
    ] = _normalize_series_dates(
        x[
            "datetime"
        ]
    )

    x[
        "precip_mm"
    ] = (
        pd.to_numeric(
            x[
                "precip_mm"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
    )

    x = x.dropna(
        subset=[
            "datetime"
        ]
    )

    if x.empty:

        return pd.DataFrame(
            columns=[
                "month",
                "day",
                "rain_climatology_mm",
            ]
        )

    x[
        "month"
    ] = x[
        "datetime"
    ].dt.month

    x[
        "day"
    ] = x[
        "datetime"
    ].dt.day

    clim = (
        x
        .groupby(
            [
                "month",
                "day",
            ],
            as_index=False,
        )[
            "precip_mm"
        ]
        .median()
        .rename(
            columns={
                "precip_mm":
                    "rain_climatology_mm"
            }
        )
    )

    return clim


# ============================================================
# EXTENDER LLUVIA HASTA 30 DÍAS
# ============================================================

def extend_rain_forecast(
    rain_forecast,
    rain_history,
    start_date,
    days,
):

    days = max(
        1,
        min(
            int(
                days
            ),
            MAX_FORECAST_DAYS,
        ),
    )

    start_date = _normalize_date(
        start_date
    )

    if pd.isna(
        start_date
    ):

        start_date = (
            pd.Timestamp.today()
            .normalize()
        )

    dates = pd.date_range(
        start_date,
        periods=days,
        freq="D",
    )

    result = pd.DataFrame(
        {
            "datetime":
                dates
        }
    )

    if (
        isinstance(
            rain_forecast,
            pd.DataFrame,
        )
        and not rain_forecast.empty
        and "datetime"
        in rain_forecast.columns
    ):

        forecast = (
            rain_forecast
            .copy()
        )

        forecast[
            "datetime"
        ] = _normalize_series_dates(
            forecast[
                "datetime"
            ]
        )

        if "precip_mm" not in forecast.columns:

            forecast[
                "precip_mm"
            ] = 0.0

        if "rain_source" not in forecast.columns:

            forecast[
                "rain_source"
            ] = "Open-Meteo forecast"

        forecast = forecast[
            [
                "datetime",
                "precip_mm",
                "rain_source",
            ]
        ].copy()

        result = result.merge(
            forecast,
            on="datetime",
            how="left",
        )

    else:

        result[
            "precip_mm"
        ] = np.nan

        result[
            "rain_source"
        ] = None

    climatology = _build_rain_climatology(
        rain_history
    )

    result[
        "month"
    ] = result[
        "datetime"
    ].dt.month

    result[
        "day"
    ] = result[
        "datetime"
    ].dt.day

    if not climatology.empty:

        result = result.merge(
            climatology,
            on=[
                "month",
                "day",
            ],
            how="left",
        )

    else:

        result[
            "rain_climatology_mm"
        ] = 0.0

    result[
        "rain_climatology_mm"
    ] = (
        pd.to_numeric(
            result[
                "rain_climatology_mm"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
    )

    missing = result[
        "precip_mm"
    ].isna()

    result.loc[
        missing,
        "precip_mm",
    ] = result.loc[
        missing,
        "rain_climatology_mm",
    ]

    result.loc[
        missing,
        "rain_source",
    ] = "Climatología histórica"

    result[
        "precip_mm"
    ] = (
        pd.to_numeric(
            result[
                "precip_mm"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
    )

    result[
        "rain_source"
    ] = result[
        "rain_source"
    ].fillna(
        "Climatología histórica"
    )

    return (
        result[
            [
                "datetime",
                "precip_mm",
                "rain_source",
            ]
        ]
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CATÁLOGO INA
# ============================================================

def _get_series_catalog():

    global _SERIES_CATALOG_CACHE

    if (
        _SERIES_CATALOG_CACHE
        is not None
    ):

        return (
            _SERIES_CATALOG_CACHE
        )

    try:

        data = _request_json(
            INA_SERIES_URL
        )

    except Exception:

        _SERIES_CATALOG_CACHE = []

        return []

    if isinstance(
        data,
        list,
    ):

        catalog = data

    elif isinstance(
        data,
        dict,
    ):

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
# BUSCAR MEJOR SERIE DE CAUDAL
# ============================================================

def find_best_caudal_series():

    catalog = (
        _get_series_catalog()
    )

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

        if varid != 4:

            continue

        station = str(
            row.get(
                "estacion_nombre",
                "",
            )
        ).strip()

        if (
            station
            not in
            CAUDAL_STATION_PRIORITY
        ):

            continue

        series_id = (
            row.get(
                "seriesid"
            )
            or row.get(
                "seriesId"
            )
        )

        if series_id is None:

            continue

        try:

            series_id = int(
                series_id
            )

        except Exception:

            continue

        try:

            procid = int(
                row.get(
                    "procid",
                    -1,
                )
            )

        except Exception:

            procid = -1

        try:

            obs_count = int(
                row.get(
                    "obs_count",
                    0,
                )
                or 0
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

        score = 0

        if procid == 1:

            score += 100

        elif procid == 2:

            score += 80

        if obs_count > 0:

            score += 40

        if obs_count > 100:

            score += 15

        if obs_count > 1000:

            score += 20

        if pd.notna(
            to_date
        ):

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

            elif age_days <= 365:

                score += 10

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

                "series_id":
                    series_id,

                "station":
                    station,

                "procid":
                    procid,

                "proc_name":
                    row.get(
                        "proc_nombre"
                    ),

                "unit":
                    row.get(
                        "unit_nombre"
                    ),

                "obs_count":
                    obs_count,

                "to_date":
                    row.get(
                        "to_date"
                    ),

                "score":
                    score,
            }
        )

    if not candidates:

        return None

    candidates = sorted(
        candidates,
        key=lambda x:
            x[
                "score"
            ],
        reverse=True,
    )

    return candidates[
        0
    ]


# ============================================================
# CAUDAL HISTÓRICO
# ============================================================

def get_caudal_history(
    start,
    end,
):

    info = (
        find_best_caudal_series()
    )

    empty = pd.DataFrame(
        columns=[
            "datetime",
            "caudal_m3s",
        ]
    )

    if info is None:

        return (
            empty,
            None,
        )

    params = {

        "tipo":
            "puntual",

        "series_id":
            info[
                "series_id"
            ],

        "timestart":
            str(
                start
            ),

        "timeend":
            str(
                end
            ),
    }

    try:

        data = _request_json(
            INA_A5_URL,
            params=params,
        )

    except Exception:

        return (
            empty,
            info,
        )

    if not isinstance(
        data,
        list,
    ):

        return (
            empty,
            info,
        )

    df = pd.DataFrame(
        data
    )

    if df.empty:

        return (
            empty,
            info,
        )

    if (
        "timestart"
        not in df.columns
        or "valor"
        not in df.columns
    ):

        return (
            empty,
            info,
        )

    df[
        "datetime"
    ] = _normalize_series_dates(
        df[
            "timestart"
        ]
    )

    df[
        "caudal_m3s"
    ] = pd.to_numeric(
        df[
            "valor"
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "datetime",
            "caudal_m3s",
        ]
    )

    df = (
        df
        .groupby(
            "datetime",
            as_index=False,
        )[
            "caudal_m3s"
        ]
        .mean()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        df,
        info,
    )


# ============================================================
# PROYECCIÓN NORMAL DE CAUDAL
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
        or not isinstance(
            history,
            pd.DataFrame,
        )
        or history.empty
        or "caudal_m3s"
        not in history.columns
    ):

        return pd.DataFrame(
            {
                "datetime":
                    future_dates,

                "caudal_m3s":
                    np.nan,

                "caudal_source":
                    "Sin datos",
            }
        )

    values = (
        pd.to_numeric(
            history[
                "caudal_m3s"
            ],
            errors="coerce",
        )
        .dropna()
        .tail(
            CAUDAL_TREND_WINDOW
        )
        .to_numpy(
            dtype=float
        )
    )

    if len(
        values
    ) == 0:

        return pd.DataFrame(
            {
                "datetime":
                    future_dates,

                "caudal_m3s":
                    np.nan,

                "caudal_source":
                    "Sin datos",
            }
        )

    base = float(
        values[
            -1
        ]
    )

    if len(
        values
    ) >= 4:

        slope = float(
            np.polyfit(
                np.arange(
                    len(
                        values
                    )
                ),
                values,
                1,
            )[0]
        )

    else:

        slope = 0.0

    max_daily_change = max(
        abs(
            base
        )
        * 0.025,
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

    previous = base

    for h in range(
        1,
        len(
            future_dates
        )
        + 1,
    ):

        damping = np.exp(
            -h
            / 12.0
        )

        daily_delta = (
            slope
            * damping
        )

        daily_limit = max(
            abs(
                previous
            )
            * 0.025,
            1.0,
        )

        daily_delta = float(
            np.clip(
                daily_delta,
                -daily_limit,
                daily_limit,
            )
        )

        value = (
            previous
            + daily_delta
        )

        value = max(
            float(
                value
            ),
            0.0,
        )

        predictions.append(
            value
        )

        previous = value

    return pd.DataFrame(
        {

            "datetime":
                future_dates,

            "caudal_m3s":
                predictions,

            "caudal_source":
                "Proyección experimental por tendencia reciente",
        }
    )


# ============================================================
# VARIABLES EXÓGENAS COMPLETAS
# ============================================================

def get_exogenous_data(
    start,
    end,
    forecast_days=15,
):

    forecast_days = max(
        1,
        min(
            int(
                forecast_days
            ),
            MAX_FORECAST_DAYS,
        ),
    )

    start_dt = pd.to_datetime(
        start,
        errors="coerce",
    )

    end_dt = pd.to_datetime(
        end,
        errors="coerce",
    )

    today = (
        pd.Timestamp.today()
        .normalize()
    )

    if pd.isna(
        start_dt
    ):

        start_dt = (
            today
            - pd.Timedelta(
                days=120
            )
        )

    if pd.isna(
        end_dt
    ):

        end_dt = today

    historical_end = min(
        end_dt.normalize(),
        today,
    )

    historical_start = min(
        start_dt.normalize(),
        historical_end,
    )

    start_text = (
        historical_start
        .strftime(
            "%Y-%m-%d"
        )
    )

    end_text = (
        historical_end
        .strftime(
            "%Y-%m-%d"
        )
    )

    # ========================================================
    # LLUVIA HISTÓRICA
    # ========================================================

    rain_history = (
        get_rain_history(
            start_text,
            end_text,
        )
    )

    # ========================================================
    # HISTORIAL PARA CLIMATOLOGÍA
    # ========================================================

    climatology_start = (
        historical_end
        - pd.DateOffset(
            years=CLIMATOLOGY_LOOKBACK_YEARS
        )
    )

    climatology_start_text = (
        climatology_start
        .strftime(
            "%Y-%m-%d"
        )
    )

    try:

        rain_climatology_history = (
            get_rain_history(
                climatology_start_text,
                end_text,
            )
        )

    except Exception:

        rain_climatology_history = (
            rain_history.copy()
        )

    if (
        rain_climatology_history.empty
        and not rain_history.empty
    ):

        rain_climatology_history = (
            rain_history.copy()
        )

    # ========================================================
    # PRONÓSTICO METEOROLÓGICO
    # ========================================================

    weather_days = min(
        forecast_days,
        WEATHER_FORECAST_DAYS,
    )

    rain_forecast = (
        get_rain_forecast(
            weather_days
        )
    )

    # ========================================================
    # FECHAS FUTURAS
    # ========================================================

    future_start = (
        historical_end
        + pd.Timedelta(
            days=1
        )
    )

    if (
        historical_end
        >= today
        - pd.Timedelta(
            days=1
        )
    ):

        future_start = (
            today
            + pd.Timedelta(
                days=1
            )
        )

    # ========================================================
    # EXTENDER LLUVIA HASTA 30 DÍAS
    # ========================================================

    rain_future = (
        extend_rain_forecast(
            rain_forecast=rain_forecast,
            rain_history=rain_climatology_history,
            start_date=future_start,
            days=forecast_days,
        )
    )

    # ========================================================
    # CAUDAL HISTÓRICO
    # ========================================================

    (
        caudal_history,
        caudal_info,
    ) = get_caudal_history(
        start_text,
        end_text,
    )

    # ========================================================
    # UNIFICAR HISTÓRICO
    # ========================================================

    if not rain_history.empty:

        history = (
            rain_history
            .copy()
        )

    else:

        history = pd.DataFrame(
            {
                "datetime":
                    pd.date_range(
                        historical_start,
                        historical_end,
                        freq="D",
                    ),

                "precip_mm":
                    0.0,
            }
        )

    if not caudal_history.empty:

        history = history.merge(
            caudal_history,
            on="datetime",
            how="outer",
        )

    if "precip_mm" not in history.columns:

        history[
            "precip_mm"
        ] = 0.0

    if "caudal_m3s" not in history.columns:

        history[
            "caudal_m3s"
        ] = np.nan

    history[
        "datetime"
    ] = _normalize_series_dates(
        history[
            "datetime"
        ]
    )

    history[
        "precip_mm"
    ] = (
        pd.to_numeric(
            history[
                "precip_mm"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
    )

    history[
        "caudal_m3s"
    ] = pd.to_numeric(
        history[
            "caudal_m3s"
        ],
        errors="coerce",
    )

    if (
        history[
            "caudal_m3s"
        ]
        .notna()
        .sum()
        >= 7
    ):

        history[
            "caudal_m3s"
        ] = (
            history[
                "caudal_m3s"
            ]
            .interpolate(
                limit=5,
                limit_direction="both",
            )
        )

    history = (
        history
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=[
                "datetime"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # CAUDAL FUTURO
    # ========================================================

    caudal_future = (
        project_caudal(
            caudal_history,
            rain_future[
                "datetime"
            ],
        )
    )

    # ========================================================
    # UNIFICAR FUTURO
    # ========================================================

    future = rain_future.merge(
        caudal_future,
        on="datetime",
        how="left",
    )

    future[
        "precip_mm"
    ] = (
        pd.to_numeric(
            future[
                "precip_mm"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
    )

    future[
        "caudal_m3s"
    ] = pd.to_numeric(
        future[
            "caudal_m3s"
        ],
        errors="coerce",
    )

    future = (
        future
        .sort_values(
            "datetime"
        )
        .head(
            forecast_days
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # DIAGNÓSTICO
    # ========================================================

    rain_valid = int(
        history[
            "precip_mm"
        ]
        .notna()
        .sum()
    )

    rain_max = (
        float(
            history[
                "precip_mm"
            ].max()
        )
        if rain_valid
        else 0.0
    )

    q_valid = int(
        history[
            "caudal_m3s"
        ]
        .notna()
        .sum()
    )

    q_last = (
        float(
            history[
                "caudal_m3s"
            ]
            .dropna()
            .iloc[-1]
        )
        if q_valid
        else None
    )

    official_rain_days = int(
        (
            future[
                "rain_source"
            ]
            == "Open-Meteo forecast"
        ).sum()
    )

    climatology_rain_days = int(
        (
            future[
                "rain_source"
            ]
            == "Climatología histórica"
        ).sum()
    )

    # ========================================================
    # META
    # ========================================================

    meta = {

        "rain_points":
            list(
                RAIN_POINTS.keys()
            ),

        "caudal_series":
            caudal_info,

        "rain_source":
            "Open-Meteo",

        "caudal_source":
            "INA",

        "caudal_projection":
            "Tendencia reciente amortiguada",

        "historical_start":
            start_text,

        "historical_end":
            end_text,

        "rain_history_records":
            rain_valid,

        "rain_history_max_mm":
            rain_max,

        "caudal_history_records":
            q_valid,

        "caudal_last_m3s":
            q_last,

        "forecast_days":
            forecast_days,

        "official_rain_forecast_days":
            official_rain_days,

        "climatology_extension_days":
            climatology_rain_days,

        "climatology_lookback_years":
            CLIMATOLOGY_LOOKBACK_YEARS,
    }

    return (
        history,
        future,
        meta,
    )
