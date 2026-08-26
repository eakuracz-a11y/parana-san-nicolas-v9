import requests
import pandas as pd
import numpy as np


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/exogenous.py
# BASE V11 CORREGIDA
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

INA_DATA_URL = (
    "https://alerta.ina.gob.ar/pub/datos/datos"
)


# ============================================================
# PUNTOS DE LLUVIA
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
# PRIORIDAD DE ESTACIONES PARA CAUDAL
# ============================================================

CAUDAL_STATION_PRIORITY = [
    "San Nicolás",
    "Rosario",
    "Paraná",
    "La Paz",
    "Goya",
    "Corrientes",
]


_SERIES_CATALOG_CACHE = None


# ============================================================
# REQUEST JSON
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
# LLUVIA HISTÓRICA
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

    if pd.isna(start_dt):

        start_dt = (
            today
            - pd.Timedelta(
                days=365
            )
        )

    if pd.isna(end_dt):

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
            ] = frame[
                f"rain_{name}"
            ].clip(
                lower=0
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
    ] = pd.to_numeric(
        result[
            "precip_mm"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    result[
        "precip_mm"
    ] = result[
        "precip_mm"
    ].clip(
        lower=0
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
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# LLUVIA FUTURA
# ============================================================

def get_rain_forecast(
    days=15,
):

    days = max(
        1,
        min(
            int(days),
            16,
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
    ] = pd.to_numeric(
        result[
            "precip_mm"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    result[
        "precip_mm"
    ] = result[
        "precip_mm"
    ].clip(
        lower=0
    )

    return (
        result[
            [
                "datetime",
                "precip_mm",
            ]
        ]
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

        # Variable 4 = caudal
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

    return candidates[0]


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
        "timeStart":
            str(start),

        "timeEnd":
            str(end),

        "seriesId":
            info[
                "series_id"
            ],

        "format":
            "json",
    }

    try:

        data = _request_json(
            INA_DATA_URL,
            params=params,
        )

    except Exception:

        return (
            empty,
            info,
        )

    if isinstance(
        data,
        dict,
    ):

        if "data" in data:

            data = data[
                "data"
            ]

        elif "results" in data:

            data = data[
                "results"
            ]

        elif "observaciones" in data:

            data = data[
                "observaciones"
            ]

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

    fecha_col = None
    valor_col = None

    for col in [
        "timestart",
        "timeStart",
        "datetime",
        "fecha",
        "date",
        "timestamp",
    ]:

        if col in df.columns:

            fecha_col = col
            break

    for col in [
        "valor",
        "value",
        "caudal",
    ]:

        if col in df.columns:

            valor_col = col
            break

    if (
        fecha_col is None
        or valor_col is None
    ):

        return (
            empty,
            info,
        )

    out = pd.DataFrame()

    out[
        "datetime"
    ] = pd.to_datetime(
        df[
            fecha_col
        ],
        errors="coerce",
        utc=True,
    )

    out[
        "datetime"
    ] = (
        out[
            "datetime"
        ]
        .dt.tz_localize(
            None
        )
        .dt.normalize()
    )

    out[
        "caudal_m3s"
    ] = pd.to_numeric(
        df[
            valor_col
        ],
        errors="coerce",
    )

    out = out.dropna(
        subset=[
            "datetime",
            "caudal_m3s",
        ]
    )

    out = (
        out
        .groupby(
            "datetime",
            as_index=False,
        )[
            "caudal_m3s"
        ]
        .mean()
    )

    return (
        out,
        info,
    )


# ============================================================
# CAUDAL FUTURO
# ============================================================

def build_caudal_future(
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
        or "caudal_m3s"
        not in history.columns
    ):

        return pd.DataFrame(
            {
                "datetime":
                    future_dates,

                "caudal_m3s":
                    np.nan,
            }
        )

    values = (
        history[
            "caudal_m3s"
        ]
        .dropna()
        .tail(
            14
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
        * 0.03,
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
        len(
            future_dates
        )
        + 1,
    ):

        damping = np.exp(
            -h
            / 10.0
        )

        value = (
            base
            + slope
            * h
            * damping
        )

        predictions.append(
            max(
                float(
                    value
                ),
                0.0,
            )
        )

    return pd.DataFrame(
        {
            "datetime":
                future_dates,

            "caudal_m3s":
                predictions,
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
        historical_start.strftime(
            "%Y-%m-%d"
        )
    )

    end_text = (
        historical_end.strftime(
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
    # LLUVIA FUTURA
    # ========================================================

    rain_future = (
        get_rain_forecast(
            forecast_days
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

    history = (
        rain_history.copy()
    )

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

    if (
        "precip_mm"
        not in history.columns
    ):

        history[
            "precip_mm"
        ] = 0.0

    if (
        "caudal_m3s"
        not in history.columns
    ):

        history[
            "caudal_m3s"
        ] = np.nan

    history[
        "precip_mm"
    ] = pd.to_numeric(
        history[
            "precip_mm"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    history[
        "caudal_m3s"
    ] = pd.to_numeric(
        history[
            "caudal_m3s"
        ],
        errors="coerce",
    )

    history = (
        history
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # FUTURO
    # ========================================================

    future = (
        rain_future.copy()
    )

    if future.empty:

        future_dates = pd.date_range(
            today
            + pd.Timedelta(
                days=1
            ),
            periods=
                forecast_days,
            freq="D",
        )

        future = pd.DataFrame(
            {
                "datetime":
                    future_dates,

                "precip_mm":
                    0.0,
            }
        )

    future[
        "datetime"
    ] = pd.to_datetime(
        future[
            "datetime"
        ],
        errors="coerce",
    )

    future[
        "precip_mm"
    ] = pd.to_numeric(
        future[
            "precip_mm"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    q_future = (
        build_caudal_future(
            caudal_history,
            future[
                "datetime"
            ],
        )
    )

    future = future.merge(
        q_future,
        on="datetime",
        how="left",
    )

    future = (
        future
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # METADATA
    # ========================================================

    meta = {
        "caudal":
            caudal_info,

        "rain_points":
            list(
                RAIN_POINTS.keys()
            ),
    }

    return (
        history,
        future,
        meta,
    )
