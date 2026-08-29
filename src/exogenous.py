# ============================================================
# PARANÁ · SAN NICOLÁS
# src/exogenous.py
# V11.11 COMPLETO
#
# MOTOR EXÓGENO HIDROLÓGICO MULTIESTACIÓN
#
# OBJETIVOS V11.11
# ------------------------------------------------------------
# 1. Lluvias históricas y pronosticadas para todo el corredor.
#
# 2. Caudales INA A5 var_id = 4.
#
# 3. Validación estricta de series del Paraná troncal.
#
# 4. Reconstrucción controlada de caudales faltantes:
#
#       observado
#       interpolado
#       estimado_vecinos
#       estimado_ratio_historico
#
# 5. Mantener separada la procedencia del valor:
#
#       q_parana
#       q_parana_source
#       q_parana_quality
#
# 6. Generar señales para el modelo:
#
#       lluvia 3 / 7 / 15 / 30 días
#       caudal cambios 1 / 3 / 7 / 14 días
#       medias 3 / 7 / 14 / 30
#       tendencia
#       caudal relativo
#
# 7. Generar índice de presión hidrológica del corredor.
#
# 8. Mantener compatibilidad:
#
#       precip_mm
#       caudal_m3s
#
# API:
#
# get_exogenous_data(start, end, forecast_days=60)
#
# retorna:
#
# history, future, metadata
#
# ============================================================


from functools import lru_cache
import re
import unicodedata

import numpy as np
import pandas as pd
import requests


# ============================================================
# VERSIÓN
# ============================================================

VERSION = "V11.11"


# ============================================================
# CONFIGURACIÓN
# ============================================================

REQUEST_TIMEOUT = 45

MAX_FORECAST_DAYS = 60

OPEN_METEO_REAL_FORECAST_DAYS = 16

FLOW_VALIDATION_DAYS = 180

FLOW_HISTORY_BLOCK_YEARS = 5

MIN_FLOW_OBSERVATIONS = 3

MIN_RATIO_OVERLAP = 15

SHORT_INTERPOLATION_LIMIT = 5


# ============================================================
# LÍMITES HIDROLÓGICOS DEL PARANÁ TRONCAL
# ============================================================

PARANA_TRUNK_MIN_MEDIAN_FLOW = 500.0

PARANA_TRUNK_MIN_RECENT_FLOW = 250.0

PARANA_TRUNK_MAX_FLOW = 100000.0


# ============================================================
# CONTROL DE ESTIMACIONES
# ============================================================

MIN_ESTIMATED_FLOW = 300.0

MAX_ESTIMATED_FLOW = 100000.0

MAX_NEIGHBOR_RATIO = 8.0

MIN_NEIGHBOR_RATIO = 0.10


# ============================================================
# OPEN METEO
# ============================================================

OPEN_METEO_ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

OPEN_METEO_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ============================================================
# INA A5
# ============================================================

INA_A5_BASE_URL = "https://alerta.ina.gob.ar/a5"

INA_SERIES_GEOJSON_URL = (
    INA_A5_BASE_URL
    + "/obs/puntual/series"
)

INA_OBSERVATIONS_URL = (
    INA_A5_BASE_URL
    + "/getObservaciones"
)

VAR_ID_CAUDAL = 4


# ============================================================
# ESTACIONES DEL CORREDOR
# ============================================================

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


# ============================================================
# ORDEN HIDROLÓGICO
# ============================================================

STATION_ORDER = {
    "Corrientes": 0,
    "Goya": 1,
    "La Paz": 2,
    "Paraná": 3,
    "Diamante": 4,
    "Rosario": 5,
    "Villa Constitución": 6,
    "San Nicolás": 7,
}


# ============================================================
# COORDENADAS
# ============================================================

RAIN_POINTS = {

    "Corrientes": {
        "lat": -27.4692,
        "lon": -58.8306,
    },

    "Goya": {
        "lat": -29.1400,
        "lon": -59.2634,
    },

    "La Paz": {
        "lat": -30.7449,
        "lon": -59.6457,
    },

    "Paraná": {
        "lat": -31.7413,
        "lon": -60.5115,
    },

    "Diamante": {
        "lat": -32.0665,
        "lon": -60.6384,
    },

    "Rosario": {
        "lat": -32.9442,
        "lon": -60.6505,
    },

    "Villa Constitución": {
        "lat": -33.2272,
        "lon": -60.3297,
    },

    "San Nicolás": {
        "lat": -33.3335,
        "lon": -60.2110,
    },
}


# ============================================================
# COLUMNAS
# ============================================================

RAIN_COLUMNS = {

    "Corrientes": "rain_corrientes",

    "Goya": "rain_goya",

    "La Paz": "rain_la_paz",

    "Paraná": "rain_parana",

    "Diamante": "rain_diamante",

    "Rosario": "rain_rosario",

    "Villa Constitución":
        "rain_villa_constitucion",

    "San Nicolás":
        "rain_san_nicolas",
}


FLOW_COLUMNS = {

    "Corrientes": "q_corrientes",

    "Goya": "q_goya",

    "La Paz": "q_la_paz",

    "Paraná": "q_parana",

    "Diamante": "q_diamante",

    "Rosario": "q_rosario",

    "Villa Constitución":
        "q_villa_constitucion",

    "San Nicolás":
        "q_san_nicolas",
}


# ============================================================
# PRIORIDAD PARA CAUDAL PRINCIPAL
# ============================================================

FLOW_PRIORITY = [
    "San Nicolás",
    "Villa Constitución",
    "Rosario",
    "Diamante",
    "Paraná",
    "La Paz",
    "Goya",
    "Corrientes",
]


# ============================================================
# ALIAS
# ============================================================

STATION_ALIASES = {

    "Corrientes": [
        "corrientes",
        "puerto corrientes",
    ],

    "Goya": [
        "goya",
        "puerto goya",
    ],

    "La Paz": [
        "la paz",
        "puerto la paz",
    ],

    "Paraná": [
        "parana",
        "paraná",
        "puerto parana",
        "puerto paraná",
    ],

    "Diamante": [
        "diamante",
        "puerto diamante",
    ],

    "Rosario": [
        "rosario",
        "puerto rosario",
    ],

    "Villa Constitución": [
        "villa constitucion",
        "villa constitución",
        "puerto villa constitucion",
        "puerto villa constitución",
    ],

    "San Nicolás": [
        "san nicolas",
        "san nicolás",
        "puerto san nicolas",
        "puerto san nicolás",
    ],
}


# ============================================================
# TÉRMINOS A PENALIZAR
# ============================================================

NON_TRUNK_TERMS = [
    "arroyo",
    "canal",
    "riacho",
    "laguna",
    "tributario",
    "afluente",
    "brazo",
    "salado",
    "carcarana",
    "carcaraña",
    "uruguay",
    "paraguay",
    "bermejo",
    "pilcomayo",
    "iguazu",
    "iguazú",
]


BAD_STATION_TERMS = [
    "meteorologica",
    "meteorologico",
    "agrometeorologica",
    "aeropuerto",
    "inta",
    "escuela",
    "pluviometrica",
    "precipitacion",
    "lluvia",
]


# ============================================================
# UTILIDADES
# ============================================================

def _normalize_text(value):

    if value is None:
        return ""

    text = str(value).lower().strip()

    text = unicodedata.normalize(
        "NFD",
        text,
    )

    text = "".join(
        ch
        for ch in text
        if unicodedata.category(ch)
        != "Mn"
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _safe_float(
    value,
    default=np.nan,
):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def _safe_int(
    value,
    default=0,
):

    try:

        return int(
            float(value)
        )

    except Exception:

        return default


def _normalize_date(value):

    dt = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(dt):

        raise ValueError(
            f"Fecha inválida: {value}"
        )

    return dt.strftime(
        "%Y-%m-%d"
    )


def _to_naive_datetime(values):

    return (
        pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
    )


def _daily_range(
    start,
    end,
):

    start = pd.to_datetime(
        start
    ).normalize()

    end = pd.to_datetime(
        end
    ).normalize()

    if end < start:

        return pd.DatetimeIndex([])

    return pd.date_range(
        start=start,
        end=end,
        freq="D",
    )


# ============================================================
# LLUVIA HISTÓRICA POR ESTACIÓN
# ============================================================

def _get_rain_history_station(
    station,
    start,
    end,
):

    point = RAIN_POINTS[
        station
    ]

    params = {

        "latitude":
            point["lat"],

        "longitude":
            point["lon"],

        "start_date":
            _normalize_date(start),

        "end_date":
            _normalize_date(end),

        "daily":
            "precipitation_sum",

        "timezone":
            "UTC",
    }

    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    daily = data.get(
        "daily",
        {}
    )

    times = daily.get(
        "time",
        []
    )

    precipitation = daily.get(
        "precipitation_sum",
        []
    )

    if not times:

        return pd.DataFrame()

    col = RAIN_COLUMNS[
        station
    ]

    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    times,
                    errors="coerce",
                ),

            col:
                pd.to_numeric(
                    pd.Series(
                        precipitation
                    ),
                    errors="coerce",
                ),
        }
    )

    result[
        "datetime"
    ] = _to_naive_datetime(
        result[
            "datetime"
        ]
    )

    result[
        "datetime"
    ] = (
        result[
            "datetime"
        ]
        .dt
        .normalize()
    )

    result[
        col
    ] = (
        pd.to_numeric(
            result[col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    return (
        result
        .dropna(
            subset=["datetime"]
        )
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LLUVIA HISTÓRICA DEL CORREDOR
# ============================================================

def get_rain_history(
    start,
    end,
):

    base = pd.DataFrame(
        {
            "datetime":
                _daily_range(
                    start,
                    end,
                )
        }
    )

    metadata = {

        "source":
            "Open-Meteo",

        "stations":
            {},

        "available_stations":
            [],
    }

    for station in STATIONS:

        col = RAIN_COLUMNS[
            station
        ]

        try:

            station_df = (
                _get_rain_history_station(
                    station,
                    start,
                    end,
                )
            )

            if station_df.empty:

                base[col] = np.nan

                metadata[
                    "stations"
                ][station] = {
                    "status":
                        "sin_datos",

                    "records":
                        0,
                }

                continue

            base = base.merge(
                station_df,
                on="datetime",
                how="left",
            )

            count = int(
                pd.to_numeric(
                    base[col],
                    errors="coerce",
                )
                .notna()
                .sum()
            )

            metadata[
                "stations"
            ][station] = {

                "status":
                    "ok",

                "records":
                    count,
            }

            if count > 0:

                metadata[
                    "available_stations"
                ].append(
                    station
                )

        except Exception as exc:

            if col not in base.columns:

                base[col] = np.nan

            metadata[
                "stations"
            ][station] = {

                "status":
                    "error",

                "records":
                    0,

                "error":
                    str(exc),
            }

    rain_cols = [
        RAIN_COLUMNS[station]
        for station in STATIONS
        if (
            RAIN_COLUMNS[station]
            in base.columns
        )
    ]

    if rain_cols:

        base[
            "precip_mm"
        ] = (
            base[
                rain_cols
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .mean(
                axis=1,
                skipna=True,
            )
        )

    else:

        base[
            "precip_mm"
        ] = np.nan

    return (
        base,
        metadata,
    )


# ============================================================
# PRONÓSTICO DE LLUVIA POR ESTACIÓN
# ============================================================

def _get_rain_forecast_station(
    station,
):

    point = RAIN_POINTS[
        station
    ]

    params = {

        "latitude":
            point["lat"],

        "longitude":
            point["lon"],

        "daily":
            "precipitation_sum",

        "forecast_days":
            OPEN_METEO_REAL_FORECAST_DAYS,

        "timezone":
            "UTC",
    }

    response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    daily = data.get(
        "daily",
        {}
    )

    times = daily.get(
        "time",
        []
    )

    precipitation = daily.get(
        "precipitation_sum",
        []
    )

    if not times:

        return pd.DataFrame()

    col = RAIN_COLUMNS[
        station
    ]

    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    times,
                    errors="coerce",
                ),

            col:
                pd.to_numeric(
                    pd.Series(
                        precipitation
                    ),
                    errors="coerce",
                ),
        }
    )

    result[
        "datetime"
    ] = _to_naive_datetime(
        result["datetime"]
    )

    result[
        "datetime"
    ] = (
        result[
            "datetime"
        ]
        .dt
        .normalize()
    )

    result[
        col
    ] = (
        pd.to_numeric(
            result[col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    return (
        result
        .dropna(
            subset=["datetime"]
        )
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PRONÓSTICO DE LLUVIA
# ============================================================

def get_rain_forecast(
    start_date,
    forecast_days=60,
):

    forecast_days = min(
        max(
            int(forecast_days),
            1,
        ),
        MAX_FORECAST_DAYS,
    )

    start_date = pd.to_datetime(
        start_date
    ).normalize()

    dates = pd.date_range(

        start=
            start_date
            + pd.Timedelta(days=1),

        periods=
            forecast_days,

        freq="D",
    )

    base = pd.DataFrame(
        {
            "datetime":
                dates
        }
    )

    metadata = {

        "source":
            "Open-Meteo",

        "real_forecast_days":
            OPEN_METEO_REAL_FORECAST_DAYS,

        "stations":
            {},

        "available_stations":
            [],
    }

    for station in STATIONS:

        col = RAIN_COLUMNS[
            station
        ]

        try:

            station_df = (
                _get_rain_forecast_station(
                    station
                )
            )

            if station_df.empty:

                base[col] = 0.0

                metadata[
                    "stations"
                ][station] = {

                    "status":
                        "sin_datos",

                    "records":
                        0,
                }

                continue

            base = base.merge(
                station_df,
                on="datetime",
                how="left",
            )

            real_count = int(
                pd.to_numeric(
                    base[col],
                    errors="coerce",
                )
                .notna()
                .sum()
            )

            metadata[
                "stations"
            ][station] = {

                "status":
                    (
                        "ok"
                        if real_count > 0
                        else "sin_datos"
                    ),

                "records":
                    real_count,
            }

            if real_count > 0:

                metadata[
                    "available_stations"
                ].append(
                    station
                )

            base[col] = (
                pd.to_numeric(
                    base[col],
                    errors="coerce",
                )
                .fillna(0.0)
            )

        except Exception as exc:

            base[col] = 0.0

            metadata[
                "stations"
            ][station] = {

                "status":
                    "error",

                "records":
                    0,

                "error":
                    str(exc),
            }

    rain_cols = [
        RAIN_COLUMNS[station]
        for station in STATIONS
    ]

    base[
        "precip_mm"
    ] = (
        base[
            rain_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .mean(
            axis=1,
            skipna=True,
        )
        .fillna(0.0)
    )

    return (
        base,
        metadata,
    )


# ============================================================
# CATÁLOGO INA
# ============================================================

@lru_cache(maxsize=1)
def get_ina_catalog():

    response = requests.get(

        INA_SERIES_GEOJSON_URL,

        params={
            "format":
                "geojson"
        },

        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    features = data.get(
        "features",
        []
    )

    rows = []

    for feature in features:

        if not isinstance(
            feature,
            dict,
        ):
            continue

        props = feature.get(
            "properties",
            {}
        )

        if not isinstance(
            props,
            dict,
        ):
            continue

        geometry = feature.get(
            "geometry",
            {}
        )

        coords = []

        if isinstance(
            geometry,
            dict,
        ):

            coords = (
                geometry.get(
                    "coordinates",
                    []
                )
                or []
            )

        lon = np.nan
        lat = np.nan

        if (
            isinstance(
                coords,
                list,
            )
            and len(coords) >= 2
        ):

            lon = _safe_float(
                coords[0]
            )

            lat = _safe_float(
                coords[1]
            )

        rows.append(
            {
                "series_id":
                    _safe_int(
                        props.get(
                            "series_id"
                        ),
                        0,
                    ),

                "nombre":
                    props.get(
                        "nombre"
                    ),

                "rio":
                    props.get(
                        "rio"
                    ),

                "var_id":
                    _safe_int(
                        props.get(
                            "var_id"
                        ),
                        0,
                    ),

                "proc_id":
                    _safe_int(
                        props.get(
                            "proc_id"
                        ),
                        0,
                    ),

                "unit_id":
                    props.get(
                        "unit_id"
                    ),

                "var_nombre":
                    props.get(
                        "var_nombre"
                    ),

                "timestart":
                    props.get(
                        "timestart"
                    ),

                "timeend":
                    props.get(
                        "timeend"
                    ),

                "count":
                    _safe_int(
                        props.get(
                            "count"
                        ),
                        0,
                    ),

                "fuente":
                    props.get(
                        "fuente"
                    ),

                "longitude":
                    lon,

                "latitude":
                    lat,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        raise RuntimeError(
            "El catálogo INA A5 no devolvió series."
        )

    return result


# ============================================================
# COINCIDENCIA DE ESTACIÓN
# ============================================================

def _station_match_score(
    station,
    name,
):

    name_norm = _normalize_text(
        name
    )

    if not name_norm:

        return -1000

    aliases = STATION_ALIASES.get(
        station,
        [station],
    )

    score = -1000

    for alias in aliases:

        alias_norm = _normalize_text(
            alias
        )

        if not alias_norm:
            continue

        if name_norm == alias_norm:

            score = max(
                score,
                150,
            )

        elif name_norm.startswith(
            alias_norm
        ):

            score = max(
                score,
                125,
            )

        elif re.search(
            r"\b"
            + re.escape(alias_norm)
            + r"\b",
            name_norm,
        ):

            score = max(
                score,
                100,
            )

        elif alias_norm in name_norm:

            score = max(
                score,
                70,
            )

    for bad in BAD_STATION_TERMS:

        if bad in name_norm:

            score -= 150

    return score


# ============================================================
# VALIDACIÓN DEL RÍO
# ============================================================

def _river_score(
    river,
):

    river_norm = _normalize_text(
        river
    )

    if not river_norm:

        return 0

    score = 0

    if "parana" in river_norm:

        score += 100

    for term in NON_TRUNK_TERMS:

        if (
            _normalize_text(term)
            in river_norm
        ):

            score -= 150

    return score


def _is_parana_trunk_candidate(
    river,
    name=None,
):

    river_norm = _normalize_text(
        river
    )

    name_norm = _normalize_text(
        name
    )

    combined = (
        river_norm
        + " "
        + name_norm
    )

    for term in NON_TRUNK_TERMS:

        if (
            _normalize_text(term)
            in combined
        ):

            return False

    if river_norm:

        return (
            "parana"
            in river_norm
        )

    return True


# ============================================================
# CANDIDATOS DE CAUDAL
# ============================================================

def candidatos_caudal_estacion(
    station,
    start=None,
    end=None,
):

    catalog = get_ina_catalog()

    candidates = catalog[
        catalog[
            "var_id"
        ]
        == VAR_ID_CAUDAL
    ].copy()

    if candidates.empty:

        return pd.DataFrame()

    candidates[
        "station_score"
    ] = candidates[
        "nombre"
    ].apply(
        lambda name:
            _station_match_score(
                station,
                name,
            )
    )

    candidates = candidates[
        candidates[
            "station_score"
        ]
        > 0
    ].copy()

    if candidates.empty:

        return pd.DataFrame()

    candidates[
        "river_score"
    ] = candidates[
        "rio"
    ].apply(
        _river_score
    )

    candidates[
        "is_parana_trunk"
    ] = candidates.apply(
        lambda row:
            _is_parana_trunk_candidate(
                row.get("rio"),
                row.get("nombre"),
            ),
        axis=1,
    )

    candidates[
        "trunk_score"
    ] = np.where(
        candidates[
            "is_parana_trunk"
        ],
        70,
        -150,
    )

    candidates[
        "count_score"
    ] = (
        np.log10(
            pd.to_numeric(
                candidates["count"],
                errors="coerce",
            )
            .fillna(0)
            .clip(lower=0)
            + 1
        )
        * 7.0
    )

    candidates[
        "score"
    ] = (
        candidates[
            "station_score"
        ]
        +
        candidates[
            "river_score"
        ]
        +
        candidates[
            "trunk_score"
        ]
        +
        candidates[
            "count_score"
        ]
    )

    return (
        candidates
        .sort_values(
            [
                "score",
                "count",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PARSER INA
# ============================================================

def _extract_records(obj):

    records = []

    if isinstance(
        obj,
        list,
    ):

        for item in obj:

            if isinstance(
                item,
                dict,
            ):

                keys = {
                    str(k).lower()
                    for k in item.keys()
                }

                date_keys = {
                    "timestart",
                    "datetime",
                    "timestamp",
                    "fecha",
                    "time",
                    "date",
                }

                value_keys = {
                    "valor",
                    "value",
                    "val",
                    "dato",
                    "obs",
                }

                if (
                    keys.intersection(
                        date_keys
                    )
                    and
                    keys.intersection(
                        value_keys
                    )
                ):

                    records.append(
                        item
                    )

                else:

                    records.extend(
                        _extract_records(
                            item
                        )
                    )

    elif isinstance(
        obj,
        dict,
    ):

        for value in obj.values():

            if isinstance(
                value,
                (dict, list),
            ):

                records.extend(
                    _extract_records(
                        value
                    )
                )

    return records


def _record_datetime(
    record,
):

    for key in [
        "timestart",
        "datetime",
        "timestamp",
        "fecha",
        "time",
        "date",
    ]:

        if key not in record:
            continue

        dt = pd.to_datetime(
            record.get(key),
            errors="coerce",
            utc=True,
        )

        if pd.isna(dt):
            continue

        try:

            return dt.tz_localize(
                None
            )

        except Exception:

            try:

                return dt.tz_convert(
                    None
                )

            except Exception:

                return dt

    return pd.NaT


def _record_value(
    record,
):

    for key in [
        "valor",
        "value",
        "val",
        "dato",
        "obs",
    ]:

        if key not in record:
            continue

        value = _safe_float(
            record.get(key)
        )

        if np.isfinite(value):

            return value

    return np.nan


# ============================================================
# CONSULTA INA A5
# ============================================================

def query_caudal_series(
    series_id,
    start,
    end,
):

    params = {

        "tipo":
            "puntual",

        "series_id":
            int(series_id),

        "timestart":
            _normalize_date(start),

        "timeend":
            _normalize_date(end),
    }

    response = requests.get(

        INA_OBSERVATIONS_URL,

        params=params,

        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    records = _extract_records(
        payload
    )

    rows = []

    for record in records:

        dt = _record_datetime(
            record
        )

        value = _record_value(
            record
        )

        if pd.isna(dt):
            continue

        if not np.isfinite(value):
            continue

        if value < 0:
            continue

        if value > PARANA_TRUNK_MAX_FLOW:
            continue

        rows.append(
            {
                "datetime":
                    dt,

                "value":
                    value,
            }
        )

    if not rows:

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "datetime"
    ] = _to_naive_datetime(
        result[
            "datetime"
        ]
    )

    result[
        "value"
    ] = pd.to_numeric(
        result[
            "value"
        ],
        errors="coerce",
    )

    return (
        result
        .dropna(
            subset=[
                "datetime",
                "value",
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


# ============================================================
# VALIDAR CAUDAL
# ============================================================

def _flow_statistics(
    df,
):

    if (
        df is None
        or df.empty
    ):

        return {
            "records": 0,
            "median": np.nan,
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "last": np.nan,
        }

    values = (
        pd.to_numeric(
            df["value"],
            errors="coerce",
        )
        .dropna()
    )

    if values.empty:

        return {
            "records": 0,
            "median": np.nan,
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "last": np.nan,
        }

    return {

        "records":
            int(len(values)),

        "median":
            float(
                values.median()
            ),

        "mean":
            float(
                values.mean()
            ),

        "min":
            float(
                values.min()
            ),

        "max":
            float(
                values.max()
            ),

        "last":
            float(
                values.iloc[-1]
            ),
    }


def _validate_trunk_flow_series(
    candidate,
    data,
):

    stats = _flow_statistics(
        data
    )

    reasons = []

    valid = True

    if (
        stats["records"]
        < MIN_FLOW_OBSERVATIONS
    ):

        valid = False
        reasons.append(
            "pocas_observaciones"
        )

    if not _is_parana_trunk_candidate(
        candidate.get("rio"),
        candidate.get("nombre"),
    ):

        valid = False
        reasons.append(
            "no_parana_troncal"
        )

    median = stats[
        "median"
    ]

    last = stats[
        "last"
    ]

    if (
        not np.isfinite(median)
        or median
        < PARANA_TRUNK_MIN_MEDIAN_FLOW
    ):

        valid = False

        reasons.append(
            "mediana_demasiado_baja"
        )

    if (
        np.isfinite(last)
        and last
        < PARANA_TRUNK_MIN_RECENT_FLOW
    ):

        valid = False

        reasons.append(
            "caudal_actual_demasiado_bajo"
        )

    return {

        "valid":
            bool(valid),

        "reasons":
            reasons,

        **stats,
    }


# ============================================================
# SELECCIONAR SERIE
# ============================================================

def seleccionar_serie_caudal(
    station,
    start,
    end,
):

    candidates = (
        candidatos_caudal_estacion(
            station,
            start,
            end,
        )
    )

    metadata = {

        "station":
            station,

        "status":
            "sin_serie",

        "candidate_count":
            int(
                len(candidates)
            ),

        "tested":
            [],
    }

    if candidates.empty:

        return (
            None,
            metadata,
        )

    end_dt = pd.to_datetime(
        end
    ).normalize()

    test_start = (
        end_dt
        - pd.Timedelta(
            days=FLOW_VALIDATION_DAYS
        )
    )

    for _, candidate in (
        candidates
        .head(25)
        .iterrows()
    ):

        series_id = _safe_int(
            candidate.get(
                "series_id"
            ),
            0,
        )

        if series_id <= 0:
            continue

        try:

            test_data = (
                query_caudal_series(
                    series_id,
                    test_start,
                    end_dt,
                )
            )

            validation = (
                _validate_trunk_flow_series(
                    candidate,
                    test_data,
                )
            )

            metadata[
                "tested"
            ].append(
                {
                    "series_id":
                        series_id,

                    "nombre":
                        candidate.get(
                            "nombre"
                        ),

                    "rio":
                        candidate.get(
                            "rio"
                        ),

                    **validation,
                }
            )

            if not validation[
                "valid"
            ]:

                continue

            selected = (
                candidate.to_dict()
            )

            metadata.update(
                {
                    "status":
                        "ok",

                    "series_id":
                        series_id,

                    "series_name":
                        candidate.get(
                            "nombre"
                        ),

                    "river":
                        candidate.get(
                            "rio"
                        ),

                    "median_flow":
                        validation[
                            "median"
                        ],

                    "last_flow":
                        validation[
                            "last"
                        ],
                }
            )

            return (
                selected,
                metadata,
            )

        except Exception as exc:

            metadata[
                "tested"
            ].append(
                {
                    "series_id":
                        series_id,

                    "valid":
                        False,

                    "error":
                        str(exc),
                }
            )

    metadata[
        "status"
    ] = "sin_serie_validada"

    return (
        None,
        metadata,
    )


# ============================================================
# CONSULTAR HISTORIA EN BLOQUES
# ============================================================

def _query_history_blocks(
    series_id,
    start,
    end,
):

    start_dt = pd.to_datetime(
        start
    ).normalize()

    end_dt = pd.to_datetime(
        end
    ).normalize()

    frames = []

    block_start = start_dt

    while block_start <= end_dt:

        block_end = min(

            block_start
            + pd.DateOffset(
                years=
                    FLOW_HISTORY_BLOCK_YEARS
            )
            - pd.Timedelta(days=1),

            end_dt,
        )

        try:

            part = query_caudal_series(
                series_id,
                block_start,
                block_end,
            )

            if not part.empty:

                frames.append(
                    part
                )

        except Exception:
            pass

        block_start = (
            block_end
            + pd.Timedelta(days=1)
        )

    if not frames:

        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    return (
        result
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# HISTORIA DE CAUDAL POR ESTACIÓN
# ============================================================

def get_caudal_station(
    station,
    start,
    end,
):

    selected, metadata = (
        seleccionar_serie_caudal(
            station,
            start,
            end,
        )
    )

    column = FLOW_COLUMNS[
        station
    ]

    if selected is None:

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    column,
                ]
            ),
            metadata,
        )

    series_id = int(
        selected[
            "series_id"
        ]
    )

    try:

        raw = _query_history_blocks(
            series_id,
            start,
            end,
        )

        if raw.empty:

            metadata[
                "status"
            ] = "sin_observaciones"

            return (
                pd.DataFrame(
                    columns=[
                        "datetime",
                        column,
                    ]
                ),
                metadata,
            )

        raw[
            "datetime"
        ] = _to_naive_datetime(
            raw["datetime"]
        )

        raw[
            "datetime"
        ] = (
            raw[
                "datetime"
            ]
            .dt
            .normalize()
        )

        raw[
            "value"
        ] = pd.to_numeric(
            raw["value"],
            errors="coerce",
        )

        raw = raw[
            (
                raw["value"] >= 0
            )
            &
            (
                raw["value"]
                <= PARANA_TRUNK_MAX_FLOW
            )
        ]

        result = (
            raw
            .groupby(
                "datetime",
                as_index=False,
            )[
                "value"
            ]
            .mean()
            .rename(
                columns={
                    "value":
                        column
                }
            )
        )

        values = (
            pd.to_numeric(
                result[column],
                errors="coerce",
            )
            .dropna()
        )

        if (
            len(values)
            < MIN_FLOW_OBSERVATIONS
        ):

            metadata[
                "status"
            ] = "pocas_observaciones"

            return (
                pd.DataFrame(
                    columns=[
                        "datetime",
                        column,
                    ]
                ),
                metadata,
            )

        metadata[
            "status"
        ] = "ok"

        metadata[
            "records"
        ] = int(
            len(result)
        )

        metadata[
            "last_flow"
        ] = float(
            values.iloc[-1]
        )

        metadata[
            "median_flow"
        ] = float(
            values.median()
        )

        return (
            result,
            metadata,
        )

    except Exception as exc:

        metadata[
            "status"
        ] = "error"

        metadata[
            "error"
        ] = str(exc)

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    column,
                ]
            ),
            metadata,
        )


# ============================================================
# OBTENER TODOS LOS CAUDALES OBSERVADOS
# ============================================================

def get_observed_caudales(
    start,
    end,
):

    base = pd.DataFrame(
        {
            "datetime":
                _daily_range(
                    start,
                    end,
                )
        }
    )

    metadata = {

        "source":
            "INA A5",

        "stations":
            {},

        "observed_stations":
            [],
    }

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        try:

            df, meta = (
                get_caudal_station(
                    station,
                    start,
                    end,
                )
            )

            metadata[
                "stations"
            ][station] = meta

            if (
                df is not None
                and not df.empty
            ):

                base = base.merge(
                    df,
                    on="datetime",
                    how="left",
                )

            else:

                base[col] = np.nan

            count = int(
                pd.to_numeric(
                    base[col],
                    errors="coerce",
                )
                .notna()
                .sum()
            )

            if count >= MIN_FLOW_OBSERVATIONS:

                metadata[
                    "observed_stations"
                ].append(
                    station
                )

        except Exception as exc:

            base[col] = np.nan

            metadata[
                "stations"
            ][station] = {

                "status":
                    "error",

                "error":
                    str(exc),
            }

    return (
        base,
        metadata,
    )


# ============================================================
# MARCAR FUENTE DE CAUDAL
# ============================================================

def _initialize_flow_sources(
    df,
):

    result = df.copy()

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        if col not in result.columns:

            result[col] = np.nan

        result[
            source_col
        ] = np.where(
            pd.to_numeric(
                result[col],
                errors="coerce",
            )
            .notna(),

            "observado",

            "faltante",
        )

        result[
            quality_col
        ] = np.where(
            result[
                source_col
            ]
            == "observado",

            1.0,

            0.0,
        )

    return result


# ============================================================
# INTERPOLAR HUECOS CORTOS
# ============================================================

def _interpolate_short_flow_gaps(
    df,
):

    result = df.copy()

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        original = pd.to_numeric(
            result[col],
            errors="coerce",
        )

        interpolated = (
            original
            .interpolate(
                limit=
                    SHORT_INTERPOLATION_LIMIT,
                limit_area="inside",
            )
        )

        mask = (
            original.isna()
            &
            interpolated.notna()
        )

        result.loc[
            mask,
            col
        ] = interpolated.loc[
            mask
        ]

        result.loc[
            mask,
            source_col
        ] = "interpolado"

        result.loc[
            mask,
            quality_col
        ] = 0.90

    return result


# ============================================================
# ENCONTRAR ESTACIONES VECINAS CON CAUDAL
# ============================================================

def _neighbor_stations(
    target_station,
):

    target_index = (
        STATION_ORDER[
            target_station
        ]
    )

    others = [
        station
        for station in STATIONS
        if station != target_station
    ]

    others.sort(
        key=lambda station:
            abs(
                STATION_ORDER[
                    station
                ]
                - target_index
            )
    )

    return others


# ============================================================
# CALCULAR RELACIÓN HISTÓRICA ENTRE CAUDALES
# ============================================================

def _historical_flow_ratio(
    df,
    target_col,
    reference_col,
):

    if (
        target_col not in df.columns
        or reference_col
        not in df.columns
    ):

        return None

    pair = df[
        [
            target_col,
            reference_col,
        ]
    ].copy()

    pair[
        target_col
    ] = pd.to_numeric(
        pair[
            target_col
        ],
        errors="coerce",
    )

    pair[
        reference_col
    ] = pd.to_numeric(
        pair[
            reference_col
        ],
        errors="coerce",
    )

    pair = pair.dropna()

    pair = pair[
        pair[
            reference_col
        ] > 0
    ]

    if (
        len(pair)
        < MIN_RATIO_OVERLAP
    ):

        return None

    ratio = (
        pair[
            target_col
        ]
        /
        pair[
            reference_col
        ]
    )

    ratio = ratio.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    ratio = ratio[
        (
            ratio
            >= MIN_NEIGHBOR_RATIO
        )
        &
        (
            ratio
            <= MAX_NEIGHBOR_RATIO
        )
    ]

    if (
        len(ratio)
        < MIN_RATIO_OVERLAP
    ):

        return None

    median_ratio = float(
        ratio.median()
    )

    dispersion = float(
        ratio.std()
    )

    return {

        "ratio":
            median_ratio,

        "dispersion":
            dispersion,

        "records":
            int(
                len(ratio)
            ),
    }


# ============================================================
# ESTIMAR CAUDAL CON RELACIÓN HISTÓRICA
# ============================================================

def _fill_from_historical_ratios(
    df,
):

    result = df.copy()

    # --------------------------------------------------------
    # Varias pasadas permiten que una estación recién estimada
    # pueda ayudar a completar otra más adelante.
    # --------------------------------------------------------

    for _ in range(3):

        changes = 0

        for station in STATIONS:

            target_col = FLOW_COLUMNS[
                station
            ]

            source_col = (
                target_col
                + "_source"
            )

            quality_col = (
                target_col
                + "_quality"
            )

            missing = (
                pd.to_numeric(
                    result[
                        target_col
                    ],
                    errors="coerce",
                )
                .isna()
            )

            if not missing.any():
                continue

            for reference_station in (
                _neighbor_stations(
                    station
                )
            ):

                reference_col = (
                    FLOW_COLUMNS[
                        reference_station
                    ]
                )

                if (
                    reference_col
                    not in result.columns
                ):
                    continue

                ratio_info = (
                    _historical_flow_ratio(
                        result,
                        target_col,
                        reference_col,
                    )
                )

                if ratio_info is None:
                    continue

                reference = (
                    pd.to_numeric(
                        result[
                            reference_col
                        ],
                        errors="coerce",
                    )
                )

                estimated = (
                    reference
                    * ratio_info[
                        "ratio"
                    ]
                )

                mask = (
                    missing
                    &
                    reference.notna()
                    &
                    estimated.notna()
                )

                estimated = estimated.clip(
                    lower=
                        MIN_ESTIMATED_FLOW,

                    upper=
                        MAX_ESTIMATED_FLOW,
                )

                if mask.any():

                    result.loc[
                        mask,
                        target_col
                    ] = estimated.loc[
                        mask
                    ]

                    result.loc[
                        mask,
                        source_col
                    ] = (
                        "estimado_ratio_"
                        + _normalize_text(
                            reference_station
                        )
                        .replace(
                            " ",
                            "_"
                        )
                    )

                    # Cuanto mayor superposición,
                    # mejor calidad.
                    overlap_factor = min(
                        ratio_info[
                            "records"
                        ]
                        / 100.0,
                        1.0,
                    )

                    dispersion = (
                        ratio_info[
                            "dispersion"
                        ]
                    )

                    if not np.isfinite(
                        dispersion
                    ):

                        dispersion = 1.0

                    quality = (
                        0.65
                        +
                        0.15
                        * overlap_factor
                        -
                        min(
                            dispersion,
                            1.0,
                        )
                        * 0.10
                    )

                    quality = float(
                        np.clip(
                            quality,
                            0.45,
                            0.82,
                        )
                    )

                    result.loc[
                        mask,
                        quality_col
                    ] = quality

                    changes += int(
                        mask.sum()
                    )

                    missing = (
                        pd.to_numeric(
                            result[
                                target_col
                            ],
                            errors="coerce",
                        )
                        .isna()
                    )

                    if not missing.any():
                        break

        if changes == 0:
            break

    return result


# ============================================================
# ESTIMACIÓN DE ÚLTIMO RECURSO CON CAUDAL DEL CORREDOR
# ============================================================

def _fill_from_corridor_median(
    df,
):

    result = df.copy()

    flow_cols = [
        FLOW_COLUMNS[
            station
        ]
        for station in STATIONS
    ]

    numeric = (
        result[
            flow_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    corridor_median = (
        numeric
        .median(
            axis=1,
            skipna=True,
        )
    )

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        current = pd.to_numeric(
            result[col],
            errors="coerce",
        )

        missing = current.isna()

        # ----------------------------------------------------
        # Sólo estimar si existen al menos dos estaciones
        # válidas ese mismo día.
        # --------------------------------------------------------

        valid_count = (
            numeric
            .notna()
            .sum(
                axis=1
            )
        )

        mask = (
            missing
            &
            corridor_median.notna()
            &
            (
                valid_count >= 2
            )
        )

        estimated = (
            corridor_median
            .clip(
                lower=
                    MIN_ESTIMATED_FLOW,

                upper=
                    MAX_ESTIMATED_FLOW,
            )
        )

        result.loc[
            mask,
            col
        ] = estimated.loc[
            mask
        ]

        result.loc[
            mask,
            source_col
        ] = "estimado_corredor"

        result.loc[
            mask,
            quality_col
        ] = 0.40

    return result


# ============================================================
# RECONSTRUIR CAUDALES FALTANTES
# ============================================================

def complete_missing_flows(
    flow_history,
):

    result = (
        _initialize_flow_sources(
            flow_history
        )
    )

    result = (
        _interpolate_short_flow_gaps(
            result
        )
    )

    result = (
        _fill_from_historical_ratios(
            result
        )
    )

    result = (
        _fill_from_corridor_median(
            result
        )
    )

    # ========================================================
    # CONTROL FINAL
    # ========================================================

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        values = pd.to_numeric(
            result[col],
            errors="coerce",
        )

        invalid = (
            (
                values
                < MIN_ESTIMATED_FLOW
            )
            |
            (
                values
                > MAX_ESTIMATED_FLOW
            )
        )

        result.loc[
            invalid,
            col
        ] = np.nan

        result.loc[
            invalid,
            source_col
        ] = "rechazado"

        result.loc[
            invalid,
            quality_col
        ] = 0.0

    return result


# ============================================================
# ELEGIR CAUDAL PRINCIPAL
# ============================================================

def elegir_caudal_principal(
    df,
):

    if (
        df is None
        or df.empty
    ):

        return (
            None,
            None,
        )

    # --------------------------------------------------------
    # Priorizamos estaciones cercanas,
    # pero el valor debe tener calidad aceptable.
    # --------------------------------------------------------

    for station in FLOW_PRIORITY:

        col = FLOW_COLUMNS[
            station
        ]

        quality_col = (
            col
            + "_quality"
        )

        if col not in df.columns:
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if quality_col in df.columns:

            quality = pd.to_numeric(
                df[
                    quality_col
                ],
                errors="coerce",
            )

            valid_mask = (
                values.notna()
                &
                (
                    quality >= 0.60
                )
            )

        else:

            valid_mask = (
                values.notna()
            )

        if (
            int(
                valid_mask.sum()
            )
            >= MIN_FLOW_OBSERVATIONS
        ):

            return (
                col,
                station,
            )

    # --------------------------------------------------------
    # Segunda opción:
    # cualquier caudal reconstruido.
    # --------------------------------------------------------

    for station in FLOW_PRIORITY:

        col = FLOW_COLUMNS[
            station
        ]

        if (
            col in df.columns
            and pd.to_numeric(
                df[col],
                errors="coerce",
            )
            .notna()
            .sum()
            >= MIN_FLOW_OBSERVATIONS
        ):

            return (
                col,
                station,
            )

    return (
        None,
        None,
    )


# ============================================================
# PROYECCIÓN FUTURA DE CAUDAL
# ============================================================

def proyectar_serie_caudal(
    history,
    column,
    future_dates,
):

    values = (
        pd.to_numeric(
            history[
                column
            ],
            errors="coerce",
        )
        .dropna()
    )

    if values.empty:

        return pd.Series(
            np.nan,
            index=range(
                len(future_dates)
            ),
            dtype=float,
        )

    current = float(
        values.iloc[-1]
    )

    recent = values.tail(
        min(
            21,
            len(values),
        )
    )

    if len(recent) >= 4:

        try:

            slope = float(
                np.polyfit(
                    np.arange(
                        len(recent)
                    ),
                    recent.to_numpy(
                        dtype=float
                    ),
                    1,
                )[0]
            )

        except Exception:

            slope = 0.0

    else:

        slope = 0.0

    max_slope = max(
        current * 0.025,
        50.0,
    )

    slope = float(
        np.clip(
            slope,
            -max_slope,
            max_slope,
        )
    )

    output = []

    level = current

    for day in range(
        1,
        len(future_dates) + 1,
    ):

        damping = np.exp(
            -day / 20.0
        )

        change = (
            slope
            * damping
        )

        level = (
            level
            + change
        )

        level = float(
            np.clip(
                level,
                MIN_ESTIMATED_FLOW,
                MAX_ESTIMATED_FLOW,
            )
        )

        output.append(
            level
        )

    return pd.Series(
        output,
        dtype=float,
    )


# ============================================================
# FEATURES DE LLUVIA
# ============================================================

def agregar_features_lluvia(
    df,
):

    result = df.copy()

    for station in STATIONS:

        col = RAIN_COLUMNS[
            station
        ]

        if col not in result.columns:
            continue

        values = (
            pd.to_numeric(
                result[col],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
        )

        for window in [
            3,
            7,
            15,
            30,
        ]:

            result[
                f"{col}_{window}d"
            ] = (
                values
                .rolling(
                    window,
                    min_periods=1,
                )
                .sum()
            )

    return result


# ============================================================
# FEATURES DE CAUDAL
# ============================================================

def agregar_features_caudal(
    df,
):

    result = df.copy()

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        if col not in result.columns:
            continue

        q = pd.to_numeric(
            result[col],
            errors="coerce",
        )

        if q.notna().sum() < 3:
            continue

        for lag in [
            1,
            3,
            7,
            14,
        ]:

            result[
                f"{col}_diff_{lag}"
            ] = q.diff(
                lag
            )

        for window in [
            3,
            7,
            14,
            30,
        ]:

            result[
                f"{col}_mean_{window}"
            ] = (
                q
                .rolling(
                    window,
                    min_periods=1,
                )
                .mean()
            )

        result[
            f"{col}_trend_7"
        ] = (
            q
            -
            q.shift(7)
        )

        result[
            f"{col}_relative_7"
        ] = (
            q.diff(7)
            /
            q.shift(7).replace(
                0,
                np.nan,
            )
        )


    return result


# ============================================================
# PESOS HIDROLÓGICOS DEL CORREDOR
#
# Más cercanía a San Nicolás = mayor peso inmediato.
# ============================================================

CORRIDOR_WEIGHTS = {

    "Corrientes": 0.08,

    "Goya": 0.09,

    "La Paz": 0.11,

    "Paraná": 0.15,

    "Diamante": 0.16,

    "Rosario": 0.17,

    "Villa Constitución": 0.14,

    "San Nicolás": 0.10,
}


# ============================================================
# ÍNDICE DE PRESIÓN POR LLUVIA
# ============================================================

def agregar_indice_lluvia_corredor(
    df,
):

    result = df.copy()

    weighted = pd.Series(
        0.0,
        index=result.index,
        dtype=float,
    )

    total_weight = 0.0

    for station in STATIONS:

        col = (
            RAIN_COLUMNS[
                station
            ]
            + "_7d"
        )

        if col not in result.columns:
            continue

        weight = (
            CORRIDOR_WEIGHTS[
                station
            ]
        )

        values = (
            pd.to_numeric(
                result[col],
                errors="coerce",
            )
            .fillna(0.0)
        )

        weighted += (
            values
            * weight
        )

        total_weight += weight

    if total_weight > 0:

        weighted = (
            weighted
            / total_weight
        )

    result[
        "rain_pressure_7d"
    ] = weighted

    return result


# ============================================================
# ÍNDICE DE PRESIÓN POR CAUDAL
# ============================================================

def agregar_indice_caudal_corredor(
    df,
):

    result = df.copy()

    signals = []

    weights = []

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        if col not in result.columns:
            continue

        q = pd.to_numeric(
            result[col],
            errors="coerce",
        )

        baseline = (
            q
            .rolling(
                30,
                min_periods=7,
            )
            .median()
        )

        relative = (
            (
                q
                - baseline
            )
            /
            baseline.replace(
                0,
                np.nan,
            )
        )

        weight = (
            CORRIDOR_WEIGHTS[
                station
            ]
        )

        signals.append(
            relative
            * weight
        )

        weights.append(
            weight
        )

    if not signals:

        result[
            "flow_pressure"
        ] = 0.0

        return result

    combined = (
        pd.concat(
            signals,
            axis=1,
        )
        .sum(
            axis=1,
            skipna=True,
        )
    )

    total_weight = sum(
        weights
    )

    if total_weight > 0:

        combined = (
            combined
            / total_weight
        )

    combined = combined.clip(
        lower=-1.0,
        upper=2.0,
    )

    result[
        "flow_pressure"
    ] = combined.fillna(0.0)

    return result


# ============================================================
# ÍNDICE HIDROLÓGICO EXÓGENO
#
# En V11.11 todavía no incluye niveles aguas arriba:
# éstos se incorporarán en hydrology/model.
#
# Aquí dejamos:
#
# 55% caudal
# 45% lluvia
# ============================================================

def agregar_indice_hidrologico_exogeno(
    df,
):

    result = df.copy()

    rain_pressure = pd.to_numeric(
        result.get(
            "rain_pressure_7d",
            0.0,
        ),
        errors="coerce",
    ).fillna(0.0)

    flow_pressure = pd.to_numeric(
        result.get(
            "flow_pressure",
            0.0,
        ),
        errors="coerce",
    ).fillna(0.0)

    rain_scaled = (
        rain_pressure
        / 100.0
    ).clip(
        lower=0.0,
        upper=2.0,
    )

    result[
        "hydro_exogenous_pressure"
    ] = (
        0.55
        * flow_pressure
        +
        0.45
        * rain_scaled
    ).clip(
        lower=-1.0,
        upper=2.0,
    )

    return result


# ============================================================
# RESUMEN DE CALIDAD DE CAUDAL
# ============================================================

def _flow_quality_summary(
    df,
):

    output = {}

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        station_info = {

            "records":
                0,

            "observed":
                0,

            "interpolated":
                0,

            "estimated":
                0,

            "missing":
                0,

            "mean_quality":
                np.nan,
        }

        if col in df.columns:

            station_info[
                "records"
            ] = int(
                pd.to_numeric(
                    df[col],
                    errors="coerce",
                )
                .notna()
                .sum()
            )

        if source_col in df.columns:

            source = (
                df[
                    source_col
                ]
                .astype(str)
            )

            station_info[
                "observed"
            ] = int(
                (
                    source
                    == "observado"
                ).sum()
            )

            station_info[
                "interpolated"
            ] = int(
                (
                    source
                    == "interpolado"
                ).sum()
            )

            station_info[
                "estimated"
            ] = int(
                source.str.startswith(
                    "estimado"
                ).sum()
            )

            station_info[
                "missing"
            ] = int(
                (
                    source
                    == "faltante"
                ).sum()
            )

        if quality_col in df.columns:

            quality = (
                pd.to_numeric(
                    df[
                        quality_col
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            if not quality.empty:

                station_info[
                    "mean_quality"
                ] = float(
                    quality.mean()
                )

        output[
            station
        ] = station_info

    return output


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def get_exogenous_data(
    start,
    end,
    forecast_days=15,
):

    start = _normalize_date(
        start
    )

    end = _normalize_date(
        end
    )

    forecast_days = min(
        max(
            int(forecast_days),
            1,
        ),
        MAX_FORECAST_DAYS,
    )

    # ========================================================
    # LLUVIA HISTÓRICA
    # ========================================================

    rain_history, rain_meta = (
        get_rain_history(
            start,
            end,
        )
    )

    # ========================================================
    # CAUDALES OBSERVADOS
    # ========================================================

    observed_flows, flow_meta = (
        get_observed_caudales(
            start,
            end,
        )
    )

    # ========================================================
    # RECONSTRUCCIÓN DE FALTANTES
    # ========================================================

    completed_flows = (
        complete_missing_flows(
            observed_flows
        )
    )

    # ========================================================
    # DATASET HISTÓRICO
    # ========================================================

    history = pd.DataFrame(
        {
            "datetime":
                _daily_range(
                    start,
                    end,
                )
        }
    )

    history = history.merge(
        rain_history,
        on="datetime",
        how="left",
    )

    flow_columns_to_merge = [
        col
        for col
        in completed_flows.columns
        if col != "datetime"
    ]

    history = history.merge(
        completed_flows[
            [
                "datetime",
                *flow_columns_to_merge,
            ]
        ],
        on="datetime",
        how="left",
    )

    # ========================================================
    # GARANTIZAR TODAS LAS COLUMNAS
    # ========================================================

    for station in STATIONS:

        rain_col = (
            RAIN_COLUMNS[
                station
            ]
        )

        flow_col = (
            FLOW_COLUMNS[
                station
            ]
        )

        source_col = (
            flow_col
            + "_source"
        )

        quality_col = (
            flow_col
            + "_quality"
        )

        if rain_col not in history.columns:

            history[
                rain_col
            ] = 0.0

        if flow_col not in history.columns:

            history[
                flow_col
            ] = np.nan

        if source_col not in history.columns:

            history[
                source_col
            ] = "faltante"

        if quality_col not in history.columns:

            history[
                quality_col
            ] = 0.0

    # ========================================================
    # CAUDAL PRINCIPAL
    # ========================================================

    (
        main_flow_col,
        main_flow_station,
    ) = elegir_caudal_principal(
        history
    )

    if main_flow_col:

        history[
            "caudal_m3s"
        ] = pd.to_numeric(
            history[
                main_flow_col
            ],
            errors="coerce",
        )

        source_col = (
            main_flow_col
            + "_source"
        )

        quality_col = (
            main_flow_col
            + "_quality"
        )

        history[
            "caudal_source"
        ] = (
            history[
                source_col
            ]
            if source_col
            in history.columns
            else "desconocido"
        )

        history[
            "caudal_quality"
        ] = (
            history[
                quality_col
            ]
            if quality_col
            in history.columns
            else 0.0
        )

    else:

        history[
            "caudal_m3s"
        ] = np.nan

        history[
            "caudal_source"
        ] = "sin_datos"

        history[
            "caudal_quality"
        ] = 0.0

    # ========================================================
    # FEATURES
    # ========================================================

    history = (
        agregar_features_lluvia(
            history
        )
    )

    history = (
        agregar_features_caudal(
            history
        )
    )

    history = (
        agregar_indice_lluvia_corredor(
            history
        )
    )

    history = (
        agregar_indice_caudal_corredor(
            history
        )
    )

    history = (
        agregar_indice_hidrologico_exogeno(
            history
        )
    )

    history[
        "datetime"
    ] = _to_naive_datetime(
        history["datetime"]
    )

    history = (
        history
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # FUTURO DE LLUVIA
    # ========================================================

    rain_future, rain_future_meta = (
        get_rain_forecast(
            end,
            forecast_days,
        )
    )

    future = rain_future.copy()

    # ========================================================
    # FUTURO DE CAUDAL
    # ========================================================

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        valid_count = int(
            pd.to_numeric(
                history[col],
                errors="coerce",
            )
            .notna()
            .sum()
        )

        if (
            valid_count
            >= MIN_FLOW_OBSERVATIONS
        ):

            future[col] = (
                proyectar_serie_caudal(
                    history,
                    col,
                    future[
                        "datetime"
                    ],
                )
            )

            future[
                source_col
            ] = "proyectado"

            # ------------------------------------------------
            # La calidad disminuye con el horizonte.
            # ------------------------------------------------

            horizon = np.arange(
                1,
                len(future) + 1,
            )

            quality = (
                0.70
                * np.exp(
                    -horizon
                    / 60.0
                )
            )

            quality = np.clip(
                quality,
                0.25,
                0.70,
            )

            future[
                quality_col
            ] = quality

        else:

            future[col] = np.nan

            future[
                source_col
            ] = "sin_datos"

            future[
                quality_col
            ] = 0.0

    # ========================================================
    # RECONSTRUIR FUTURO FALTANTE ENTRE ESTACIONES
    # ========================================================

    future = (
        complete_missing_flows(
            future
        )
    )

    # No queremos que complete_missing_flows cambie
    # "proyectado" a "observado" en valores ya generados.
    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        if source_col not in future.columns:

            future[
                source_col
            ] = "proyectado"

        if quality_col not in future.columns:

            future[
                quality_col
            ] = 0.45

    # ========================================================
    # CAUDAL PRINCIPAL FUTURO
    # ========================================================

    if (
        main_flow_col
        and
        main_flow_col
        in future.columns
    ):

        future[
            "caudal_m3s"
        ] = pd.to_numeric(
            future[
                main_flow_col
            ],
            errors="coerce",
        )

        source_col = (
            main_flow_col
            + "_source"
        )

        quality_col = (
            main_flow_col
            + "_quality"
        )

        future[
            "caudal_source"
        ] = future.get(
            source_col,
            "proyectado",
        )

        future[
            "caudal_quality"
        ] = future.get(
            quality_col,
            0.50,
        )

    else:

        future[
            "caudal_m3s"
        ] = np.nan

        future[
            "caudal_source"
        ] = "sin_datos"

        future[
            "caudal_quality"
        ] = 0.0

    # ========================================================
    # FEATURES FUTURAS
    # ========================================================

    history_tail = (
        history
        .tail(45)
        .copy()
    )

    common_columns = sorted(
        set(
            history_tail.columns
        )
        |
        set(
            future.columns
        )
    )

    for col in common_columns:

        if col not in history_tail.columns:
            history_tail[col] = np.nan

        if col not in future.columns:
            future[col] = np.nan

    combined = pd.concat(
        [
            history_tail[
                common_columns
            ],

            future[
                common_columns
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    combined = (
        agregar_features_lluvia(
            combined
        )
    )

    combined = (
        agregar_features_caudal(
            combined
        )
    )

    combined = (
        agregar_indice_lluvia_corredor(
            combined
        )
    )

    combined = (
        agregar_indice_caudal_corredor(
            combined
        )
    )

    combined = (
        agregar_indice_hidrologico_exogeno(
            combined
        )
    )

    combined[
        "datetime"
    ] = _to_naive_datetime(
        combined[
            "datetime"
        ]
    )

    future_start = (
        pd.to_datetime(
            end
        ).normalize()
        + pd.Timedelta(
            days=1
        )
    )

    future = (
        combined[
            combined[
                "datetime"
            ]
            >= future_start
        ]
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .head(
            forecast_days
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # CALIDAD / COBERTURA
    # ========================================================

    flow_quality = (
        _flow_quality_summary(
            history
        )
    )

    rain_stations = []

    flow_observed_stations = []

    flow_available_stations = []

    for station in STATIONS:

        rain_col = (
            RAIN_COLUMNS[
                station
            ]
        )

        flow_col = (
            FLOW_COLUMNS[
                station
            ]
        )

        source_col = (
            flow_col
            + "_source"
        )

        if (
            rain_col in history.columns
            and pd.to_numeric(
                history[
                    rain_col
                ],
                errors="coerce",
            )
            .notna()
            .any()
        ):

            rain_stations.append(
                station
            )

        if (
            flow_col in history.columns
            and pd.to_numeric(
                history[
                    flow_col
                ],
                errors="coerce",
            )
            .notna()
            .any()
        ):

            flow_available_stations.append(
                station
            )

        if (
            source_col
            in history.columns
            and (
                history[
                    source_col
                ]
                == "observado"
            ).any()
        ):

            flow_observed_stations.append(
                station
            )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "version":
            VERSION,

        "status":
            "ok",

        "rain_source":
            "Open-Meteo",

        "flow_source":
            "INA A5 + reconstrucción hidrológica",

        "main_flow_station":
            main_flow_station,

        "main_flow_column":
            main_flow_col,

        "rain_stations":
            rain_stations,

        "flow_observed_stations":
            flow_observed_stations,

        "flow_available_stations":
            flow_available_stations,

        "rain_station_count":
            len(
                rain_stations
            ),

        "flow_observed_station_count":
            len(
                flow_observed_stations
            ),

        "flow_available_station_count":
            len(
                flow_available_stations
            ),

        "flow_quality":
            flow_quality,

        "rain_history":
            rain_meta,

        "rain_forecast":
            rain_future_meta,

        "flow_history":
            flow_meta,

        "history_rows":
            int(
                len(history)
            ),

        "future_rows":
            int(
                len(future)
            ),

        "real_weather_forecast_days":
            OPEN_METEO_REAL_FORECAST_DAYS,

        "reconstruction":
            {

                "short_interpolation_days":
                    SHORT_INTERPOLATION_LIMIT,

                "historical_ratio_min_overlap":
                    MIN_RATIO_OVERLAP,

                "methods":
                    [
                        "observado",
                        "interpolado",
                        "estimado_ratio",
                        "estimado_corredor",
                    ],
            },
    }

    return (
        history,
        future,
        metadata,
    )


# ============================================================
# DIAGNÓSTICO
# ============================================================

def diagnostic(
    start,
    end,
):

    result = {

        "version":
            VERSION,

        "start":
            _normalize_date(start),

        "end":
            _normalize_date(end),

        "status":
            "pendiente",
    }

    try:

        catalog = (
            get_ina_catalog()
        )

        result[
            "ina_catalog_records"
        ] = int(
            len(catalog)
        )

        result[
            "ina_flow_series"
        ] = int(
            (
                catalog[
                    "var_id"
                ]
                == VAR_ID_CAUDAL
            ).sum()
        )

    except Exception as exc:

        result[
            "catalog_error"
        ] = str(exc)

    try:

        history, future, metadata = (
            get_exogenous_data(
                start,
                end,
                forecast_days=60,
            )
        )

        result[
            "status"
        ] = "ok"

        result[
            "history_rows"
        ] = int(
            len(history)
        )

        result[
            "future_rows"
        ] = int(
            len(future)
        )

        result[
            "main_flow_station"
        ] = metadata.get(
            "main_flow_station"
        )

        result[
            "main_flow_column"
        ] = metadata.get(
            "main_flow_column"
        )

        result[
            "flow_observed_stations"
        ] = metadata.get(
            "flow_observed_stations",
            [],
        )

        result[
            "flow_available_stations"
        ] = metadata.get(
            "flow_available_stations",
            [],
        )

        result[
            "flow_quality"
        ] = metadata.get(
            "flow_quality",
            {},
        )

        current_flows = {}

        for station in STATIONS:

            col = FLOW_COLUMNS[
                station
            ]

            source_col = (
                col
                + "_source"
            )

            quality_col = (
                col
                + "_quality"
            )

            if col not in history.columns:
                continue

            valid = history[
                [
                    "datetime",
                    col,
                    source_col,
                    quality_col,
                ]
            ].copy()

            valid[
                col
            ] = pd.to_numeric(
                valid[
                    col
                ],
                errors="coerce",
            )

            valid = valid.dropna(
                subset=[col]
            )

            if valid.empty:
                continue

            row = valid.iloc[-1]

            current_flows[
                station
            ] = {

                "datetime":
                    row[
                        "datetime"
                    ],

                "flow_m3s":
                    float(
                        row[col]
                    ),

                "source":
                    row[
                        source_col
                    ],

                "quality":
                    _safe_float(
                        row[
                            quality_col
                        ],
                        0.0,
                    ),
            }

        result[
            "current_flows"
        ] = current_flows

    except Exception as exc:

        result[
            "status"
        ] = "error"

        result[
            "error"
        ] = str(exc)

    return result
