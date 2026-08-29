# ============================================================
# PARANÁ · SAN NICOLÁS
# src/exogenous.py
# V11.10.2 COMPLETO
#
# VARIABLES EXÓGENAS MULTIESTACIÓN
#
# CAMBIO PRINCIPAL V11.10.2
# ------------------------------------------------------------
# VALIDACIÓN HIDROLÓGICA DE CAUDALES INA
#
# - var_id = 4 -> Caudal
# - Coincidencia real de estación
# - Preferencia por río Paraná
# - Penalización de tributarios / canales / arroyos
# - Validación con observaciones reales
# - Validación de magnitud para el Paraná troncal
# - Mediana reciente mínima configurable
# - Comparación aproximada con caudales vecinos
# - No inventa caudales donde no existe serie confiable
# - Selecciona como caudal principal la estación más próxima
#   a San Nicolás QUE SUPERE TODAS LAS VALIDACIONES
#
# PRECIPITACIÓN:
# - Corrientes
# - Goya
# - La Paz
# - Paraná
# - Diamante
# - Rosario
# - Villa Constitución
# - San Nicolás
#
# CAUDAL:
# - q_corrientes
# - q_goya
# - q_la_paz
# - q_parana
# - q_diamante
# - q_rosario
# - q_villa_constitucion
# - q_san_nicolas
#
# COMPATIBILIDAD:
# - precip_mm
# - caudal_m3s
#
# API:
# get_exogenous_data(start, end, forecast_days=60)
#
# retorna:
# history, future, metadata
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

VERSION = "V11.10.2"


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

REQUEST_TIMEOUT = 45

MAX_FORECAST_DAYS = 60

OPEN_METEO_REAL_FORECAST_DAYS = 16

MIN_FLOW_OBSERVATIONS = 3

FLOW_VALIDATION_DAYS = 180

FLOW_HISTORY_BLOCK_YEARS = 5


# ============================================================
# VALIDACIÓN DEL CAUDAL DEL PARANÁ TRONCAL
#
# Este proyecto sigue el cauce principal del Paraná entre
# Corrientes y San Nicolás.
#
# No es un criterio universal para cualquier río.
# ============================================================

PARANA_TRUNK_MIN_MEDIAN_FLOW = 500.0

PARANA_TRUNK_MIN_RECENT_FLOW = 250.0

PARANA_TRUNK_MAX_FLOW = 100000.0

FLOW_NEIGHBOR_RATIO_MIN = 0.15

FLOW_NEIGHBOR_RATIO_MAX = 6.50


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

INA_A5_BASE_URL = (
    "https://alerta.ina.gob.ar/a5"
)

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

    "Corrientes":
        "rain_corrientes",

    "Goya":
        "rain_goya",

    "La Paz":
        "rain_la_paz",

    "Paraná":
        "rain_parana",

    "Diamante":
        "rain_diamante",

    "Rosario":
        "rain_rosario",

    "Villa Constitución":
        "rain_villa_constitucion",

    "San Nicolás":
        "rain_san_nicolas",
}


FLOW_COLUMNS = {

    "Corrientes":
        "q_corrientes",

    "Goya":
        "q_goya",

    "La Paz":
        "q_la_paz",

    "Paraná":
        "q_parana",

    "Diamante":
        "q_diamante",

    "Rosario":
        "q_rosario",

    "Villa Constitución":
        "q_villa_constitucion",

    "San Nicolás":
        "q_san_nicolas",
}


# ============================================================
# PRIORIDAD DEL CAUDAL PRINCIPAL
#
# La prioridad es geográfica.
# Pero sólo se utiliza una estación si su serie fue validada.
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
        "v constitucion",
        "v. constitucion",
    ],

    "San Nicolás": [
        "san nicolas",
        "san nicolás",
        "puerto san nicolas",
        "puerto san nicolás",
    ],
}


# ============================================================
# TÉRMINOS QUE INDICAN QUE NO ES EL PARANÁ TRONCAL
# ============================================================

NON_TRUNK_TERMS = [
    "arroyo",
    "canal",
    "riacho",
    "laguna",
    "afluente",
    "tributario",
    "desembocadura",
    "brazo",
    "aliviador",
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


# ============================================================
# UTILIDADES
# ============================================================

def _normalize_text(value):

    if value is None:
        return ""

    text = str(
        value
    ).lower().strip()

    text = unicodedata.normalize(
        "NFD",
        text,
    )

    text = "".join(
        ch
        for ch in text
        if unicodedata.category(
            ch
        ) != "Mn"
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


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


def _safe_float(
    value,
    default=np.nan,
):

    try:

        value = float(
            value
        )

        if np.isfinite(
            value
        ):

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
            float(
                value
            )
        )

    except Exception:

        return default


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
# PRECIPITACIÓN HISTÓRICA
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
            _normalize_date(
                start
            ),

        "end_date":
            _normalize_date(
                end
            ),

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

    precip = daily.get(
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
                        precip
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
            result[
                col
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

    return (
        result
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


# ============================================================
# LLUVIA HISTÓRICA DEL CORREDOR
# ============================================================

def get_rain_history(
    start,
    end,
):

    start = _normalize_date(
        start
    )

    end = _normalize_date(
        end
    )

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

                base[
                    col
                ] = np.nan

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
                    base[
                        col
                    ],
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

                base[
                    col
                ] = np.nan

            metadata[
                "stations"
            ][station] = {

                "status":
                    "error",

                "error":
                    str(
                        exc
                    ),

                "records":
                    0,
            }

    available_cols = [

        RAIN_COLUMNS[
            station
        ]

        for station in STATIONS

        if (
            RAIN_COLUMNS[
                station
            ]
            in base.columns
            and pd.to_numeric(
                base[
                    RAIN_COLUMNS[
                        station
                    ]
                ],
                errors="coerce",
            )
            .notna()
            .any()
        )
    ]

    if available_cols:

        base[
            "precip_mm"
        ] = (
            base[
                available_cols
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
# PRONÓSTICO DE LLUVIA
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

    precip = daily.get(
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
                        precip
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
            result[
                col
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

    return (
        result
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


def get_rain_forecast(
    start_date,
    forecast_days=60,
):

    forecast_days = min(
        max(
            int(
                forecast_days
            ),
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
            + pd.Timedelta(
                days=1
            ),

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

        "after_real_forecast":
            (
                "Los valores posteriores al horizonte real "
                "meteorológico no deben interpretarse como "
                "pronóstico determinista."
            ),
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

                base[
                    col
                ] = 0.0

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
                    base[
                        col
                    ],
                    errors="coerce",
                )
                .notna()
                .sum()
            )

            if real_count > 0:

                metadata[
                    "available_stations"
                ].append(
                    station
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

            base[
                col
            ] = (
                pd.to_numeric(
                    base[
                        col
                    ],
                    errors="coerce",
                )
                .fillna(
                    0.0
                )
            )

        except Exception as exc:

            base[
                col
            ] = 0.0

            metadata[
                "stations"
            ][station] = {

                "status":
                    "error",

                "error":
                    str(
                        exc
                    ),

                "records":
                    0,
            }

    available_cols = [

        RAIN_COLUMNS[
            station
        ]

        for station in STATIONS

        if RAIN_COLUMNS[
            station
        ] in base.columns
    ]

    if available_cols:

        base[
            "precip_mm"
        ] = (
            base[
                available_cols
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .mean(
                axis=1,
                skipna=True,
            )
            .fillna(
                0.0
            )
        )

    else:

        base[
            "precip_mm"
        ] = 0.0

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
            and len(
                coords
            ) >= 2
        ):

            lon = _safe_float(
                coords[0]
            )

            lat = _safe_float(
                coords[1]
            )

        rows.append(
            {
                "id":
                    props.get(
                        "id"
                    ),

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

                "estacion_id":
                    props.get(
                        "estacion_id"
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

                "data_availability":
                    props.get(
                        "data_availability"
                    ),

                "fuente":
                    props.get(
                        "fuente"
                    ),

                "public":
                    props.get(
                        "public"
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
# PUNTAJE DE ESTACIÓN
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
        [
            station
        ],
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
                130,
            )

        elif name_norm.startswith(
            alias_norm
        ):

            score = max(
                score,
                110,
            )

        elif re.search(
            r"\b"
            + re.escape(
                alias_norm
            )
            + r"\b",
            name_norm,
        ):

            score = max(
                score,
                90,
            )

        elif alias_norm in name_norm:

            score = max(
                score,
                65,
            )

    bad_terms = [
        "meteorologica",
        "meteorologico",
        "aeropuerto",
        "inta",
        "escuela",
        "agrometeorologica",
        "pluviometrica",
        "precipitacion",
        "lluvia",
    ]

    for bad in bad_terms:

        if bad in name_norm:

            score -= 100

    return score


# ============================================================
# PUNTAJE DEL RÍO
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

        score += 80

    for term in NON_TRUNK_TERMS:

        term_norm = _normalize_text(
            term
        )

        if term_norm in river_norm:

            score -= 90

    return score


# ============================================================
# VERIFICAR SI PARECE PARANÁ TRONCAL
# ============================================================

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

    # Si se especifica Paraná explícitamente es buena señal.
    has_parana = (
        "parana"
        in combined
    )

    # Rechazar si aparecen términos de curso secundario.
    has_bad_term = any(
        _normalize_text(
            term
        )
        in combined
        for term in NON_TRUNK_TERMS
    )

    if has_bad_term:

        return False

    if river_norm:

        return (
            "parana"
            in river_norm
        )

    # Si el campo río está vacío, no podemos descartarlo
    # sólo por eso; se validará por magnitud.
    return has_parana or not river_norm


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

    candidates[
        "river_score"
    ] = candidates[
        "rio"
    ].apply(
        _river_score
    )

    # --------------------------------------------------------
    # Debe coincidir realmente con la estación.
    # --------------------------------------------------------

    candidates = candidates[
        candidates[
            "station_score"
        ]
        > 0
    ].copy()

    if candidates.empty:

        return pd.DataFrame()

    candidates[
        "is_parana_trunk"
    ] = candidates.apply(
        lambda row:
            _is_parana_trunk_candidate(
                row.get(
                    "rio"
                ),
                row.get(
                    "nombre"
                ),
            ),
        axis=1,
    )

    # --------------------------------------------------------
    # Preferir Paraná troncal.
    # No se elimina aún el candidato si el campo río viene
    # vacío, porque la prueba de caudal decidirá luego.
    # --------------------------------------------------------

    candidates[
        "trunk_score"
    ] = np.where(
        candidates[
            "is_parana_trunk"
        ],
        60,
        -100,
    )

    candidates[
        "start_dt"
    ] = pd.to_datetime(
        candidates[
            "timestart"
        ],
        errors="coerce",
        utc=True,
    )

    candidates[
        "end_dt"
    ] = pd.to_datetime(
        candidates[
            "timeend"
        ],
        errors="coerce",
        utc=True,
    )

    candidates[
        "proc_score"
    ] = (
        pd.to_numeric(
            candidates[
                "proc_id"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .apply(
            lambda value:
                10
                if value > 0
                else 0
        )
    )

    candidates[
        "count_score"
    ] = (
        np.log10(
            pd.to_numeric(
                candidates[
                    "count"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .clip(
                lower=0
            )
            + 1
        )
        * 6.0
    )

    candidates[
        "recency_score"
    ] = 0.0

    valid_end = candidates[
        "end_dt"
    ].notna()

    if valid_end.any():

        now = pd.Timestamp.now(
            tz="UTC"
        )

        age_days = (
            now
            - candidates.loc[
                valid_end,
                "end_dt"
            ]
        ).dt.days

        candidates.loc[
            valid_end,
            "recency_score"
        ] = np.where(
            age_days <= 60,
            25,
            np.where(
                age_days <= 365,
                15,
                np.where(
                    age_days <= 365 * 5,
                    5,
                    0,
                ),
            ),
        )

    candidates[
        "overlap_score"
    ] = 0.0

    if (
        start is not None
        and end is not None
    ):

        request_start = pd.Timestamp(
            _normalize_date(
                start
            ),
            tz="UTC",
        )

        request_end = pd.Timestamp(
            _normalize_date(
                end
            ),
            tz="UTC",
        )

        overlap = (
            candidates[
                "start_dt"
            ].fillna(
                request_start
            )
            <= request_end
        ) & (
            candidates[
                "end_dt"
            ].fillna(
                request_end
            )
            >= request_start
        )

        candidates[
            "overlap_score"
        ] = np.where(
            overlap,
            20,
            -60,
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
            "proc_score"
        ]
        +
        candidates[
            "count_score"
        ]
        +
        candidates[
            "recency_score"
        ]
        +
        candidates[
            "overlap_score"
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
# EXTRAER REGISTROS A5
# ============================================================

def _extract_records(
    obj,
):

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
                    str(
                        key
                    ).lower()
                    for key in item.keys()
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
                (
                    list,
                    dict,
                ),
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
            record.get(
                key
            ),
            errors="coerce",
            utc=True,
        )

        if pd.isna(
            dt
        ):
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
            record.get(
                key
            )
        )

        if np.isfinite(
            value
        ):

            return value

    return np.nan


# ============================================================
# CONSULTA INA
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
            int(
                series_id
            ),

        "timestart":
            _normalize_date(
                start
            ),

        "timeend":
            _normalize_date(
                end
            ),
    }

    response = requests.get(

        INA_OBSERVATIONS_URL,

        params=params,

        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    records = _extract_records(
        data
    )

    rows = []

    for record in records:

        dt = _record_datetime(
            record
        )

        value = _record_value(
            record
        )

        if (
            pd.isna(
                dt
            )
            or not np.isfinite(
                value
            )
        ):

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

    result = result.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )

    return (
        result
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
# ESTADÍSTICAS DE VALIDACIÓN
# ============================================================

def _flow_statistics(
    df,
):

    if (
        df is None
        or df.empty
        or "value"
        not in df.columns
    ):

        return {
            "records":
                0,

            "median":
                np.nan,

            "mean":
                np.nan,

            "min":
                np.nan,

            "max":
                np.nan,

            "last":
                np.nan,
        }

    values = (
        pd.to_numeric(
            df[
                "value"
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if values.empty:

        return {
            "records":
                0,

            "median":
                np.nan,

            "mean":
                np.nan,

            "min":
                np.nan,

            "max":
                np.nan,

            "last":
                np.nan,
        }

    return {

        "records":
            int(
                len(
                    values
                )
            ),

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
                values.iloc[
                    -1
                ]
            ),
    }


# ============================================================
# VALIDACIÓN HIDROLÓGICA DE UNA SERIE
# ============================================================

def _validate_trunk_flow_series(
    candidate,
    data,
):

    stats = _flow_statistics(
        data
    )

    reasons = []

    valid = True

    name = candidate.get(
        "nombre"
    )

    river = candidate.get(
        "rio"
    )

    # --------------------------------------------------------
    # Cantidad mínima.
    # --------------------------------------------------------

    if (
        stats[
            "records"
        ]
        < MIN_FLOW_OBSERVATIONS
    ):

        valid = False

        reasons.append(
            "pocas_observaciones"
        )

    # --------------------------------------------------------
    # Curso principal.
    # --------------------------------------------------------

    trunk_candidate = (
        _is_parana_trunk_candidate(
            river,
            name,
        )
    )

    if not trunk_candidate:

        valid = False

        reasons.append(
            "no_parana_troncal"
        )

    # --------------------------------------------------------
    # Magnitud.
    #
    # Esto evita aceptar 8 m3/s, 20 m3/s, etc.
    # como si fueran el caudal del Paraná principal.
    # --------------------------------------------------------

    median = stats[
        "median"
    ]

    last = stats[
        "last"
    ]

    if (
        not np.isfinite(
            median
        )
        or median
        < PARANA_TRUNK_MIN_MEDIAN_FLOW
    ):

        valid = False

        reasons.append(
            "mediana_demasiado_baja"
        )

    if (
        np.isfinite(
            last
        )
        and last
        < PARANA_TRUNK_MIN_RECENT_FLOW
    ):

        valid = False

        reasons.append(
            "caudal_actual_demasiado_bajo"
        )

    if (
        np.isfinite(
            stats[
                "max"
            ]
        )
        and stats[
            "max"
        ]
        > PARANA_TRUNK_MAX_FLOW
    ):

        valid = False

        reasons.append(
            "caudal_maximo_fuera_rango"
        )

    return {

        "valid":
            bool(
                valid
            ),

        "is_parana_trunk":
            bool(
                trunk_candidate
            ),

        "reasons":
            reasons,

        **stats,
    }


# ============================================================
# VENTANA DE VALIDACIÓN
# ============================================================

def _candidate_validation_window(
    candidate,
    requested_start,
    requested_end,
):

    request_start = pd.to_datetime(
        requested_start
    ).normalize()

    request_end = pd.to_datetime(
        requested_end
    ).normalize()

    catalog_start = pd.to_datetime(
        candidate.get(
            "timestart"
        ),
        errors="coerce",
        utc=True,
    )

    catalog_end = pd.to_datetime(
        candidate.get(
            "timeend"
        ),
        errors="coerce",
        utc=True,
    )

    if not pd.isna(
        catalog_start
    ):

        catalog_start = (
            catalog_start
            .tz_localize(
                None
            )
            .normalize()
        )

    else:

        catalog_start = request_start

    if not pd.isna(
        catalog_end
    ):

        catalog_end = (
            catalog_end
            .tz_localize(
                None
            )
            .normalize()
        )

    else:

        catalog_end = request_end

    overlap_start = max(
        request_start,
        catalog_start,
    )

    overlap_end = min(
        request_end,
        catalog_end,
    )

    if overlap_end < overlap_start:

        return None

    validation_start = max(

        overlap_start,

        overlap_end
        - pd.Timedelta(
            days=
                FLOW_VALIDATION_DAYS
        ),
    )

    return (
        validation_start,
        overlap_end,
    )


# ============================================================
# SELECCIONAR SERIE DE CAUDAL
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

        "candidate_count":
            int(
                len(
                    candidates
                )
            ),

        "tested":
            [],

        "status":
            "sin_serie",
    }

    if candidates.empty:

        return (
            None,
            metadata,
        )

    for _, candidate in (
        candidates
        .head(
            25
        )
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

        validation_window = (
            _candidate_validation_window(
                candidate,
                start,
                end,
            )
        )

        if validation_window is None:
            continue

        test_start, test_end = (
            validation_window
        )

        try:

            test_data = query_caudal_series(

                series_id,

                test_start,

                test_end,
            )

            validation = (
                _validate_trunk_flow_series(
                    candidate,
                    test_data,
                )
            )

            test_info = {

                "series_id":
                    series_id,

                "name":
                    candidate.get(
                        "nombre"
                    ),

                "river":
                    candidate.get(
                        "rio"
                    ),

                "score":
                    _safe_float(
                        candidate.get(
                            "score"
                        ),
                        0.0,
                    ),

                **validation,
            }

            metadata[
                "tested"
            ].append(
                test_info
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

                    "validation_records":
                        validation[
                            "records"
                        ],

                    "is_parana_trunk":
                        validation[
                            "is_parana_trunk"
                        ],

                    "catalog_timestart":
                        candidate.get(
                            "timestart"
                        ),

                    "catalog_timeend":
                        candidate.get(
                            "timeend"
                        ),

                    "catalog_count":
                        _safe_int(
                            candidate.get(
                                "count"
                            ),
                            0,
                        ),
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

                    "status":
                        "error",

                    "error":
                        str(
                            exc
                        ),
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
# HISTORIA DE SERIE
# ============================================================

def _query_selected_series_history(
    selected,
    start,
    end,
):

    series_id = int(
        selected[
            "series_id"
        ]
    )

    request_start = pd.to_datetime(
        start
    ).normalize()

    request_end = pd.to_datetime(
        end
    ).normalize()

    catalog_start = pd.to_datetime(
        selected.get(
            "timestart"
        ),
        errors="coerce",
        utc=True,
    )

    catalog_end = pd.to_datetime(
        selected.get(
            "timeend"
        ),
        errors="coerce",
        utc=True,
    )

    if not pd.isna(
        catalog_start
    ):

        catalog_start = (
            catalog_start
            .tz_localize(
                None
            )
            .normalize()
        )

        request_start = max(
            request_start,
            catalog_start,
        )

    if not pd.isna(
        catalog_end
    ):

        catalog_end = (
            catalog_end
            .tz_localize(
                None
            )
            .normalize()
        )

        request_end = min(
            request_end,
            catalog_end,
        )

    if request_end < request_start:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Para períodos cortos intentamos una consulta.
    # Para períodos muy extensos dividimos en bloques.
    # --------------------------------------------------------

    total_days = (
        request_end
        - request_start
    ).days

    if total_days <= 365 * FLOW_HISTORY_BLOCK_YEARS:

        try:

            return query_caudal_series(
                series_id,
                request_start,
                request_end,
            )

        except Exception:

            pass

    frames = []

    block_start = request_start

    while block_start <= request_end:

        block_end = min(

            block_start
            + pd.DateOffset(
                years=
                    FLOW_HISTORY_BLOCK_YEARS
            )
            - pd.Timedelta(
                days=1
            ),

            request_end,
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
            + pd.Timedelta(
                days=1
            )
        )

    if not frames:

        return pd.DataFrame()

    return (
        pd.concat(
            frames,
            ignore_index=True,
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
# CAUDAL POR ESTACIÓN
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

    try:

        data = (
            _query_selected_series_history(
                selected,
                start,
                end,
            )
        )

        if data.empty:

            metadata[
                "status"
            ] = "sin_observaciones"

            metadata[
                "records"
            ] = 0

            return (
                pd.DataFrame(
                    columns=[
                        "datetime",
                        column,
                    ]
                ),
                metadata,
            )

        # ----------------------------------------------------
        # Segunda validación usando la historia descargada.
        # ----------------------------------------------------

        validation = (
            _validate_trunk_flow_series(
                selected,
                data.tail(
                    max(
                        FLOW_VALIDATION_DAYS,
                        30,
                    )
                ),
            )
        )

        if not validation[
            "valid"
        ]:

            metadata[
                "status"
            ] = "rechazada_historia"

            metadata[
                "validation"
            ] = validation

            metadata[
                "records"
            ] = 0

            return (
                pd.DataFrame(
                    columns=[
                        "datetime",
                        column,
                    ]
                ),
                metadata,
            )

        result = data.rename(
            columns={
                "value":
                    column
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
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )

        result = result.dropna(
            subset=[
                "datetime",
                column,
            ]
        )

        result = result[
            (
                result[
                    column
                ]
                >= 0
            )
            &
            (
                result[
                    column
                ]
                <= PARANA_TRUNK_MAX_FLOW
            )
        ]

        result = (
            result
            .groupby(
                "datetime",
                as_index=False,
            )[column]
            .mean()
            .sort_values(
                "datetime"
            )
            .reset_index(
                drop=True
            )
        )

        metadata[
            "records"
        ] = int(
            len(
                result
            )
        )

        metadata[
            "status"
        ] = "ok"

        metadata[
            "median_flow_history"
        ] = _safe_float(
            result[
                column
            ].median()
        )

        metadata[
            "last_flow_history"
        ] = _safe_float(
            result[
                column
            ].dropna().iloc[
                -1
            ]
            if result[
                column
            ].notna().any()
            else np.nan
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
        ] = str(
            exc
        )

        metadata[
            "records"
        ] = 0

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
# VALIDAR COHERENCIA ENTRE ESTACIONES VECINAS
# ============================================================

def _median_recent(
    df,
    col,
    days=30,
):

    if (
        df is None
        or df.empty
        or col not in df.columns
    ):

        return np.nan

    values = (
        pd.to_numeric(
            df[
                col
            ],
            errors="coerce",
        )
        .dropna()
        .tail(
            days
        )
    )

    if values.empty:

        return np.nan

    return float(
        values.median()
    )


def _validate_neighbor_consistency(
    base,
    metadata,
):

    """
    Una estación no se rechaza sólo porque sea diferente
    de otra, ya que el Paraná tiene aportes y derivaciones.

    Pero si una serie tiene una magnitud totalmente incompatible
    con las demás series troncal válidas, se descarta.
    """

    station_medians = {}

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        median = _median_recent(
            base,
            col,
            days=30,
        )

        if np.isfinite(
            median
        ):

            station_medians[
                station
            ] = median

    if len(
        station_medians
    ) < 2:

        return (
            base,
            metadata,
        )

    reference_values = list(
        station_medians.values()
    )

    reference_median = float(
        np.median(
            reference_values
        )
    )

    if (
        not np.isfinite(
            reference_median
        )
        or reference_median <= 0
    ):

        return (
            base,
            metadata,
        )

    rejected = []

    for station, station_median in (
        station_medians.items()
    ):

        ratio = (
            station_median
            / reference_median
        )

        if (
            ratio
            < FLOW_NEIGHBOR_RATIO_MIN
            or ratio
            > FLOW_NEIGHBOR_RATIO_MAX
        ):

            col = FLOW_COLUMNS[
                station
            ]

            base[
                col
            ] = np.nan

            rejected.append(
                {
                    "station":
                        station,

                    "median":
                        station_median,

                    "reference_median":
                        reference_median,

                    "ratio":
                        ratio,
                }
            )

            station_meta = (
                metadata[
                    "stations"
                ].get(
                    station,
                    {}
                )
            )

            station_meta[
                "status"
            ] = "rechazada_incoherencia_vecinos"

            station_meta[
                "neighbor_ratio"
            ] = ratio

            station_meta[
                "reference_median"
            ] = reference_median

            metadata[
                "stations"
            ][station] = station_meta

    metadata[
        "neighbor_validation_reference_median"
    ] = reference_median

    metadata[
        "neighbor_rejected"
    ] = rejected

    return (
        base,
        metadata,
    )


# ============================================================
# TODOS LOS CAUDALES
# ============================================================

def get_all_caudales(
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

        "validation":
            "Paraná troncal V11.10.2",

        "stations":
            {},

        "available_stations":
            [],

        "flow_series":
            {},
    }

    for station in STATIONS:

        column = FLOW_COLUMNS[
            station
        ]

        try:

            flow_df, flow_meta = (
                get_caudal_station(
                    station,
                    start,
                    end,
                )
            )

            metadata[
                "stations"
            ][station] = flow_meta

            metadata[
                "flow_series"
            ][station] = flow_meta

            if (
                flow_df is not None
                and not flow_df.empty
            ):

                base = base.merge(
                    flow_df,
                    on="datetime",
                    how="left",
                )

            else:

                base[
                    column
                ] = np.nan

        except Exception as exc:

            base[
                column
            ] = np.nan

            flow_meta = {

                "station":
                    station,

                "status":
                    "error",

                "error":
                    str(
                        exc
                    ),

                "records":
                    0,
            }

            metadata[
                "stations"
            ][station] = flow_meta

            metadata[
                "flow_series"
            ][station] = flow_meta

    # ========================================================
    # VALIDACIÓN ENTRE SERIES
    # ========================================================

    base, metadata = (
        _validate_neighbor_consistency(
            base,
            metadata,
        )
    )

    # ========================================================
    # RECONSTRUIR DISPONIBILIDAD DESPUÉS DE VALIDACIONES
    # ========================================================

    available = []

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        if col not in base.columns:
            continue

        values = (
            pd.to_numeric(
                base[
                    col
                ],
                errors="coerce",
            )
            .dropna()
        )

        if (
            len(
                values
            )
            < MIN_FLOW_OBSERVATIONS
        ):

            continue

        recent_median = float(
            values.tail(
                30
            ).median()
        )

        if (
            not np.isfinite(
                recent_median
            )
            or recent_median
            < PARANA_TRUNK_MIN_MEDIAN_FLOW
        ):

            base[
                col
            ] = np.nan

            station_meta = (
                metadata[
                    "stations"
                ].get(
                    station,
                    {}
                )
            )

            station_meta[
                "status"
            ] = "rechazada_magnitud_final"

            station_meta[
                "recent_median"
            ] = recent_median

            metadata[
                "stations"
            ][station] = station_meta

            continue

        available.append(
            station
        )

    metadata[
        "available_stations"
    ] = available

    return (
        base,
        metadata,
    )


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

    for station in FLOW_PRIORITY:

        col = FLOW_COLUMNS[
            station
        ]

        if col not in df.columns:
            continue

        valid = (
            pd.to_numeric(
                df[
                    col
                ],
                errors="coerce",
            )
            .dropna()
        )

        if (
            len(
                valid
            )
            < MIN_FLOW_OBSERVATIONS
        ):

            continue

        recent = valid.tail(
            30
        )

        recent_median = float(
            recent.median()
        )

        if (
            not np.isfinite(
                recent_median
            )
            or recent_median
            < PARANA_TRUNK_MIN_MEDIAN_FLOW
        ):

            continue

        return (
            col,
            station,
        )

    return (
        None,
        None,
    )


# ============================================================
# PROYECCIÓN DE CAUDAL
# ============================================================

def proyectar_serie_caudal(
    history,
    column,
    future_dates,
):

    if (
        history is None
        or history.empty
        or column not in history.columns
    ):

        return pd.Series(
            np.nan,
            index=range(
                len(
                    future_dates
                )
            ),
            dtype=float,
        )

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
                len(
                    future_dates
                )
            ),
            dtype=float,
        )

    current = float(
        values.iloc[
            -1
        ]
    )

    recent = values.tail(
        min(
            21,
            len(
                values
            ),
        )
    )

    if len(
        recent
    ) >= 4:

        try:

            slope = float(
                np.polyfit(
                    np.arange(
                        len(
                            recent
                        ),
                        dtype=float,
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
        abs(
            current
        )
        * 0.025,
        50.0,
    )

    slope = float(
        np.clip(
            slope,
            -max_slope,
            max_slope,
        )
    )

    projected = []

    level = current

    for day in range(
        1,
        len(
            future_dates
        )
        + 1,
    ):

        damping = np.exp(
            -day
            / 18.0
        )

        change = (
            slope
            * damping
        )

        level = max(
            0.0,
            level
            + change,
        )

        projected.append(
            level
        )

    return pd.Series(
        projected,
        dtype=float,
    )


# ============================================================
# FEATURES LLUVIA
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

        values = pd.to_numeric(
            result[
                col
            ],
            errors="coerce",
        )

        if not values.notna().any():
            continue

        values = (
            values
            .fillna(
                0.0
            )
            .clip(
                lower=0.0
            )
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
# FEATURES CAUDAL
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
            result[
                col
            ],
            errors="coerce",
        )

        if (
            q.notna().sum()
            < MIN_FLOW_OBSERVATIONS
        ):

            continue

        result[
            f"{col}_diff_1"
        ] = q.diff(
            1
        )

        result[
            f"{col}_diff_3"
        ] = q.diff(
            3
        )

        result[
            f"{col}_diff_7"
        ] = q.diff(
            7
        )

        result[
            f"{col}_mean_3"
        ] = (
            q
            .rolling(
                3,
                min_periods=1,
            )
            .mean()
        )

        result[
            f"{col}_mean_7"
        ] = (
            q
            .rolling(
                7,
                min_periods=2,
            )
            .mean()
        )

        result[
            f"{col}_mean_14"
        ] = (
            q
            .rolling(
                14,
                min_periods=3,
            )
            .mean()
        )

        result[
            f"{col}_relative_7"
        ] = (
            q.diff(
                7
            )
            /
            q.shift(
                7
            ).replace(
                0,
                np.nan,
            )
        )

    return result


# ============================================================
# COBERTURA
# ============================================================

def _column_coverage(
    df,
    columns,
):

    output = {}

    for col in columns:

        if (
            df is not None
            and not df.empty
            and col in df.columns
        ):

            output[
                col
            ] = int(
                pd.to_numeric(
                    df[
                        col
                    ],
                    errors="coerce",
                )
                .notna()
                .sum()
            )

        else:

            output[
                col
            ] = 0

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
            int(
                forecast_days
            ),
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
    # CAUDAL HISTÓRICO VALIDADO
    # ========================================================

    flow_history, flow_meta = (
        get_all_caudales(
            start,
            end,
        )
    )

    # ========================================================
    # BASE
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

    if (
        rain_history is not None
        and not rain_history.empty
    ):

        history = history.merge(
            rain_history,
            on="datetime",
            how="left",
        )

    if (
        flow_history is not None
        and not flow_history.empty
    ):

        flow_cols = [
            col
            for col in flow_history.columns
            if col != "datetime"
        ]

        history = history.merge(

            flow_history[
                [
                    "datetime",
                    *flow_cols,
                ]
            ],

            on="datetime",

            how="left",
        )

    # ========================================================
    # GARANTIZAR COLUMNAS
    # ========================================================

    for station in STATIONS:

        rain_col = RAIN_COLUMNS[
            station
        ]

        flow_col = FLOW_COLUMNS[
            station
        ]

        if rain_col not in history.columns:

            history[
                rain_col
            ] = np.nan

        if flow_col not in history.columns:

            history[
                flow_col
            ] = np.nan

    # ========================================================
    # INTERPOLACIÓN LIMITADA
    # ========================================================

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        q = pd.to_numeric(
            history[
                col
            ],
            errors="coerce",
        )

        if (
            q.notna().sum()
            >= MIN_FLOW_OBSERVATIONS
        ):

            history[
                col
            ] = (
                q
                .interpolate(
                    limit=5,
                    limit_area="inside",
                )
            )

    # ========================================================
    # CAUDAL PRINCIPAL
    # ========================================================

    (
        main_flow_col,
        main_flow_station,
    ) = elegir_caudal_principal(
        history
    )

    if main_flow_col is not None:

        history[
            "caudal_m3s"
        ] = pd.to_numeric(
            history[
                main_flow_col
            ],
            errors="coerce",
        )

    else:

        history[
            "caudal_m3s"
        ] = np.nan

    # ========================================================
    # PRECIPITACIÓN LEGACY
    # ========================================================

    if "precip_mm" not in history.columns:

        rain_cols = [
            RAIN_COLUMNS[
                station
            ]
            for station in STATIONS
        ]

        history[
            "precip_mm"
        ] = (
            history[
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

    # ========================================================
    # FEATURES
    # ========================================================

    history = agregar_features_lluvia(
        history
    )

    history = agregar_features_caudal(
        history
    )

    history[
        "datetime"
    ] = _to_naive_datetime(
        history[
            "datetime"
        ]
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
    # FUTURO LLUVIA
    # ========================================================

    rain_future, rain_future_meta = (
        get_rain_forecast(
            end,
            forecast_days,
        )
    )

    future = rain_future.copy()

    # ========================================================
    # FUTURO CAUDAL
    # ========================================================

    future_dates = future[
        "datetime"
    ]

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        valid_count = int(
            pd.to_numeric(
                history[
                    col
                ],
                errors="coerce",
            )
            .notna()
            .sum()
        )

        if (
            valid_count
            >= MIN_FLOW_OBSERVATIONS
        ):

            future[
                col
            ] = proyectar_serie_caudal(

                history,

                col,

                future_dates,
            )

        else:

            future[
                col
            ] = np.nan

    if (
        main_flow_col is not None
        and main_flow_col in future.columns
    ):

        future[
            "caudal_m3s"
        ] = pd.to_numeric(
            future[
                main_flow_col
            ],
            errors="coerce",
        )

    else:

        future[
            "caudal_m3s"
        ] = np.nan

    # ========================================================
    # FEATURES FUTURAS
    # ========================================================

    raw_cols = [
        "datetime",
        "precip_mm",
        "caudal_m3s",
    ]

    for station in STATIONS:

        raw_cols.append(
            RAIN_COLUMNS[
                station
            ]
        )

        raw_cols.append(
            FLOW_COLUMNS[
                station
            ]
        )

    history_cols = [
        col
        for col in raw_cols
        if col in history.columns
    ]

    future_cols = [
        col
        for col in raw_cols
        if col in future.columns
    ]

    combined = pd.concat(

        [
            history[
                history_cols
            ].tail(
                45
            ),

            future[
                future_cols
            ],
        ],

        ignore_index=True,

        sort=False,
    )

    combined = agregar_features_lluvia(
        combined
    )

    combined = agregar_features_caudal(
        combined
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
        )
        + pd.Timedelta(
            days=1
        )
    ).normalize()

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
            subset=[
                "datetime"
            ],
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
    # COBERTURA
    # ========================================================

    rain_base_columns = [
        RAIN_COLUMNS[
            station
        ]
        for station in STATIONS
    ]

    flow_base_columns = [
        FLOW_COLUMNS[
            station
        ]
        for station in STATIONS
    ]

    rain_coverage = _column_coverage(
        history,
        rain_base_columns,
    )

    flow_coverage = _column_coverage(
        history,
        flow_base_columns,
    )

    available_rain_stations = [
        station
        for station in STATIONS
        if rain_coverage.get(
            RAIN_COLUMNS[
                station
            ],
            0,
        ) > 0
    ]

    available_flow_stations = [
        station
        for station in STATIONS
        if flow_coverage.get(
            FLOW_COLUMNS[
                station
            ],
            0,
        ) >= MIN_FLOW_OBSERVATIONS
    ]

    # ========================================================
    # MEDIANAS / ÚLTIMOS CAUDALES
    # ========================================================

    flow_summary = {}

    for station in available_flow_stations:

        col = FLOW_COLUMNS[
            station
        ]

        values = (
            pd.to_numeric(
                history[
                    col
                ],
                errors="coerce",
            )
            .dropna()
        )

        if values.empty:
            continue

        flow_summary[
            station
        ] = {

            "column":
                col,

            "records":
                int(
                    len(
                        values
                    )
                ),

            "median_30d":
                float(
                    values.tail(
                        30
                    ).median()
                ),

            "last":
                float(
                    values.iloc[
                        -1
                    ]
                ),
        }

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

        "caudal_source":
            (
                "INA A5 · Paraná troncal validado"
                if available_flow_stations
                else "Sin serie INA de caudal troncal validada"
            ),

        "rain_stations":
            available_rain_stations,

        "flow_stations":
            available_flow_stations,

        "rain_station_count":
            len(
                available_rain_stations
            ),

        "flow_station_count":
            len(
                available_flow_stations
            ),

        "main_flow_station":
            main_flow_station,

        "main_flow_column":
            main_flow_col,

        "flow_summary":
            flow_summary,

        "rain_history":
            rain_meta,

        "rain_forecast":
            rain_future_meta,

        "flow_history":
            flow_meta,

        "flow_series":
            flow_meta.get(
                "flow_series",
                {}
            ),

        "rain_coverage":
            rain_coverage,

        "flow_coverage":
            flow_coverage,

        "history_rows":
            int(
                len(
                    history
                )
            ),

        "future_rows":
            int(
                len(
                    future
                )
            ),

        "history_start":
            (
                history[
                    "datetime"
                ].min()
                if not history.empty
                else None
            ),

        "history_end":
            (
                history[
                    "datetime"
                ].max()
                if not history.empty
                else None
            ),

        "flow_validation":
            {

                "min_median_m3s":
                    PARANA_TRUNK_MIN_MEDIAN_FLOW,

                "min_recent_m3s":
                    PARANA_TRUNK_MIN_RECENT_FLOW,

                "max_m3s":
                    PARANA_TRUNK_MAX_FLOW,

                "neighbor_ratio_min":
                    FLOW_NEIGHBOR_RATIO_MIN,

                "neighbor_ratio_max":
                    FLOW_NEIGHBOR_RATIO_MAX,
            },

        "real_weather_forecast_days":
            OPEN_METEO_REAL_FORECAST_DAYS,

        "rain_after_real_forecast":
            (
                "No se interpreta como pronóstico "
                "meteorológico determinista después "
                "del horizonte disponible."
            ),
    }

    return (
        history,
        future,
        metadata,
    )


# ============================================================
# DIAGNÓSTICO COMPLETO
# ============================================================

def diagnostic(
    start,
    end,
):

    result = {

        "version":
            VERSION,

        "start":
            _normalize_date(
                start
            ),

        "end":
            _normalize_date(
                end
            ),

        "status":
            "pendiente",
    }

    # ========================================================
    # CATÁLOGO
    # ========================================================

    try:

        catalog = get_ina_catalog()

        caudal_catalog = catalog[
            catalog[
                "var_id"
            ]
            == VAR_ID_CAUDAL
        ]

        result[
            "ina_catalog_records"
        ] = int(
            len(
                catalog
            )
        )

        result[
            "ina_caudal_series"
        ] = int(
            len(
                caudal_catalog
            )
        )

    except Exception as exc:

        result[
            "ina_catalog_error"
        ] = str(
            exc
        )

    # ========================================================
    # CANDIDATOS Y VALIDACIÓN
    # ========================================================

    station_results = {}

    for station in STATIONS:

        info = {}

        try:

            candidates = (
                candidatos_caudal_estacion(
                    station,
                    start,
                    end,
                )
            )

            info[
                "candidate_count"
            ] = int(
                len(
                    candidates
                )
            )

            if not candidates.empty:

                preview_cols = [
                    col
                    for col in [
                        "series_id",
                        "nombre",
                        "rio",
                        "timestart",
                        "timeend",
                        "count",
                        "is_parana_trunk",
                        "score",
                    ]
                    if col
                    in candidates.columns
                ]

                info[
                    "top_candidates"
                ] = (
                    candidates[
                        preview_cols
                    ]
                    .head(
                        5
                    )
                    .to_dict(
                        orient="records"
                    )
                )

            selected, meta = (
                seleccionar_serie_caudal(
                    station,
                    start,
                    end,
                )
            )

            info[
                "selection"
            ] = meta

            if selected is not None:

                info[
                    "selected_series_id"
                ] = selected.get(
                    "series_id"
                )

                info[
                    "selected_name"
                ] = selected.get(
                    "nombre"
                )

                info[
                    "selected_river"
                ] = selected.get(
                    "rio"
                )

        except Exception as exc:

            info[
                "error"
            ] = str(
                exc
            )

        station_results[
            station
        ] = info

    result[
        "stations"
    ] = station_results

    # ========================================================
    # TEST FINAL
    # ========================================================

    try:

        (
            history,
            future,
            metadata,
        ) = get_exogenous_data(
            start,
            end,
            forecast_days=60,
        )

        result[
            "status"
        ] = "ok"

        result[
            "history_rows"
        ] = int(
            len(
                history
            )
        )

        result[
            "future_rows"
        ] = int(
            len(
                future
            )
        )

        result[
            "rain_stations"
        ] = metadata.get(
            "rain_stations",
            []
        )

        result[
            "flow_stations"
        ] = metadata.get(
            "flow_stations",
            []
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
            "flow_summary"
        ] = metadata.get(
            "flow_summary",
            {}
        )

        result[
            "rain_coverage"
        ] = metadata.get(
            "rain_coverage",
            {}
        )

        result[
            "flow_coverage"
        ] = metadata.get(
            "flow_coverage",
            {}
        )

        result[
            "neighbor_rejected"
        ] = (
            metadata.get(
                "flow_history",
                {}
            )
            .get(
                "neighbor_rejected",
                []
            )
        )

        result[
            "rain_columns_present"
        ] = [
            col
            for col
            in RAIN_COLUMNS.values()
            if (
                col in history.columns
                and pd.to_numeric(
                    history[
                        col
                    ],
                    errors="coerce",
                )
                .notna()
                .any()
            )
        ]

        result[
            "flow_columns_present"
        ] = [
            col
            for col
            in FLOW_COLUMNS.values()
            if (
                col in history.columns
                and pd.to_numeric(
                    history[
                        col
                    ],
                    errors="coerce",
                )
                .notna()
                .any()
            )
        ]

    except Exception as exc:

        result[
            "status"
        ] = "error"

        result[
            "error"
        ] = str(
            exc
        )

    return result
