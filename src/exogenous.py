# ============================================================
# PARANÁ · SAN NICOLÁS
# src/exogenous.py
# V11.11.1 COMPLETO
#
# VARIABLES EXÓGENAS
#
# Incluye:
# - lluvia histórica Open-Meteo
# - lluvia pronosticada Open-Meteo
# - caudales INA A5
# - validación de series de caudal
# - reconstrucción de caudales faltantes
# - trazabilidad de origen
# - calidad de cada dato de caudal
# - proyección de caudales
#
# IMPORTANTE:
# Un dato reconstruido o proyectado NUNCA se marca como
# "observado".
#
# API:
#
# get_exogenous_data(
#     start,
#     end,
#     forecast_days=15
# )
#
# Devuelve:
#     history
#     future
#     metadata
#
# ============================================================

from functools import lru_cache

import numpy as np
import pandas as pd
import requests


# ============================================================
# VERSIÓN
# ============================================================

VERSION = "V11.11.1"


# ============================================================
# CONFIGURACIÓN GENERAL
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
# LÍMITES FÍSICOS / VALIDACIÓN
# ============================================================

PARANA_TRUNK_MIN_MEDIAN_FLOW = 500.0

PARANA_TRUNK_MIN_RECENT_FLOW = 250.0

PARANA_TRUNK_MAX_FLOW = 100000.0

ESTIMATED_FLOW_MIN = 300.0

ESTIMATED_FLOW_MAX = 100000.0

RATIO_MIN = 0.10

RATIO_MAX = 8.00


# ============================================================
# FUENTES
# ============================================================

OPEN_METEO_ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

OPEN_METEO_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

INA_BASE_URL = (
    "https://alerta.ina.gob.ar/a5"
)

INA_SERIES_GEOJSON_URL = (
    INA_BASE_URL
    + "/obs/puntual/series"
)

INA_OBSERVATIONS_URL = (
    INA_BASE_URL
    + "/getObservaciones"
)

VAR_ID_CAUDAL = 4


# ============================================================
# ESTACIONES
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
# PUNTOS DE LLUVIA
# ============================================================

RAIN_POINTS = {

    "Corrientes": {
        "lat": -27.4692,
        "lon": -58.8306,
    },

    "Goya": {
        "lat": -29.1400,
        "lon": -59.2626,
    },

    "La Paz": {
        "lat": -30.7449,
        "lon": -59.6457,
    },

    "Paraná": {
        "lat": -31.7319,
        "lon": -60.5238,
    },

    "Diamante": {
        "lat": -32.0664,
        "lon": -60.6384,
    },

    "Rosario": {
        "lat": -32.9468,
        "lon": -60.6393,
    },

    "Villa Constitución": {
        "lat": -33.2272,
        "lon": -60.3297,
    },

    "San Nicolás": {
        "lat": -33.3358,
        "lon": -60.2252,
    },
}


# ============================================================
# COLUMNAS DE LLUVIA
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


# ============================================================
# COLUMNAS DE CAUDAL
# ============================================================

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
# ALIAS DE ESTACIONES
# ============================================================

STATION_ALIASES = {

    "Corrientes": [
        "corrientes",
    ],

    "Goya": [
        "goya",
    ],

    "La Paz": [
        "la paz",
        "lapaz",
    ],

    "Paraná": [
        "parana",
        "paraná",
    ],

    "Diamante": [
        "diamante",
    ],

    "Rosario": [
        "rosario",
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
        "s nicolas",
        "s. nicolas",
    ],
}


# ============================================================
# TÉRMINOS QUE DESCARTAN SERIES NO REPRESENTATIVAS
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


BAD_STATION_TERMS = [
    "meteorologica",
    "meteorológica",
    "meteo",
    "inta",
    "escuela",
    "aeropuerto",
    "pluviometro",
    "pluviómetro",
]


# ============================================================
# PESOS DEL CORREDOR
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
# UTILIDADES
# ============================================================

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
    default=None,
):

    try:

        return int(
            float(value)
        )

    except Exception:

        return default


def _numeric(
    series,
):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _datetime_naive(
    values,
):

    return (
        pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
    )


def _normalize_date(
    value,
):

    return pd.Timestamp(
        value
    ).strftime(
        "%Y-%m-%d"
    )


def _normalize_text(
    value,
):

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
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
            point[
                "lat"
            ],

        "longitude":
            point[
                "lon"
            ],

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
            "America/Argentina/Buenos_Aires",
    }

    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
        timeout=
            REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    daily = data.get(
        "daily",
        {}
    )

    dates = daily.get(
        "time",
        []
    )

    rain = daily.get(
        "precipitation_sum",
        []
    )

    col = RAIN_COLUMNS[
        station
    ]

    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    dates,
                    errors="coerce",
                ),

            col:
                pd.to_numeric(
                    rain,
                    errors="coerce",
                ),
        }
    )

    result[
        "datetime"
    ] = _datetime_naive(
        result[
            "datetime"
        ]
    )

    result[col] = (
        _numeric(
            result[col]
        )
        .fillna(0.0)
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

    result = None

    metadata = {
        "stations": {},
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

            metadata[
                "stations"
            ][
                station
            ] = {
                "status":
                    "ok",

                "records":
                    int(
                        station_df[
                            col
                        ]
                        .notna()
                        .sum()
                    ),
            }

        except Exception as exc:

            station_df = pd.DataFrame(
                columns=[
                    "datetime",
                    col,
                ]
            )

            metadata[
                "stations"
            ][
                station
            ] = {
                "status":
                    "error",

                "error":
                    str(exc),
            }

        if result is None:

            result = station_df.copy()

        else:

            result = result.merge(
                station_df,
                on="datetime",
                how="outer",
            )

    if result is None:

        result = pd.DataFrame(
            columns=[
                "datetime"
            ]
        )

    result[
        "datetime"
    ] = _datetime_naive(
        result[
            "datetime"
        ]
    )

    result = (
        result
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    for col in RAIN_COLUMNS.values():

        if col not in result.columns:
            result[col] = 0.0

        result[col] = (
            _numeric(
                result[col]
            )
            .fillna(0.0)
            .clip(
                lower=0.0
            )
        )

    result[
        "precip_mm"
    ] = result[
        list(
            RAIN_COLUMNS.values()
        )
    ].mean(
        axis=1
    )

    return (
        result,
        metadata,
    )


# ============================================================
# PRONÓSTICO DE LLUVIA POR ESTACIÓN
# ============================================================

def _get_rain_forecast_station(
    station,
    days,
):

    point = RAIN_POINTS[
        station
    ]

    request_days = int(
        np.clip(
            days,
            1,
            OPEN_METEO_REAL_FORECAST_DAYS,
        )
    )

    params = {
        "latitude":
            point[
                "lat"
            ],

        "longitude":
            point[
                "lon"
            ],

        "daily":
            "precipitation_sum",

        "forecast_days":
            request_days,

        "timezone":
            "America/Argentina/Buenos_Aires",
    }

    response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params=params,
        timeout=
            REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    daily = data.get(
        "daily",
        {}
    )

    dates = daily.get(
        "time",
        []
    )

    rain = daily.get(
        "precipitation_sum",
        []
    )

    col = RAIN_COLUMNS[
        station
    ]

    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    dates,
                    errors="coerce",
                ),

            col:
                pd.to_numeric(
                    rain,
                    errors="coerce",
                ),
        }
    )

    result[
        "datetime"
    ] = _datetime_naive(
        result[
            "datetime"
        ]
    )

    result[col] = (
        _numeric(
            result[col]
        )
        .fillna(0.0)
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
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LLUVIA FUTURA DEL CORREDOR
# ============================================================

def get_rain_forecast(
    days=15,
):

    days = int(
        np.clip(
            days,
            1,
            MAX_FORECAST_DAYS,
        )
    )

    result = None

    metadata = {
        "stations": {},
        "real_forecast_days":
            OPEN_METEO_REAL_FORECAST_DAYS,
    }

    for station in STATIONS:

        col = RAIN_COLUMNS[
            station
        ]

        try:

            station_df = (
                _get_rain_forecast_station(
                    station,
                    min(
                        days,
                        OPEN_METEO_REAL_FORECAST_DAYS,
                    ),
                )
            )

            metadata[
                "stations"
            ][
                station
            ] = {
                "status":
                    "ok",

                "records":
                    int(
                        station_df[
                            col
                        ]
                        .notna()
                        .sum()
                    ),
            }

        except Exception as exc:

            station_df = pd.DataFrame(
                columns=[
                    "datetime",
                    col,
                ]
            )

            metadata[
                "stations"
            ][
                station
            ] = {
                "status":
                    "error",

                "error":
                    str(exc),
            }

        if result is None:

            result = station_df.copy()

        else:

            result = result.merge(
                station_df,
                on="datetime",
                how="outer",
            )

    if result is None:

        result = pd.DataFrame(
            columns=[
                "datetime"
            ]
        )

    today = pd.Timestamp.today().normalize()

    full_dates = pd.DataFrame(
        {
            "datetime":
                pd.date_range(
                    start=today,
                    periods=days,
                    freq="D",
                )
        }
    )

    result = full_dates.merge(
        result,
        on="datetime",
        how="left",
    )

    for col in RAIN_COLUMNS.values():

        if col not in result.columns:
            result[col] = np.nan

        # ----------------------------------------------------
        # Sólo los primeros días son pronóstico meteorológico
        # real.
        #
        # Para compatibilidad con la app se completa el resto
        # con 0, pero el modelo V11.11 no debe interpretarlo
        # como "ausencia determinística de lluvia".
        # ----------------------------------------------------

        result[col] = (
            _numeric(
                result[col]
            )
            .fillna(0.0)
            .clip(
                lower=0.0
            )
        )

    result[
        "precip_mm"
    ] = result[
        list(
            RAIN_COLUMNS.values()
        )
    ].mean(
        axis=1
    )

    metadata[
        "warning"
    ] = (
        "Open-Meteo aporta pronóstico meteorológico sólo "
        "para el horizonte disponible. Los días posteriores "
        "se mantienen en cero por compatibilidad y deben "
        "interpretarse como sin pronóstico meteorológico, "
        "no como lluvia futura nula."
    )

    return (
        result,
        metadata,
    )


# ============================================================
# CATÁLOGO INA A5
# ============================================================

@lru_cache(
    maxsize=1
)
def get_ina_catalog():

    response = requests.get(
        INA_SERIES_GEOJSON_URL,
        params={
            "format":
                "geojson"
        },
        timeout=
            REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    features = data.get(
        "features",
        []
    )

    rows = []

    for feature in features:

        properties = feature.get(
            "properties",
            {}
        ) or {}

        geometry = feature.get(
            "geometry",
            {}
        ) or {}

        coordinates = geometry.get(
            "coordinates",
            []
        ) or []

        lon = np.nan
        lat = np.nan

        if (
            isinstance(
                coordinates,
                list,
            )
            and len(
                coordinates
            )
            >= 2
        ):
            lon = _safe_float(
                coordinates[0]
            )

            lat = _safe_float(
                coordinates[1]
            )

        rows.append(
            {
                "series_id":
                    properties.get(
                        "series_id"
                    ),

                "id":
                    properties.get(
                        "id"
                    ),

                "nombre":
                    properties.get(
                        "nombre"
                    ),

                "estacion_id":
                    properties.get(
                        "estacion_id"
                    ),

                "rio":
                    properties.get(
                        "rio"
                    ),

                "var_id":
                    properties.get(
                        "var_id"
                    ),

                "proc_id":
                    properties.get(
                        "proc_id"
                    ),

                "unit_id":
                    properties.get(
                        "unit_id"
                    ),

                "var_nombre":
                    properties.get(
                        "var_nombre"
                    ),

                "timestart":
                    properties.get(
                        "timestart"
                    ),

                "timeend":
                    properties.get(
                        "timeend"
                    ),

                "count":
                    properties.get(
                        "count"
                    ),

                "data_availability":
                    properties.get(
                        "data_availability"
                    ),

                "fuente":
                    properties.get(
                        "fuente"
                    ),

                "public":
                    properties.get(
                        "public"
                    ),

                "lat":
                    lat,

                "lon":
                    lon,
            }
        )

    catalog = pd.DataFrame(
        rows
    )

    if catalog.empty:

        return catalog

    catalog[
        "series_id"
    ] = pd.to_numeric(
        catalog[
            "series_id"
        ],
        errors="coerce",
    )

    catalog[
        "var_id"
    ] = pd.to_numeric(
        catalog[
            "var_id"
        ],
        errors="coerce",
    )

    catalog[
        "proc_id"
    ] = pd.to_numeric(
        catalog[
            "proc_id"
        ],
        errors="coerce",
    )

    catalog[
        "count"
    ] = pd.to_numeric(
        catalog[
            "count"
        ],
        errors="coerce",
    )

    catalog[
        "timestart"
    ] = pd.to_datetime(
        catalog[
            "timestart"
        ],
        errors="coerce",
        utc=True,
    )

    catalog[
        "timeend"
    ] = pd.to_datetime(
        catalog[
            "timeend"
        ],
        errors="coerce",
        utc=True,
    )

    return catalog


# ============================================================
# PUNTAJE DE ESTACIÓN
# ============================================================

def _station_match_score(
    station,
    name,
):

    text = _normalize_text(
        name
    )

    if not text:
        return 0.0

    aliases = STATION_ALIASES[
        station
    ]

    score = 0.0

    for alias in aliases:

        alias_norm = _normalize_text(
            alias
        )

        if text == alias_norm:
            score = max(
                score,
                100.0,
            )

        elif text.startswith(
            alias_norm
        ):
            score = max(
                score,
                85.0,
            )

        elif (
            f" {alias_norm} "
            in f" {text} "
        ):
            score = max(
                score,
                75.0,
            )

        elif alias_norm in text:
            score = max(
                score,
                55.0,
            )

    for bad in BAD_STATION_TERMS:

        if _normalize_text(
            bad
        ) in text:

            score -= 40.0

    return score


# ============================================================
# VALIDACIÓN DEL RÍO
# ============================================================

def _parana_score(
    river,
):

    text = _normalize_text(
        river
    )

    if not text:
        return 0.0

    score = 0.0

    if "parana" in text:
        score += 50.0

    for term in NON_TRUNK_TERMS:

        if _normalize_text(
            term
        ) in text:

            score -= 80.0

    return score


def _is_parana_trunk_candidate(
    river,
):

    text = _normalize_text(
        river
    )

    if not text:
        return True

    if "parana" not in text:
        return False

    for term in NON_TRUNK_TERMS:

        if _normalize_text(
            term
        ) in text:

            return False

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

    if catalog.empty:

        return pd.DataFrame()

    x = catalog[
        catalog[
            "var_id"
        ]
        == VAR_ID_CAUDAL
    ].copy()

    if x.empty:

        return x

    x[
        "station_score"
    ] = x[
        "nombre"
    ].apply(
        lambda value:
            _station_match_score(
                station,
                value,
            )
    )

    # --------------------------------------------------------
    # Exigimos coincidencia real con el nombre.
    # --------------------------------------------------------

    x = x[
        x[
            "station_score"
        ] > 0
    ].copy()

    if x.empty:

        return x

    x[
        "river_score"
    ] = x[
        "rio"
    ].apply(
        _parana_score
    )

    x[
        "count_score"
    ] = (
        pd.to_numeric(
            x[
                "count"
            ],
            errors="coerce",
        )
        .fillna(0)
        .clip(
            upper=100000
        )
        / 10000.0
    )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    recency_days = (
        now
        - x[
            "timeend"
        ]
    ).dt.days

    x[
        "recency_score"
    ] = (
        30.0
        -
        recency_days
        .fillna(
            3650
        )
        .clip(
            lower=0,
            upper=3650,
        )
        / 120.0
    )

    x[
        "proc_score"
    ] = (
        x[
            "proc_id"
        ]
        .fillna(0)
        .apply(
            lambda value:
                5.0
                if value == 1
                else 0.0
        )
    )

    overlap_score = pd.Series(
        0.0,
        index=x.index,
    )

    if (
        start is not None
        and end is not None
    ):

        start_ts = pd.Timestamp(
            start,
            tz="UTC",
        )

        end_ts = pd.Timestamp(
            end,
            tz="UTC",
        )

        overlap = (
            (
                x[
                    "timeend"
                ].isna()
                |
                (
                    x[
                        "timeend"
                    ]
                    >= start_ts
                )
            )
            &
            (
                x[
                    "timestart"
                ].isna()
                |
                (
                    x[
                        "timestart"
                    ]
                    <= end_ts
                )
            )
        )

        overlap_score.loc[
            overlap
        ] = 20.0

    x[
        "score"
    ] = (
        x[
            "station_score"
        ]
        +
        x[
            "river_score"
        ]
        +
        x[
            "count_score"
        ]
        +
        x[
            "recency_score"
        ]
        +
        x[
            "proc_score"
        ]
        +
        overlap_score
    )

    return (
        x.sort_values(
            [
                "score",
                "count",
                "timeend",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# EXTRAER REGISTROS DE LA RESPUESTA INA
# ============================================================

def _extract_records(
    data,
):

    if data is None:

        return []

    if isinstance(
        data,
        list,
    ):

        if not data:
            return []

        if all(
            isinstance(
                item,
                dict,
            )
            for item in data
        ):

            # Si parecen observaciones
            keys = set()

            for item in data[:5]:
                keys.update(
                    item.keys()
                )

            observation_keys = {
                "timestart",
                "time",
                "timestamp",
                "fecha",
                "valor",
                "value",
            }

            if keys.intersection(
                observation_keys
            ):
                return data

        for item in data:

            records = _extract_records(
                item
            )

            if records:
                return records

        return []

    if isinstance(
        data,
        dict,
    ):

        for key in [
            "observaciones",
            "observations",
            "data",
            "records",
            "result",
            "items",
        ]:

            if key in data:

                records = _extract_records(
                    data[
                        key
                    ]
                )

                if records:
                    return records

        for value in data.values():

            records = _extract_records(
                value
            )

            if records:
                return records

    return []


# ============================================================
# CONSULTA A5 DE UNA SERIE
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
        timeout=
            REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    records = _extract_records(
        data
    )

    rows = []

    for record in records:

        if not isinstance(
            record,
            dict,
        ):
            continue

        timestamp = None

        for key in [
            "timestart",
            "time",
            "timestamp",
            "fecha",
            "datetime",
        ]:

            if key in record:

                timestamp = record[
                    key
                ]

                break

        value = None

        for key in [
            "valor",
            "value",
            "val",
        ]:

            if key in record:

                value = record[
                    key
                ]

                break

        if (
            timestamp is None
            or value is None
        ):
            continue

        rows.append(
            {
                "datetime":
                    timestamp,

                "value":
                    value,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )

    result[
        "datetime"
    ] = _datetime_naive(
        result[
            "datetime"
        ]
    )

    result[
        "value"
    ] = _numeric(
        result[
            "value"
        ]
    )

    result = result.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )

    # --------------------------------------------------------
    # Sólo valores físicos positivos.
    # --------------------------------------------------------

    result = result[
        (
            result[
                "value"
            ] > 0
        )
        &
        (
            result[
                "value"
            ]
            <= PARANA_TRUNK_MAX_FLOW
        )
    ]

    result[
        "datetime"
    ] = (
        result[
            "datetime"
        ]
        .dt
        .normalize()
    )

    result = (
        result.groupby(
            "datetime",
            as_index=False,
        )[
            "value"
        ]
        .mean()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# ESTADÍSTICAS DE CAUDAL
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
            "records": 0,
            "median": np.nan,
            "recent": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
        }

    values = (
        _numeric(
            df[
                "value"
            ]
        )
        .dropna()
    )

    if values.empty:

        return {
            "records": 0,
            "median": np.nan,
            "recent": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
        }

    return {
        "records":
            int(
                len(values)
            ),

        "median":
            float(
                values.median()
            ),

        "recent":
            float(
                values.iloc[-1]
            ),

        "minimum":
            float(
                values.min()
            ),

        "maximum":
            float(
                values.max()
            ),
    }


# ============================================================
# VALIDACIÓN DE CAUDAL PRINCIPAL
# ============================================================

def _validate_trunk_flow_series(
    station,
    candidate,
    observations,
):

    stats = _flow_statistics(
        observations
    )

    if stats[
        "records"
    ] < MIN_FLOW_OBSERVATIONS:

        return (
            False,
            "menos de 3 observaciones",
            stats,
        )

    river = candidate.get(
        "rio"
    )

    if not _is_parana_trunk_candidate(
        river
    ):

        return (
            False,
            "no corresponde al cauce principal del Paraná",
            stats,
        )

    median = stats[
        "median"
    ]

    recent = stats[
        "recent"
    ]

    maximum = stats[
        "maximum"
    ]

    if (
        not np.isfinite(
            median
        )
        or median
        < PARANA_TRUNK_MIN_MEDIAN_FLOW
    ):

        return (
            False,
            "caudal mediano demasiado bajo para el Paraná principal",
            stats,
        )

    if (
        not np.isfinite(
            recent
        )
        or recent
        < PARANA_TRUNK_MIN_RECENT_FLOW
    ):

        return (
            False,
            "caudal reciente demasiado bajo para el Paraná principal",
            stats,
        )

    if (
        np.isfinite(
            maximum
        )
        and maximum
        > PARANA_TRUNK_MAX_FLOW
    ):

        return (
            False,
            "caudal máximo fuera de rango físico",
            stats,
        )

    return (
        True,
        "ok",
        stats,
    )


# ============================================================
# VENTANA DE VALIDACIÓN
# ============================================================

def _candidate_validation_window(
    candidate,
    requested_start,
    requested_end,
):

    requested_start = pd.Timestamp(
        requested_start
    )

    requested_end = pd.Timestamp(
        requested_end
    )

    catalog_start = candidate.get(
        "timestart"
    )

    catalog_end = candidate.get(
        "timeend"
    )

    if pd.notna(
        catalog_start
    ):

        catalog_start = (
            pd.Timestamp(
                catalog_start
            )
            .tz_convert(None)
        )

    else:

        catalog_start = requested_start

    if pd.notna(
        catalog_end
    ):

        catalog_end = (
            pd.Timestamp(
                catalog_end
            )
            .tz_convert(None)
        )

    else:

        catalog_end = requested_end

    end = min(
        requested_end,
        catalog_end,
    )

    start = max(
        requested_start,
        catalog_start,
        end
        - pd.Timedelta(
            days=
                FLOW_VALIDATION_DAYS
        ),
    )

    if start > end:

        return (
            None,
            None,
        )

    return (
        start,
        end,
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

        "status":
            "sin_serie",

        "tested":
            [],
    }

    if candidates.empty:

        metadata[
            "reason"
        ] = (
            "No hay candidatos de caudal en el catálogo INA."
        )

        return (
            None,
            metadata,
        )

    for _, candidate in (
        candidates
        .head(25)
        .iterrows()
    ):

        series_id = _safe_int(
            candidate.get(
                "series_id"
            )
        )

        if series_id is None:
            continue

        validation_start, validation_end = (
            _candidate_validation_window(
                candidate,
                start,
                end,
            )
        )

        if (
            validation_start is None
            or validation_end is None
        ):

            metadata[
                "tested"
            ].append(
                {
                    "series_id":
                        series_id,

                    "name":
                        candidate.get(
                            "nombre"
                        ),

                    "status":
                        "sin superposición temporal",
                }
            )

            continue

        try:

            observations = (
                query_caudal_series(
                    series_id,
                    validation_start,
                    validation_end,
                )
            )

            valid, reason, stats = (
                _validate_trunk_flow_series(
                    station,
                    candidate,
                    observations,
                )
            )

            metadata[
                "tested"
            ].append(
                {
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

                    "valid":
                        valid,

                    "reason":
                        reason,

                    **stats,
                }
            )

            if valid:

                metadata.update(
                    {
                        "status":
                            "ok",

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

                        "stats":
                            stats,
                    }
                )

                return (
                    candidate.to_dict(),
                    metadata,
                )

        except Exception as exc:

            metadata[
                "tested"
            ].append(
                {
                    "series_id":
                        series_id,

                    "name":
                        candidate.get(
                            "nombre"
                        ),

                    "valid":
                        False,

                    "reason":
                        str(exc),
                }
            )

    metadata[
        "reason"
    ] = (
        "Ningún candidato superó la validación de caudal."
    )

    return (
        None,
        metadata,
    )


# ============================================================
# DESCARGA DE HISTORIA POR BLOQUES
# ============================================================

def _query_history_blocks(
    series_id,
    start,
    end,
):

    start_ts = pd.Timestamp(
        start
    ).normalize()

    end_ts = pd.Timestamp(
        end
    ).normalize()

    frames = []

    block_start = start_ts

    while block_start <= end_ts:

        block_end = min(
            block_start
            + pd.DateOffset(
                years=
                    FLOW_HISTORY_BLOCK_YEARS
            )
            - pd.Timedelta(
                days=1
            ),
            end_ts,
        )

        try:

            block = (
                query_caudal_series(
                    series_id,
                    block_start,
                    block_end,
                )
            )

            if not block.empty:

                frames.append(
                    block
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

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )

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
# CAUDAL DE UNA ESTACIÓN
# ============================================================

def get_caudal_station(
    station,
    start,
    end,
):

    candidate, metadata = (
        seleccionar_serie_caudal(
            station,
            start,
            end,
        )
    )

    col = FLOW_COLUMNS[
        station
    ]

    if candidate is None:

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    col,
                ]
            ),
            metadata,
        )

    series_id = _safe_int(
        candidate.get(
            "series_id"
        )
    )

    try:

        history = (
            _query_history_blocks(
                series_id,
                start,
                end,
            )
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
                    col,
                ]
            ),
            metadata,
        )

    if history.empty:

        metadata[
            "status"
        ] = "sin_datos"

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    col,
                ]
            ),
            metadata,
        )

    result = history.rename(
        columns={
            "value":
                col
        }
    )

    result[
        col
    ] = _numeric(
        result[
            col
        ]
    )

    result = result[
        (
            result[
                col
            ]
            >= ESTIMATED_FLOW_MIN
        )
        &
        (
            result[
                col
            ]
            <= ESTIMATED_FLOW_MAX
        )
    ]

    metadata[
        "records"
    ] = int(
        result[
            col
        ]
        .notna()
        .sum()
    )

    metadata[
        "start"
    ] = (
        result[
            "datetime"
        ].min()
        if not result.empty
        else None
    )

    metadata[
        "end"
    ] = (
        result[
            "datetime"
        ].max()
        if not result.empty
        else None
    )

    return (
        result[
            [
                "datetime",
                col,
            ]
        ],
        metadata,
    )


# ============================================================
# CAUDALES OBSERVADOS DEL CORREDOR
# ============================================================

def get_observed_caudales(
    start,
    end,
):

    result = None

    metadata = {
        "stations": {},
    }

    for station in STATIONS:

        col = FLOW_COLUMNS[
            station
        ]

        try:

            station_df, station_meta = (
                get_caudal_station(
                    station,
                    start,
                    end,
                )
            )

        except Exception as exc:

            station_df = pd.DataFrame(
                columns=[
                    "datetime",
                    col,
                ]
            )

            station_meta = {
                "station":
                    station,

                "status":
                    "error",

                "error":
                    str(exc),
            }

        metadata[
            "stations"
        ][
            station
        ] = station_meta

        if result is None:

            result = station_df.copy()

        else:

            result = result.merge(
                station_df,
                on="datetime",
                how="outer",
            )

    if result is None:

        result = pd.DataFrame(
            columns=[
                "datetime"
            ]
        )

    if "datetime" in result.columns:

        result[
            "datetime"
        ] = _datetime_naive(
            result[
                "datetime"
            ]
        )

    for col in FLOW_COLUMNS.values():

        if col not in result.columns:

            result[col] = np.nan

        result[col] = _numeric(
            result[col]
        )

    result = (
        result.sort_values(
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

    return (
        result,
        metadata,
    )


# ============================================================
# INICIALIZAR ORIGEN Y CALIDAD
#
# Esta función corrige el problema de V11.11:
# si ya existe source/quality, NO lo sobrescribe.
# ============================================================

def _initialize_flow_sources(
    df,
    default_existing_source="observado",
    default_existing_quality=1.0,
):

    result = df.copy()

    for station, col in (
        FLOW_COLUMNS.items()
    ):

        if col not in result.columns:
            result[col] = np.nan

        result[col] = _numeric(
            result[col]
        )

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        if source_col not in result.columns:

            result[
                source_col
            ] = None

        if quality_col not in result.columns:

            result[
                quality_col
            ] = np.nan

        # ----------------------------------------------------
        # Sólo completar etiquetas faltantes.
        # Nunca pisar "proyectado", "estimado...", etc.
        # ----------------------------------------------------

        has_value = result[
            col
        ].notna()

        missing_source = (
            result[
                source_col
            ]
            .isna()
            |
            (
                result[
                    source_col
                ]
                .astype(str)
                .str.strip()
                == ""
            )
        )

        missing_quality = (
            pd.to_numeric(
                result[
                    quality_col
                ],
                errors="coerce",
            )
            .isna()
        )

        result.loc[
            has_value
            &
            missing_source,
            source_col,
        ] = (
            default_existing_source
        )

        result.loc[
            has_value
            &
            missing_quality,
            quality_col,
        ] = (
            default_existing_quality
        )

        result[
            quality_col
        ] = (
            pd.to_numeric(
                result[
                    quality_col
                ],
                errors="coerce",
            )
            .clip(
                lower=0.0,
                upper=1.0,
            )
        )

    return result


# ============================================================
# INTERPOLAR HUECOS CORTOS
# ============================================================

def _interpolate_short_flow_gaps(
    df,
):

    result = df.copy()

    for station, col in (
        FLOW_COLUMNS.items()
    ):

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        original = _numeric(
            result[
                col
            ]
        )

        interpolated = (
            original.interpolate(
                method="linear",
                limit=
                    SHORT_INTERPOLATION_LIMIT,
                limit_area=
                    "inside",
            )
        )

        new_mask = (
            original.isna()
            &
            interpolated.notna()
        )

        result[
            col
        ] = interpolated

        result.loc[
            new_mask,
            source_col,
        ] = "interpolado"

        result.loc[
            new_mask,
            quality_col,
        ] = 0.90

    return result


# ============================================================
# ESTACIONES VECINAS
# ============================================================

def _neighbor_stations(
    station,
):

    index = STATIONS.index(
        station
    )

    neighbors = []

    if index > 0:
        neighbors.append(
            STATIONS[
                index - 1
            ]
        )

    if index < (
        len(STATIONS) - 1
    ):
        neighbors.append(
            STATIONS[
                index + 1
            ]
        )

    # segunda vecindad como respaldo
    if index > 1:
        neighbors.append(
            STATIONS[
                index - 2
            ]
        )

    if index < (
        len(STATIONS) - 2
    ):
        neighbors.append(
            STATIONS[
                index + 2
            ]
        )

    return neighbors


# ============================================================
# RELACIÓN HISTÓRICA ENTRE CAUDALES
#
# Usa exclusivamente datos de alta calidad para calibrar
# la relación. No deja que estimaciones previas contaminen
# el cociente.
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

    target_source_col = (
        target_col
        + "_source"
    )

    target_quality_col = (
        target_col
        + "_quality"
    )

    ref_source_col = (
        reference_col
        + "_source"
    )

    ref_quality_col = (
        reference_col
        + "_quality"
    )

    target = _numeric(
        df[
            target_col
        ]
    )

    reference = _numeric(
        df[
            reference_col
        ]
    )

    if target_quality_col in df.columns:

        tq = (
            _numeric(
                df[
                    target_quality_col
                ]
            )
            .fillna(0.0)
        )

    else:

        tq = pd.Series(
            1.0,
            index=df.index,
        )

    if ref_quality_col in df.columns:

        rq = (
            _numeric(
                df[
                    ref_quality_col
                ]
            )
            .fillna(0.0)
        )

    else:

        rq = pd.Series(
            1.0,
            index=df.index,
        )

    valid = (
        target.notna()
        &
        reference.notna()
        &
        (reference > 0)
        &
        (target > 0)
        &
        (tq >= 0.85)
        &
        (rq >= 0.85)
    )

    if (
        target_source_col
        in df.columns
    ):

        target_source = (
            df[
                target_source_col
            ]
            .astype(str)
            .str.lower()
        )

        valid &= (
            target_source.isin(
                [
                    "observado",
                    "interpolado",
                ]
            )
        )

    if (
        ref_source_col
        in df.columns
    ):

        ref_source = (
            df[
                ref_source_col
            ]
            .astype(str)
            .str.lower()
        )

        valid &= (
            ref_source.isin(
                [
                    "observado",
                    "interpolado",
                ]
            )
        )

    if int(
        valid.sum()
    ) < MIN_RATIO_OVERLAP:

        return None

    ratio = (
        target[
            valid
        ]
        /
        reference[
            valid
        ]
    )

    ratio = ratio[
        (
            ratio
            >= RATIO_MIN
        )
        &
        (
            ratio
            <= RATIO_MAX
        )
    ]

    if len(
        ratio
    ) < MIN_RATIO_OVERLAP:

        return None

    median = float(
        ratio.median()
    )

    std = float(
        ratio.std()
    )

    if not np.isfinite(
        std
    ):

        std = 0.0

    variability = (
        std / median
        if median > 0
        else 1.0
    )

    quality = float(
        np.clip(
            0.82
            - 0.30
            * variability,
            0.45,
            0.82,
        )
    )

    return {
        "ratio":
            median,

        "std":
            std,

        "records":
            int(
                len(ratio)
            ),

        "quality":
            quality,
    }


# ============================================================
# RELLENAR POR RELACIÓN HISTÓRICA CON VECINOS
# ============================================================

def _fill_from_historical_ratios(
    df,
):

    result = df.copy()

    # --------------------------------------------------------
    # Relaciones se calculan una sola vez con datos de alta
    # calidad. Así las nuevas estimaciones no contaminan
    # la calibración.
    # --------------------------------------------------------

    ratio_map = {}

    for target_station in STATIONS:

        target_col = FLOW_COLUMNS[
            target_station
        ]

        ratio_map[
            target_station
        ] = []

        for reference_station in (
            _neighbor_stations(
                target_station
            )
        ):

            reference_col = (
                FLOW_COLUMNS[
                    reference_station
                ]
            )

            relation = (
                _historical_flow_ratio(
                    result,
                    target_col,
                    reference_col,
                )
            )

            if relation is None:
                continue

            ratio_map[
                target_station
            ].append(
                {
                    "reference_station":
                        reference_station,

                    **relation,
                }
            )

    # --------------------------------------------------------
    # Tres pasadas permiten completar gradualmente,
    # pero las relaciones ya quedaron fijas.
    # --------------------------------------------------------

    for _ in range(3):

        changed = False

        for target_station in STATIONS:

            target_col = (
                FLOW_COLUMNS[
                    target_station
                ]
            )

            source_col = (
                target_col
                + "_source"
            )

            quality_col = (
                target_col
                + "_quality"
            )

            missing = result[
                target_col
            ].isna()

            if not missing.any():
                continue

            relations = ratio_map.get(
                target_station,
                []
            )

            if not relations:
                continue

            estimates = []

            weights = []

            labels = []

            for relation in relations:

                reference_station = (
                    relation[
                        "reference_station"
                    ]
                )

                reference_col = (
                    FLOW_COLUMNS[
                        reference_station
                    ]
                )

                reference_quality_col = (
                    reference_col
                    + "_quality"
                )

                reference = _numeric(
                    result[
                        reference_col
                    ]
                )

                if (
                    reference_quality_col
                    in result.columns
                ):

                    reference_quality = (
                        _numeric(
                            result[
                                reference_quality_col
                            ]
                        )
                        .fillna(0.0)
                    )

                else:

                    reference_quality = (
                        pd.Series(
                            1.0,
                            index=result.index,
                        )
                    )

                estimated = (
                    reference
                    * relation[
                        "ratio"
                    ]
                )

                estimated = estimated.where(
                    (
                        estimated
                        >= ESTIMATED_FLOW_MIN
                    )
                    &
                    (
                        estimated
                        <= ESTIMATED_FLOW_MAX
                    )
                )

                donor_weight = (
                    reference_quality
                    *
                    relation[
                        "quality"
                    ]
                )

                estimates.append(
                    estimated
                )

                weights.append(
                    donor_weight
                )

                labels.append(
                    reference_station
                )

            if not estimates:
                continue

            estimate_matrix = pd.concat(
                estimates,
                axis=1,
            )

            weight_matrix = pd.concat(
                weights,
                axis=1,
            )

            weighted_sum = (
                estimate_matrix
                * weight_matrix
            ).sum(
                axis=1,
                min_count=1,
            )

            total_weight = (
                weight_matrix.where(
                    estimate_matrix.notna()
                )
                .sum(
                    axis=1,
                    min_count=1,
                )
            )

            combined = (
                weighted_sum
                /
                total_weight.replace(
                    0,
                    np.nan,
                )
            )

            fill_mask = (
                missing
                &
                combined.notna()
            )

            if not fill_mask.any():
                continue

            result.loc[
                fill_mask,
                target_col,
            ] = combined[
                fill_mask
            ]

            # ------------------------------------------------
            # Calidad combinada
            # ------------------------------------------------

            combined_quality = (
                total_weight
                /
                max(
                    len(estimates),
                    1,
                )
            ).clip(
                lower=0.45,
                upper=0.82,
            )

            result.loc[
                fill_mask,
                quality_col,
            ] = combined_quality[
                fill_mask
            ]

            # ------------------------------------------------
            # Trazabilidad
            # ------------------------------------------------

            if len(labels) == 1:

                source_label = (
                    "estimado_ratio_"
                    + _normalize_text(
                        labels[0]
                    )
                    .replace(
                        " ",
                        "_"
                    )
                )

            else:

                source_label = (
                    "estimado_ratios_vecinos"
                )

            result.loc[
                fill_mask,
                source_col,
            ] = source_label

            changed = True

        if not changed:
            break

    return result


# ============================================================
# RELLENO POR CORREDOR
#
# Sólo utiliza donantes de calidad >= 0.70.
# Evita construir una estimación a partir de varias
# estimaciones débiles.
# ============================================================

def _fill_from_corridor_median(
    df,
):

    result = df.copy()

    for target_station in STATIONS:

        target_col = (
            FLOW_COLUMNS[
                target_station
            ]
        )

        source_col = (
            target_col
            + "_source"
        )

        quality_col = (
            target_col
            + "_quality"
        )

        missing_indices = result.index[
            result[
                target_col
            ].isna()
        ]

        for idx in missing_indices:

            donor_values = []

            donor_qualities = []

            for donor_station in STATIONS:

                if donor_station == target_station:
                    continue

                donor_col = (
                    FLOW_COLUMNS[
                        donor_station
                    ]
                )

                donor_quality_col = (
                    donor_col
                    + "_quality"
                )

                donor_value = _safe_float(
                    result.at[
                        idx,
                        donor_col,
                    ]
                )

                donor_quality = _safe_float(
                    result.at[
                        idx,
                        donor_quality_col,
                    ],
                    0.0,
                )

                if (
                    not np.isfinite(
                        donor_value
                    )
                    or donor_quality < 0.70
                ):
                    continue

                donor_values.append(
                    donor_value
                )

                donor_qualities.append(
                    donor_quality
                )

            if len(
                donor_values
            ) < 2:

                continue

            estimate = float(
                np.average(
                    donor_values,
                    weights=
                        donor_qualities,
                )
            )

            if (
                estimate
                < ESTIMATED_FLOW_MIN
                or estimate
                > ESTIMATED_FLOW_MAX
            ):

                continue

            result.at[
                idx,
                target_col,
            ] = estimate

            result.at[
                idx,
                source_col,
            ] = "estimado_corredor"

            result.at[
                idx,
                quality_col,
            ] = 0.40

    return result


# ============================================================
# COMPLETAR CAUDALES FALTANTES
# ============================================================

def complete_missing_flows(
    df,
    default_existing_source="observado",
    default_existing_quality=1.0,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
    ):

        return pd.DataFrame()

    result = df.copy()

    result = (
        _initialize_flow_sources(
            result,
            default_existing_source=
                default_existing_source,
            default_existing_quality=
                default_existing_quality,
        )
    )

    # --------------------------------------------------------
    # 1. interpolación corta
    # --------------------------------------------------------

    result = (
        _interpolate_short_flow_gaps(
            result
        )
    )

    # --------------------------------------------------------
    # 2. relación histórica con vecinos
    # --------------------------------------------------------

    result = (
        _fill_from_historical_ratios(
            result
        )
    )

    # --------------------------------------------------------
    # 3. último respaldo de corredor
    # --------------------------------------------------------

    result = (
        _fill_from_corridor_median(
            result
        )
    )

    # --------------------------------------------------------
    # Validación final
    # --------------------------------------------------------

    for station, col in (
        FLOW_COLUMNS.items()
    ):

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        invalid = (
            result[
                col
            ].notna()
            &
            (
                (
                    result[
                        col
                    ]
                    < ESTIMATED_FLOW_MIN
                )
                |
                (
                    result[
                        col
                    ]
                    > ESTIMATED_FLOW_MAX
                )
            )
        )

        result.loc[
            invalid,
            col,
        ] = np.nan

        result.loc[
            invalid,
            source_col,
        ] = None

        result.loc[
            invalid,
            quality_col,
        ] = np.nan

        result[
            quality_col
        ] = (
            _numeric(
                result[
                    quality_col
                ]
            )
            .clip(
                lower=0.0,
                upper=1.0,
            )
        )

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
    # Primero exigir calidad >= 0.60.
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

        q = _numeric(
            df[col]
        )

        if quality_col in df.columns:

            quality = (
                _numeric(
                    df[
                        quality_col
                    ]
                )
                .fillna(0.0)
            )

        else:

            quality = pd.Series(
                1.0,
                index=df.index,
            )

        valid = (
            q.notna()
            &
            (
                quality
                >= 0.60
            )
        )

        if int(
            valid.sum()
        ) >= 3:

            return (
                station,
                col,
            )

    # --------------------------------------------------------
    # Segundo respaldo: cualquier serie utilizable.
    # --------------------------------------------------------

    for station in FLOW_PRIORITY:

        col = FLOW_COLUMNS[
            station
        ]

        if (
            col in df.columns
            and _numeric(
                df[col]
            )
            .notna()
            .sum()
            >= 3
        ):

            return (
                station,
                col,
            )

    return (
        None,
        None,
    )


# ============================================================
# PROYECCIÓN DE UNA SERIE DE CAUDAL
# ============================================================

def proyectar_serie_caudal(
    history,
    col,
    days,
):

    days = int(
        np.clip(
            days,
            1,
            MAX_FORECAST_DAYS,
        )
    )

    q = (
        _numeric(
            history[
                col
            ]
        )
        .dropna()
    )

    if len(q) < 3:

        return np.full(
            days,
            np.nan,
        )

    current = float(
        q.iloc[-1]
    )

    recent = q.tail(
        min(
            21,
            len(q),
        )
    )

    if len(
        recent
    ) >= 4:

        try:

            slope = np.polyfit(
                np.arange(
                    len(recent)
                ),
                recent.to_numpy(
                    dtype=float
                ),
                1,
            )[0]

        except Exception:

            slope = 0.0

    else:

        slope = 0.0

    max_daily_change = max(
        abs(
            current
        )
        * 0.025,
        50.0,
    )

    slope = float(
        np.clip(
            slope,
            -max_daily_change,
            max_daily_change,
        )
    )

    values = []

    value = current

    for day in range(
        1,
        days + 1,
    ):

        damping = np.exp(
            -day / 20.0
        )

        daily_change = (
            slope
            * damping
        )

        value = value + daily_change

        value = float(
            np.clip(
                value,
                ESTIMATED_FLOW_MIN,
                ESTIMATED_FLOW_MAX,
            )
        )

        values.append(
            value
        )

    return np.asarray(
        values,
        dtype=float,
    )


# ============================================================
# FEATURES DE LLUVIA
# ============================================================

def _add_rain_features(
    df,
):

    result = df.copy()

    for station, col in (
        RAIN_COLUMNS.items()
    ):

        if col not in result.columns:
            result[col] = 0.0

        rain = (
            _numeric(
                result[
                    col
                ]
            )
            .fillna(0.0)
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
                rain.rolling(
                    window,
                    min_periods=1,
                )
                .sum()
            )

    return result


# ============================================================
# FEATURES DE CAUDAL
# ============================================================

def _add_flow_features(
    df,
):

    result = df.copy()

    for station, col in (
        FLOW_COLUMNS.items()
    ):

        if col not in result.columns:
            result[col] = np.nan

        q = _numeric(
            result[
                col
            ]
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
                q.rolling(
                    window,
                    min_periods=1,
                )
                .mean()
            )

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
# PRESIÓN DE LLUVIA DEL CORREDOR
# ============================================================

def _rain_pressure(
    df,
):

    signals = []

    weights = []

    for station, col in (
        RAIN_COLUMNS.items()
    ):

        if col not in df.columns:
            continue

        rain = (
            _numeric(
                df[col]
            )
            .fillna(0.0)
        )

        accumulated = (
            rain.rolling(
                7,
                min_periods=1,
            )
            .sum()
        )

        signals.append(
            (
                accumulated
                / 100.0
            )
            .clip(
                lower=0.0,
                upper=3.0,
            )
        )

        weights.append(
            CORRIDOR_WEIGHTS[
                station
            ]
        )

    if not signals:

        return pd.Series(
            0.0,
            index=df.index,
        )

    matrix = pd.concat(
        signals,
        axis=1,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    return pd.Series(
        np.average(
            matrix.fillna(
                0.0
            ).to_numpy(),
            axis=1,
            weights=weights,
        ),
        index=df.index,
    )


# ============================================================
# PRESIÓN DE CAUDAL
# ============================================================

def _flow_pressure(
    df,
):

    signals = []

    weights = []

    for station, col in (
        FLOW_COLUMNS.items()
    ):

        if col not in df.columns:
            continue

        q = _numeric(
            df[col]
        )

        quality_col = (
            col
            + "_quality"
        )

        if quality_col in df.columns:

            quality = (
                _numeric(
                    df[
                        quality_col
                    ]
                )
                .fillna(0.0)
            )

        else:

            quality = pd.Series(
                1.0,
                index=df.index,
            )

        baseline = (
            q.rolling(
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

        relative = relative.clip(
            lower=-1.0,
            upper=2.0,
        )

        signals.append(
            relative
            * quality
        )

        weights.append(
            CORRIDOR_WEIGHTS[
                station
            ]
        )

    if not signals:

        return pd.Series(
            0.0,
            index=df.index,
        )

    matrix = pd.concat(
        signals,
        axis=1,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    return pd.Series(
        np.average(
            matrix.fillna(
                0.0
            ).to_numpy(),
            axis=1,
            weights=weights,
        ),
        index=df.index,
    )


# ============================================================
# FEATURES EXÓGENAS GLOBALES
# ============================================================

def add_exogenous_features(
    df,
):

    result = df.copy()

    result = (
        _add_rain_features(
            result
        )
    )

    result = (
        _add_flow_features(
            result
        )
    )

    result[
        "rain_pressure_7d"
    ] = _rain_pressure(
        result
    )

    result[
        "flow_pressure"
    ] = _flow_pressure(
        result
    )

    result[
        "hydro_exogenous_pressure"
    ] = (
        0.55
        * result[
            "flow_pressure"
        ]
        +
        0.45
        * result[
            "rain_pressure_7d"
        ]
    )

    return result


# ============================================================
# RESUMEN DE CALIDAD
# ============================================================

def _quality_summary(
    df,
):

    rows = []

    for station, col in (
        FLOW_COLUMNS.items()
    ):

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        valid = _numeric(
            df[
                col
            ]
        ).notna()

        records = int(
            valid.sum()
        )

        observed = 0
        interpolated = 0
        estimated = 0
        projected = 0

        if source_col in df.columns:

            source = (
                df[
                    source_col
                ]
                .fillna("")
                .astype(str)
                .str.lower()
            )

            observed = int(
                (
                    valid
                    &
                    (
                        source
                        == "observado"
                    )
                )
                .sum()
            )

            interpolated = int(
                (
                    valid
                    &
                    (
                        source
                        == "interpolado"
                    )
                )
                .sum()
            )

            estimated = int(
                (
                    valid
                    &
                    source.str.startswith(
                        "estimado"
                    )
                )
                .sum()
            )

            projected = int(
                (
                    valid
                    &
                    (
                        source
                        == "proyectado"
                    )
                )
                .sum()
            )

        if quality_col in df.columns:

            quality = (
                _numeric(
                    df[
                        quality_col
                    ]
                )
                .dropna()
            )

            mean_quality = (
                float(
                    quality.mean()
                )
                if not quality.empty
                else np.nan
            )

        else:

            mean_quality = np.nan

        rows.append(
            {
                "station":
                    station,

                "records":
                    records,

                "observed":
                    observed,

                "interpolated":
                    interpolated,

                "estimated":
                    estimated,

                "projected":
                    projected,

                "mean_quality":
                    mean_quality,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# API PRINCIPAL
# ============================================================

def get_exogenous_data(
    start,
    end,
    forecast_days=15,
):

    forecast_days = int(
        np.clip(
            forecast_days,
            1,
            MAX_FORECAST_DAYS,
        )
    )

    start_ts = pd.Timestamp(
        start
    ).normalize()

    end_ts = pd.Timestamp(
        end
    ).normalize()

    # ========================================================
    # 1. LLUVIA HISTÓRICA
    # ========================================================

    rain_history, rain_meta = (
        get_rain_history(
            start_ts,
            end_ts,
        )
    )

    # ========================================================
    # 2. CAUDALES OBSERVADOS
    # ========================================================

    observed_flows, flow_meta = (
        get_observed_caudales(
            start_ts,
            end_ts,
        )
    )

    # ========================================================
    # 3. EJE DIARIO COMPLETO
    # ========================================================

    daily = pd.DataFrame(
        {
            "datetime":
                pd.date_range(
                    start=start_ts,
                    end=end_ts,
                    freq="D",
                )
        }
    )

    history = daily.merge(
        rain_history,
        on="datetime",
        how="left",
    )

    history = history.merge(
        observed_flows,
        on="datetime",
        how="left",
    )

    # ========================================================
    # 4. GARANTIZAR COLUMNAS
    # ========================================================

    for col in RAIN_COLUMNS.values():

        if col not in history.columns:
            history[col] = 0.0

        history[col] = (
            _numeric(
                history[col]
            )
            .fillna(0.0)
            .clip(
                lower=0.0
            )
        )

    for col in FLOW_COLUMNS.values():

        if col not in history.columns:
            history[col] = np.nan

        history[col] = _numeric(
            history[col]
        )

    # ========================================================
    # 5. COMPLETAR CAUDALES HISTÓRICOS
    # ========================================================

    history = complete_missing_flows(
        history,
        default_existing_source=
            "observado",
        default_existing_quality=
            1.0,
    )

    # ========================================================
    # 6. CAUDAL PRINCIPAL
    # ========================================================

    main_flow_station, main_flow_col = (
        elegir_caudal_principal(
            history
        )
    )

    if main_flow_col is not None:

        history[
            "caudal_m3s"
        ] = history[
            main_flow_col
        ]

        history[
            "caudal_source"
        ] = history[
            main_flow_col
            + "_source"
        ]

        history[
            "caudal_quality"
        ] = history[
            main_flow_col
            + "_quality"
        ]

    else:

        history[
            "caudal_m3s"
        ] = np.nan

        history[
            "caudal_source"
        ] = None

        history[
            "caudal_quality"
        ] = np.nan

    # ========================================================
    # 7. FEATURES HISTÓRICAS
    # ========================================================

    history = add_exogenous_features(
        history
    )

    # ========================================================
    # 8. LLUVIA FUTURA
    # ========================================================

    rain_future, rain_forecast_meta = (
        get_rain_forecast(
            forecast_days
        )
    )

    # --------------------------------------------------------
    # Queremos que el futuro comience al día siguiente de
    # la fecha base del conjunto histórico.
    # --------------------------------------------------------

    future_dates = pd.DataFrame(
        {
            "datetime":
                pd.date_range(
                    start=
                        end_ts
                        + pd.Timedelta(
                            days=1
                        ),

                    periods=
                        forecast_days,

                    freq="D",
                )
        }
    )

    rain_future = rain_future.copy()

    if not rain_future.empty:

        rain_future = (
            rain_future
            .sort_values(
                "datetime"
            )
            .reset_index(
                drop=True
            )
        )

        # Vincular por orden para evitar diferencias entre
        # "hoy" de Open-Meteo y la fecha base elegida.
        rain_values = (
            rain_future[
                list(
                    RAIN_COLUMNS.values()
                )
            ]
            .head(
                forecast_days
            )
            .reset_index(
                drop=True
            )
        )

        future = future_dates.copy()

        for col in RAIN_COLUMNS.values():

            future[col] = 0.0

            n = min(
                len(
                    rain_values
                ),
                forecast_days,
            )

            if (
                col
                in rain_values.columns
                and n > 0
            ):

                future.loc[
                    :n - 1,
                    col,
                ] = (
                    _numeric(
                        rain_values[
                            col
                        ]
                    )
                    .fillna(0.0)
                    .to_numpy()[
                        :n
                    ]
                )

    else:

        future = (
            future_dates.copy()
        )

        for col in RAIN_COLUMNS.values():
            future[col] = 0.0

    future[
        "precip_mm"
    ] = future[
        list(
            RAIN_COLUMNS.values()
        )
    ].mean(
        axis=1
    )

    # ========================================================
    # 9. PROYECTAR CAUDALES
    # ========================================================

    for station, col in (
        FLOW_COLUMNS.items()
    ):

        source_col = (
            col
            + "_source"
        )

        quality_col = (
            col
            + "_quality"
        )

        q_history = (
            _numeric(
                history[
                    col
                ]
            )
            .dropna()
        )

        if len(
            q_history
        ) >= 3:

            projected = (
                proyectar_serie_caudal(
                    history,
                    col,
                    forecast_days,
                )
            )

            future[col] = projected

            # ------------------------------------------------
            # CORRECCIÓN CLAVE V11.11.1
            #
            # Un valor futuro es PROYECTADO.
            # Nunca observado.
            # ------------------------------------------------

            future[
                source_col
            ] = "proyectado"

            future[
                quality_col
            ] = (
                0.65
                * np.exp(
                    -np.arange(
                        forecast_days
                    )
                    / 90.0
                )
            ).clip(
                0.35,
                0.65,
            )

        else:

            future[col] = np.nan

            future[
                source_col
            ] = None

            future[
                quality_col
            ] = np.nan

    # ========================================================
    # 10. COMPLETAR CAUDALES FUTUROS FALTANTES
    #
    # Importantísimo:
    # se preservan "proyectado" + calidad correspondiente.
    # ========================================================

    future = complete_missing_flows(
        future,
        default_existing_source=
            "proyectado",
        default_existing_quality=
            0.60,
    )

    # ========================================================
    # 11. FORZAR TRAZABILIDAD FUTURA
    #
    # Cualquier dato ya presente antes del relleno sigue
    # siendo como máximo proyectado. No permitir observado.
    # ========================================================

    for station, col in (
        FLOW_COLUMNS.items()
    ):

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
            ] = None

        if quality_col not in future.columns:
            future[
                quality_col
            ] = np.nan

        valid = future[
            col
        ].notna()

        observed_mask = (
            future[
                source_col
            ]
            .astype(str)
            .str.lower()
            == "observado"
        )

        future.loc[
            valid
            &
            observed_mask,
            source_col,
        ] = "proyectado"

        future.loc[
            valid,
            quality_col,
        ] = (
            _numeric(
                future.loc[
                    valid,
                    quality_col,
                ]
            )
            .clip(
                upper=0.65
            )
        )

    # ========================================================
    # 12. CAUDAL PRINCIPAL FUTURO
    # ========================================================

    if main_flow_col is not None:

        future[
            "caudal_m3s"
        ] = future[
            main_flow_col
        ]

        future[
            "caudal_source"
        ] = future[
            main_flow_col
            + "_source"
        ]

        future[
            "caudal_quality"
        ] = future[
            main_flow_col
            + "_quality"
        ]

    else:

        future[
            "caudal_m3s"
        ] = np.nan

        future[
            "caudal_source"
        ] = None

        future[
            "caudal_quality"
        ] = np.nan

    # ========================================================
    # 13. FEATURES FUTURAS
    # ========================================================

    combined_for_features = pd.concat(
        [
            history,
            future,
        ],
        ignore_index=True,
        sort=False,
    )

    combined_for_features = (
        add_exogenous_features(
            combined_for_features
        )
    )

    future = (
        combined_for_features[
            combined_for_features[
                "datetime"
            ]
            > end_ts
        ]
        .copy()
        .head(
            forecast_days
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # 14. METADATA
    # ========================================================

    observed_stations = []

    available_stations = []

    reconstructed_stations = []

    for station, col in (
        FLOW_COLUMNS.items()
    ):

        source_col = (
            col
            + "_source"
        )

        if (
            col in history.columns
            and _numeric(
                history[col]
            )
            .notna()
            .any()
        ):

            available_stations.append(
                station
            )

        if (
            source_col
            in history.columns
        ):

            source = (
                history[
                    source_col
                ]
                .fillna("")
                .astype(str)
                .str.lower()
            )

            if (
                source
                == "observado"
            ).any():

                observed_stations.append(
                    station
                )

            if (
                source.str.startswith(
                    "estimado"
                )
                |
                (
                    source
                    == "interpolado"
                )
            ).any():

                reconstructed_stations.append(
                    station
                )

    quality_history = (
        _quality_summary(
            history
        )
    )

    quality_future = (
        _quality_summary(
            future
        )
    )

    metadata = {

        "version":
            VERSION,

        "rain":
            rain_meta,

        "rain_forecast":
            rain_forecast_meta,

        "flow":
            flow_meta,

        "main_flow_station":
            main_flow_station,

        "main_flow_column":
            main_flow_col,

        "flow_observed_stations":
            observed_stations,

        "flow_available_stations":
            available_stations,

        "flow_reconstructed_stations":
            reconstructed_stations,

        "flow_quality_history":
            quality_history,

        "flow_quality_future":
            quality_future,

        "history_records":
            int(
                len(history)
            ),

        "future_records":
            int(
                len(future)
            ),

        "warning":
            (
                "Los caudales reconstruidos y proyectados "
                "se mantienen diferenciados de las observaciones "
                "reales mediante source y quality."
            ),
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

        "status":
            "pendiente",
    }

    try:

        catalog = get_ina_catalog()

        result[
            "catalog_records"
        ] = int(
            len(catalog)
        )

        if not catalog.empty:

            caudal_catalog = catalog[
                catalog[
                    "var_id"
                ]
                == VAR_ID_CAUDAL
            ].copy()

            result[
                "caudal_catalog_records"
            ] = int(
                len(
                    caudal_catalog
                )
            )

        station_results = {}

        for station in STATIONS:

            candidates = (
                candidatos_caudal_estacion(
                    station,
                    start,
                    end,
                )
            )

            station_results[
                station
            ] = {
                "candidate_count":
                    int(
                        len(
                            candidates
                        )
                    ),

                "top_candidates":
                    (
                        candidates[
                            [
                                col
                                for col in [
                                    "series_id",
                                    "nombre",
                                    "rio",
                                    "count",
                                    "timeend",
                                    "score",
                                ]
                                if col
                                in candidates.columns
                            ]
                        ]
                        .head(5)
                        .to_dict(
                            orient="records"
                        )
                        if not candidates.empty
                        else []
                    ),
            }

        result[
            "candidate_summary"
        ] = station_results

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
            "history_records"
        ] = int(
            len(history)
        )

        result[
            "future_records"
        ] = int(
            len(future)
        )

        result[
            "main_flow_station"
        ] = metadata.get(
            "main_flow_station"
        )

        result[
            "observed_stations"
        ] = metadata.get(
            "flow_observed_stations",
            [],
        )

        result[
            "available_stations"
        ] = metadata.get(
            "flow_available_stations",
            [],
        )

        result[
            "reconstructed_stations"
        ] = metadata.get(
            "flow_reconstructed_stations",
            [],
        )

        result[
            "quality_history"
        ] = (
            metadata.get(
                "flow_quality_history",
                pd.DataFrame(),
            )
            .to_dict(
                orient="records"
            )
            if isinstance(
                metadata.get(
                    "flow_quality_history"
                ),
                pd.DataFrame,
            )
            else []
        )

        result[
            "quality_future"
        ] = (
            metadata.get(
                "flow_quality_future",
                pd.DataFrame(),
            )
            .to_dict(
                orient="records"
            )
            if isinstance(
                metadata.get(
                    "flow_quality_future"
                ),
                pd.DataFrame,
            )
            else []
        )

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
