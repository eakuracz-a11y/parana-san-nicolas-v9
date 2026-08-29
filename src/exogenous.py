# ============================================================
# PARANÁ · SAN NICOLÁS
# src/exogenous.py
# V11.10.1 COMPLETO
#
# VARIABLES EXÓGENAS MULTIESTACIÓN
#
# ------------------------------------------------------------
# PRECIPITACIÓN
# - Corrientes
# - Goya
# - La Paz
# - Paraná
# - Diamante
# - Rosario
# - Villa Constitución
# - San Nicolás
#
# CAUDAL INA A5
# - búsqueda automática de series var_id = 4
# - validación mediante observaciones reales
# - una serie por estación cuando existe
# - NO inventa caudales faltantes
#
# COMPATIBILIDAD
# ------------------------------------------------------------
# Mantiene:
#   precip_mm
#   caudal_m3s
#
# Agrega:
#   rain_corrientes
#   rain_goya
#   rain_la_paz
#   rain_parana
#   rain_diamante
#   rain_rosario
#   rain_villa_constitucion
#   rain_san_nicolas
#
#   q_corrientes
#   q_goya
#   q_la_paz
#   q_parana
#   q_diamante
#   q_rosario
#   q_villa_constitucion
#   q_san_nicolas
#
# Además:
#   acumulados lluvia 3 / 7 / 15 / 30 días
#   cambios caudal 1 / 3 / 7 días
#   medias caudal 3 / 7 / 14 días
#
# API PRINCIPAL:
#
# get_exogenous_data(
#     start,
#     end,
#     forecast_days=60
# )
#
# retorna:
#     history, future, metadata
# ============================================================


from functools import lru_cache
from datetime import datetime, timedelta
import re
import unicodedata

import numpy as np
import pandas as pd
import requests


# ============================================================
# VERSIÓN
# ============================================================

VERSION = "V11.10.1"


# ============================================================
# CONFIGURACIÓN
# ============================================================

REQUEST_TIMEOUT = 45

MAX_FORECAST_DAYS = 60

OPEN_METEO_REAL_FORECAST_DAYS = 16

MIN_FLOW_OBSERVATIONS = 3

FLOW_VALIDATION_DAYS = 180


# ============================================================
# OPEN-METEO
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

# var_id = 4 -> Caudal
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
# COORDENADAS DE PRECIPITACIÓN
#
# Se utilizan como puntos representativos de cada estación
# hidrométrica/corredor.
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
# NOMBRES DE COLUMNAS
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
# PRIORIDAD PARA CAUDAL PRINCIPAL
#
# Se intenta utilizar la serie más próxima a San Nicolás.
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
# ALIAS PARA IDENTIFICAR ESTACIONES INA
# ============================================================

STATION_ALIASES = {

    "Corrientes": [
        "corrientes",
        "puerto corrientes",
        "corriente",
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
        if unicodedata.category(ch) != "Mn"
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


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


def _safe_date(
    value,
):

    dt = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(dt):
        return pd.NaT

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


# ============================================================
# RANGO DIARIO
# ============================================================

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
# PRECIPITACIÓN HISTÓRICA OPEN-METEO
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


    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    times,
                    errors="coerce",
                ),

            RAIN_COLUMNS[
                station
            ]:
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
        RAIN_COLUMNS[
            station
        ]
    ] = (
        pd.to_numeric(
            result[
                RAIN_COLUMNS[
                    station
                ]
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


    result = result.dropna(
        subset=[
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
# PRECIPITACIÓN HISTÓRICA MULTIESTACIÓN
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

                base[col] = np.nan

                metadata[
                    "stations"
                ][station] = {
                    "status":
                        "sin_datos",

                    "records":
                        0,
                }


            else:

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

                "error":
                    str(
                        exc
                    ),

                "records":
                    0,
            }


    # --------------------------------------------------------
    # Promedio del corredor
    # --------------------------------------------------------

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
# PRONÓSTICO LLUVIA POR ESTACIÓN
# ============================================================

def _get_rain_forecast_station(
    station,
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


    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    times,
                    errors="coerce",
                ),

            RAIN_COLUMNS[
                station
            ]:
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
        RAIN_COLUMNS[
            station
        ]
    ] = (
        pd.to_numeric(
            result[
                RAIN_COLUMNS[
                    station
                ]
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
# PRONÓSTICO DE LLUVIA MULTIESTACIÓN
# ============================================================

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
                "Precipitación desconocida. "
                "Los 0 posteriores al pronóstico real "
                "son marcadores técnicos y no deben "
                "interpretarse como ausencia de lluvia."
            ),
    }


    for station in STATIONS:

        col = RAIN_COLUMNS[
            station
        ]


        try:

            station_forecast = (
                _get_rain_forecast_station(
                    station
                )
            )


            if station_forecast.empty:

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


            else:

                base = base.merge(
                    station_forecast,
                    on="datetime",
                    how="left",
                )


                # ------------------------------------------------
                # Los días posteriores al horizonte meteorológico
                # se dejan como 0 técnico por compatibilidad.
                #
                # model.py V11.10 debe considerar que después del
                # horizonte meteorológico no es lluvia prevista.
                # ------------------------------------------------

                base[
                    col
                ] = (
                    pd.to_numeric(
                        base[
                            col
                        ],
                        errors="coerce",
                    )
                )


                real_count = int(
                    base[
                        col
                    ]
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
                ] = base[
                    col
                ].fillna(
                    0.0
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
# CATÁLOGO INA A5
# ============================================================

@lru_cache(maxsize=1)
def get_ina_catalog():

    params = {
        "format":
            "geojson"
    }


    response = requests.get(

        INA_SERIES_GEOJSON_URL,

        params=params,

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
                "id":
                    props.get(
                        "id"
                    ),

                "series_id":
                    _safe_int(
                        props.get(
                            "series_id"
                        ),
                        default=0,
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
                        default=0,
                    ),

                "proc_id":
                    _safe_int(
                        props.get(
                            "proc_id"
                        ),
                        default=0,
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
                        default=0,
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
                120,
            )


        elif name_norm.startswith(
            alias_norm
        ):

            score = max(
                score,
                100,
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
                85,
            )


        elif alias_norm in name_norm:

            score = max(
                score,
                65,
            )


    # --------------------------------------------------------
    # Penalizaciones para estaciones meteorológicas,
    # aeropuertos, INTA, escuelas, etc.
    # --------------------------------------------------------

    bad_terms = [

        "meteorologica",
        "meteorologico",
        "aeropuerto",
        "inta",
        "escuela",
        "agrometeorologica",
        "pluviometrica",
        "lluvia",
        "precipitacion",
    ]


    for bad in bad_terms:

        if bad in name_norm:

            score -= 80


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

        score += 45


    tributaries = [

        "paraguay",
        "uruguay",
        "salado",
        "carcarana",
        "carcaraña",
        "bermejo",
        "pilcomayo",
        "iguazu",
        "iguazú",
    ]


    for name in tributaries:

        if _normalize_text(
            name
        ) in river_norm:

            score -= 45


    return score


# ============================================================
# CANDIDATOS DE CAUDAL POR ESTACIÓN
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
        lambda x:
            _station_match_score(
                station,
                x,
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
    # No utilizar cualquier serie del río Paraná.
    # Debe existir coincidencia de estación.
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
            lambda x:
                8
                if x > 0
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
        * 5.0
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
            20,
            np.where(
                age_days <= 365,
                12,
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
            15,
            -40,
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
# BUSCAR REGISTROS DE OBSERVACIONES
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

                # Posible observación
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


# ============================================================
# OBTENER FECHA / VALOR DE UN REGISTRO
# ============================================================

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

        if key in record:

            dt = pd.to_datetime(
                record.get(
                    key
                ),
                errors="coerce",
                utc=True,
            )


            if not pd.isna(
                dt
            ):

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

        if key in record:

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
# CONSULTA A5
# ============================================================

def query_caudal_series(
    series_id,
    start,
    end,
):

    series_id = int(
        series_id
    )


    params = {

        "tipo":
            "puntual",

        "series_id":
            series_id,

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


        # Caudal negativo no es válido
        if value < 0:
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


    result = result[
        result[
            "value"
        ]
        >= 0
    ]


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
# VENTANA DE VALIDACIÓN DE CANDIDATO
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


    # --------------------------------------------------------
    # Validamos máximo 20 candidatos.
    # --------------------------------------------------------

    for _, candidate in (
        candidates
        .head(
            20
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


            valid = (
                pd.to_numeric(
                    test_data[
                        "value"
                    ],
                    errors="coerce",
                )
                .dropna()
            )


            valid = valid[
                valid
                >= 0
            ]


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

                "records":
                    int(
                        len(
                            valid
                        )
                    ),
            }


            metadata[
                "tested"
            ].append(
                test_info
            )


            if (
                len(valid)
                >= MIN_FLOW_OBSERVATIONS
                and valid.max() > 0
            ):

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

                        "validation_records":
                            int(
                                len(
                                    valid
                                )
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


    return (
        None,
        metadata,
    )


# ============================================================
# HISTORIA COMPLETA DE UNA SERIE SELECCIONADA
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
    # Primero intentamos una consulta completa.
    # Si falla por volumen, dividimos en bloques.
    # --------------------------------------------------------

    try:

        return query_caudal_series(
            series_id,
            request_start,
            request_end,
        )


    except Exception:

        frames = []


        block_start = request_start


        while block_start <= request_end:

            block_end = min(

                block_start
                + pd.DateOffset(
                    years=5
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
# CAUDAL DE UNA ESTACIÓN
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


        # --------------------------------------------------------
        # Una serie puntual puede tener varias observaciones/día.
        # Para el modelo utilizamos media diaria.
        # --------------------------------------------------------

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


                count = int(
                    pd.to_numeric(
                        base[
                            column
                        ],
                        errors="coerce",
                    )
                    .notna()
                    .sum()
                )


                if count >= MIN_FLOW_OBSERVATIONS:

                    metadata[
                        "available_stations"
                    ].append(
                        station
                    )


            else:

                if column not in base.columns:

                    base[
                        column
                    ] = np.nan


        except Exception as exc:

            if column not in base.columns:

                base[
                    column
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


            metadata[
                "flow_series"
            ][station] = (
                metadata[
                    "stations"
                ][station]
            )


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
            len(valid)
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
                        len(recent),
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


    # --------------------------------------------------------
    # Limitar pendiente exagerada.
    # --------------------------------------------------------

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

        # La tendencia pierde fuerza con el horizonte.
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
                result[
                    col
                ],
                errors="coerce",
            )
        )


        # En historia, NaN no se convierte indiscriminadamente
        # en lluvia cero si la fuente faltó.
        valid_exists = (
            values.notna().any()
        )


        if not valid_exists:
            continue


        values = values.fillna(
            0.0
        ).clip(
            lower=0.0
        )


        result[
            f"{col}_3d"
        ] = (
            values
            .rolling(
                3,
                min_periods=1,
            )
            .sum()
        )


        result[
            f"{col}_7d"
        ] = (
            values
            .rolling(
                7,
                min_periods=1,
            )
            .sum()
        )


        result[
            f"{col}_15d"
        ] = (
            values
            .rolling(
                15,
                min_periods=1,
            )
            .sum()
        )


        result[
            f"{col}_30d"
        ] = (
            values
            .rolling(
                30,
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
# INFORMACIÓN DE COBERTURA
# ============================================================

def _column_coverage(
    df,
    columns,
):

    result = {}


    if (
        df is None
        or df.empty
    ):

        for col in columns:

            result[
                col
            ] = 0

        return result


    for col in columns:

        if col in df.columns:

            result[
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

            result[
                col
            ] = 0


    return result


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def get_exogenous_data(
    start,
    end,
    forecast_days=15,
):

    """
    Retorna:
        history
        future
        metadata

    history:
        datetime
        precip_mm
        caudal_m3s
        rain_*
        q_*
        features derivadas

    future:
        datetime
        precip_mm
        caudal_m3s
        rain_*
        q_*
        features derivadas/proyectadas
    """

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
    # CAUDALES HISTÓRICOS
    # ========================================================

    flow_history, flow_meta = (
        get_all_caudales(
            start,
            end,
        )
    )


    # ========================================================
    # HISTORIA BASE
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

        # Evitar columnas duplicadas de datetime.
        flow_cols = [
            c
            for c in flow_history.columns
            if c != "datetime"
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
    # GARANTIZAR COLUMNAS MULTIESTACIÓN
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
    # INTERPOLACIÓN LIMITADA DE CAUDAL
    #
    # Sólo se rellenan huecos pequeños dentro de una serie
    # existente. No se inventan estaciones sin caudal.
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
    # CAUDAL LEGACY PRINCIPAL
    # ========================================================

    main_flow_col, main_flow_station = (
        elegir_caudal_principal(
            history
        )
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
    # PRECIP LEGACY
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
    # FEATURES HISTÓRICAS
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
    # LLUVIA FUTURA
    # ========================================================

    rain_future, rain_future_meta = (
        get_rain_forecast(
            end,
            forecast_days,
        )
    )


    future = rain_future.copy()


    # ========================================================
    # PROYECCIÓN DE CADA CAUDAL REALMENTE DISPONIBLE
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


    # ========================================================
    # CAUDAL PRINCIPAL FUTURO
    # ========================================================

    if (
        main_flow_col is not None
        and main_flow_col
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

    else:

        future[
            "caudal_m3s"
        ] = np.nan


    # ========================================================
    # FEATURES FUTURAS
    #
    # Para acumulados de lluvia necesitamos concatenar algunos
    # días anteriores con el futuro.
    # ========================================================

    tail_cols = [
        "datetime",
        "precip_mm",
        "caudal_m3s",
    ]


    for station in STATIONS:

        tail_cols.append(
            RAIN_COLUMNS[
                station
            ]
        )

        tail_cols.append(
            FLOW_COLUMNS[
                station
            ]
        )


    tail_cols = [
        col
        for col in tail_cols
        if col in history.columns
    ]


    combined = pd.concat(

        [
            history[
                tail_cols
            ].tail(
                45
            ),

            future[
                [
                    col
                    for col in tail_cols
                    if col
                    in future.columns
                ]
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


    future_start = (
        pd.to_datetime(
            end
        )
        + pd.Timedelta(
            days=1
        )
    ).normalize()


    combined[
        "datetime"
    ] = _to_naive_datetime(
        combined[
            "datetime"
        ]
    )


    future_features = combined[
        combined[
            "datetime"
        ]
        >= future_start
    ].copy()


    # Mantener exactamente forecast_days.
    future_features = (
        future_features
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


    future = future_features


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
    # METADATA COMPATIBLE CON APP V11.10
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
                "INA A5"
                if available_flow_stations
                else "Sin serie INA utilizable"
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

        "rain_history":
            rain_meta,

        "rain_forecast":
            rain_future_meta,

        "flow_history":
            flow_meta,

        # ----------------------------------------------------
        # APP V11.10 busca específicamente "flow_series".
        # ----------------------------------------------------

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

        "real_weather_forecast_days":
            OPEN_METEO_REAL_FORECAST_DAYS,

        "rain_after_real_forecast":
            (
                "No se interpreta como pronóstico meteorológico "
                "determinista después del horizonte disponible."
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
    # CATÁLOGO INA
    # ========================================================

    try:

        catalog = get_ina_catalog()


        caudal_catalog = catalog[
            catalog[
                "var_id"
            ]
            == VAR_ID_CAUDAL
        ].copy()


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
    # CANDIDATOS POR ESTACIÓN
    # ========================================================

    station_results = {}


    for station in STATIONS:

        station_info = {}


        try:

            candidates = (
                candidatos_caudal_estacion(
                    station,
                    start,
                    end,
                )
            )


            station_info[
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
                        "score",

                    ]

                    if col
                    in candidates.columns
                ]


                station_info[
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


            station_info[
                "selection"
            ] = meta


            if selected is not None:

                station_info[
                    "selected_series_id"
                ] = selected.get(
                    "series_id"
                )


                station_info[
                    "selected_name"
                ] = selected.get(
                    "nombre"
                )


                station_info[
                    "selected_river"
                ] = selected.get(
                    "rio"
                )


        except Exception as exc:

            station_info[
                "error"
            ] = str(
                exc
            )


        station_results[
            station
        ] = station_info


    result[
        "stations"
    ] = station_results


    # ========================================================
    # TEST COMPLETO
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
            "main_flow_station"
        ] = metadata.get(
            "main_flow_station"
        )


        result[
            "main_flow_column"
        ] = metadata.get(
            "main_flow_column"
        )


        # ----------------------------------------------------
        # COLUMNAS REALES QUE llegan al modelo
        # ----------------------------------------------------

        result[
            "rain_columns_present"
        ] = [

            col

            for col in RAIN_COLUMNS.values()

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

            for col in FLOW_COLUMNS.values()

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
