# ============================================================
# PARANÁ · SAN NICOLÁS
# src/exogenous.py
# V11.9.3 COMPLETO
#
# MEJORAS
# ------------------------------------------------------------
# - Lluvia histórica: Open-Meteo
# - Lluvia prevista: Open-Meteo
# - Caudal: catálogo oficial INA A5 GeoJSON
# - Variable INA Caudal: var_id = 4
# - Busca automáticamente la mejor serie disponible
# - Valida candidatos contra getObservaciones
# - Normaliza salida a:
#
#       datetime
#       precip_mm
#       caudal_m3s
#
# - Todas las fechas salen SIN timezone
# - Compatible con app/model actuales
# - Diagnóstico completo de la serie de caudal utilizada
# ============================================================


from functools import lru_cache
import unicodedata

import numpy as np
import pandas as pd
import requests


# ============================================================
# VERSION
# ============================================================

VERSION = "V11.9.3"


# ============================================================
# OPEN-METEO
# ============================================================

FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

HISTORICAL_WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
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


# ============================================================
# VARIABLE CAUDAL INA
# ============================================================

VAR_ID_CAUDAL = 4


# ============================================================
# HTTP
# ============================================================

REQUEST_TIMEOUT = 45

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
            "Parana-San-Nicolas/11.9.3",

        "Accept":
            "application/json",
    }
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

    "Diamante": (
        -32.0667,
        -60.6333,
    ),

    "Rosario": (
        -32.9442,
        -60.6505,
    ),

    "Villa Constitución": (
        -33.2278,
        -60.3297,
    ),

    "San Nicolás": (
        -33.3358,
        -60.2252,
    ),
}


# ============================================================
# ESTACIONES PREFERIDAS PARA CAUDAL
#
# No se inventa series_id.
# Se busca primero la estación más representativa disponible.
# ============================================================

CAUDAL_STATIONS = {

    "San Nicolás": [
        "San Nicolás",
        "San Nicolas",
    ],

    "Villa Constitución": [
        "Villa Constitución",
        "Villa Constitucion",
    ],

    "Rosario": [
        "Rosario",
    ],

    "Diamante": [
        "Diamante",
    ],

    "Paraná": [
        "Paraná",
        "Parana",
    ],

    "La Paz": [
        "La Paz",
    ],

    "Goya": [
        "Goya",
    ],

    "Corrientes": [
        "Corrientes",
    ],
}


CAUDAL_PRIORITY = [
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
# UTILIDADES
# ============================================================

def _normalizar_texto(
    value,
):

    if value is None:
        return ""

    text = str(
        value
    ).strip()

    text = unicodedata.normalize(
        "NFD",
        text,
    )

    text = "".join(
        c
        for c in text
        if unicodedata.category(
            c
        ) != "Mn"
    )

    text = (
        text
        .lower()
        .replace("_", " ")
        .replace("–", "-")
        .replace("—", "-")
    )

    return " ".join(
        text.split()
    )


def _to_datetime_naive(
    values,
):

    return (
        pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(
            None
        )
    )


def _normalizar_fecha(
    value,
):

    dt = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(
        dt
    ):
        return None

    return dt.strftime(
        "%Y-%m-%d"
    )


def _safe_int(
    value,
    default=None,
):

    try:

        if value is None:
            return default

        return int(
            float(
                value
            )
        )

    except Exception:

        return default


def _safe_float(
    value,
    default=np.nan,
):

    try:

        result = float(
            value
        )

        if np.isfinite(
            result
        ):

            return result

    except Exception:

        pass

    return default


def _request_json(
    url,
    params=None,
    timeout=
        REQUEST_TIMEOUT,
):

    response = SESSION.get(
        url,
        params=params,
        timeout=timeout,
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

    if pd.isna(
        end_dt
    ):

        end_dt = today

    end_dt = min(
        end_dt.normalize(),
        today,
    )

    if pd.isna(
        start_dt
    ):

        start_dt = (
            end_dt
            - pd.Timedelta(
                days=365
            )
        )

    start_dt = min(
        start_dt.normalize(),
        end_dt,
    )

    frames = []

    for (
        station,
        (
            latitude,
            longitude,
        ),
    ) in RAIN_POINTS.items():

        params = {

            "latitude":
                latitude,

            "longitude":
                longitude,

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

        except Exception:

            continue

        daily = data.get(
            "daily",
            {},
        )

        dates = daily.get(
            "time",
            [],
        )

        rain = daily.get(
            "precipitation_sum",
            [],
        )

        if (
            not dates
            or len(dates)
            != len(rain)
        ):

            continue

        column = (
            "rain_"
            + _normalizar_texto(
                station
            )
            .replace(
                " ",
                "_",
            )
        )

        frame = pd.DataFrame(
            {
                "datetime":
                    pd.to_datetime(
                        dates,
                        errors="coerce",
                    ),

                column:
                    pd.to_numeric(
                        rain,
                        errors="coerce",
                    ),
            }
        )

        frame[column] = (
            frame[column]
            .clip(
                lower=0.0
            )
        )

        frames.append(
            frame
        )

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
        col
        for col in result.columns
        if col.startswith(
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

    result = result[
        [
            "datetime",
            "precip_mm",
        ]
    ].copy()

    result[
        "datetime"
    ] = pd.to_datetime(
        result[
            "datetime"
        ],
        errors="coerce",
    )

    result = (
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

    return result


# ============================================================
# LLUVIA PREVISTA
# ============================================================

def get_rain_forecast(
    days=15,
):

    requested_days = max(
        1,
        int(
            days
        ),
    )

    # Open-Meteo permite un horizonte limitado.
    api_days = min(
        requested_days,
        16,
    )

    frames = []

    for (
        station,
        (
            latitude,
            longitude,
        ),
    ) in RAIN_POINTS.items():

        params = {

            "latitude":
                latitude,

            "longitude":
                longitude,

            "daily":
                "precipitation_sum",

            "forecast_days":
                api_days,

            "timezone":
                "America/Argentina/Buenos_Aires",
        }

        try:

            data = _request_json(
                FORECAST_URL,
                params=params,
            )

        except Exception:

            continue

        daily = data.get(
            "daily",
            {},
        )

        dates = daily.get(
            "time",
            [],
        )

        rain = daily.get(
            "precipitation_sum",
            [],
        )

        if (
            not dates
            or len(dates)
            != len(rain)
        ):

            continue

        column = (
            "rain_"
            + _normalizar_texto(
                station
            )
            .replace(
                " ",
                "_",
            )
        )

        frame = pd.DataFrame(
            {
                "datetime":
                    pd.to_datetime(
                        dates,
                        errors="coerce",
                    ),

                column:
                    pd.to_numeric(
                        rain,
                        errors="coerce",
                    ),
            }
        )

        frames.append(
            frame
        )

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
        col
        for col in result.columns
        if col.startswith(
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

    result = result[
        [
            "datetime",
            "precip_mm",
        ]
    ].copy()

    result = (
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
        .head(
            api_days
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# CATÁLOGO INA A5
# ============================================================

@lru_cache(
    maxsize=1
)
def _get_ina_catalog():

    data = _request_json(
        INA_SERIES_GEOJSON_URL,
        params={
            "format":
                "geojson",
        },
    )

    if not isinstance(
        data,
        dict,
    ):

        return pd.DataFrame()

    features = data.get(
        "features",
        [],
    )

    rows = []

    for feature in features:

        if not isinstance(
            feature,
            dict,
        ):

            continue

        properties = feature.get(
            "properties",
            {},
        )

        if not isinstance(
            properties,
            dict,
        ):

            continue

        row = properties.copy()

        if (
            row.get(
                "series_id"
            )
            is None
        ):

            row[
                "series_id"
            ] = feature.get(
                "id"
            )

        rows.append(
            row
        )

    catalog = pd.DataFrame(
        rows
    )

    if catalog.empty:

        return catalog

    for col in [
        "series_id",
        "var_id",
        "proc_id",
        "unit_id",
        "count",
    ]:

        if col in catalog.columns:

            catalog[col] = pd.to_numeric(
                catalog[col],
                errors="coerce",
            )

    for col in [
        "timestart",
        "timeend",
    ]:

        if col in catalog.columns:

            catalog[col] = pd.to_datetime(
                catalog[col],
                errors="coerce",
                utc=True,
            )

    if (
        "nombre"
        in catalog.columns
    ):

        catalog[
            "_name"
        ] = (
            catalog[
                "nombre"
            ]
            .fillna("")
            .map(
                _normalizar_texto
            )
        )

    else:

        catalog[
            "_name"
        ] = ""

    return catalog


# ============================================================
# SCORE DE ESTACIÓN
# ============================================================

def _station_score(
    station_name,
    aliases,
):

    name = _normalizar_texto(
        station_name
    )

    best = 0

    for alias in aliases:

        target = _normalizar_texto(
            alias
        )

        if not target:
            continue

        if name == target:

            score = 1000

        elif name.startswith(
            target + " "
        ):

            score = 900

        elif name.startswith(
            target + "-"
        ):

            score = 900

        elif (
            " "
            + target
            + " "
        ) in (
            " "
            + name
            + " "
        ):

            score = 800

        elif target in name:

            score = 700

        else:

            score = 0

        best = max(
            best,
            score,
        )

    return best


# ============================================================
# CANDIDATOS DE CAUDAL
# ============================================================

def _find_caudal_candidates(
    start=None,
    end=None,
):

    catalog = _get_ina_catalog()

    if catalog.empty:

        return pd.DataFrame()

    x = catalog.copy()

    if (
        "var_id"
        not in x.columns
    ):

        return pd.DataFrame()

    x = x[
        x[
            "var_id"
        ]
        == VAR_ID_CAUDAL
    ].copy()

    if x.empty:

        return x

    candidates = []

    for priority_index, station in enumerate(
        CAUDAL_PRIORITY
    ):

        aliases = CAUDAL_STATIONS[
            station
        ]

        temp = x.copy()

        temp[
            "_station_score"
        ] = temp[
            "nombre"
        ].fillna(
            ""
        ).apply(
            lambda value:
                _station_score(
                    value,
                    aliases,
                )
        )

        temp = temp[
            temp[
                "_station_score"
            ]
            >= 700
        ].copy()

        if temp.empty:

            continue

        temp[
            "_station"
        ] = station

        # Estaciones más cercanas a San Nicolás primero.
        temp[
            "_priority_score"
        ] = (
            len(
                CAUDAL_PRIORITY
            )
            - priority_index
        ) * 20

        candidates.append(
            temp
        )

    if not candidates:

        # ----------------------------------------------------
        # Fallback:
        # si no hay coincidencia con nuestros nombres,
        # permitir todas las series de Caudal.
        # ----------------------------------------------------

        x[
            "_station"
        ] = (
            x[
                "nombre"
            ]
            .fillna(
                "Serie INA"
            )
        )

        x[
            "_station_score"
        ] = 0

        x[
            "_priority_score"
        ] = 0

        result = x.copy()

    else:

        result = pd.concat(
            candidates,
            ignore_index=True,
        )

    # --------------------------------------------------------
    # Procedimiento
    # --------------------------------------------------------

    result[
        "_proc_score"
    ] = 0

    if (
        "proc_id"
        in result.columns
    ):

        result.loc[
            result[
                "proc_id"
            ] == 1,
            "_proc_score",
        ] = 80

        result.loc[
            result[
                "proc_id"
            ] == 2,
            "_proc_score",
        ] = 50

    # --------------------------------------------------------
    # Cantidad de datos
    # --------------------------------------------------------

    if (
        "count"
        in result.columns
    ):

        counts = (
            pd.to_numeric(
                result[
                    "count"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
        )

        result[
            "_count_score"
        ] = (
            np.log1p(
                counts
            )
            * 5
        )

    else:

        result[
            "_count_score"
        ] = 0

    # --------------------------------------------------------
    # Actualidad
    # --------------------------------------------------------

    result[
        "_recent_score"
    ] = 0

    if (
        "timeend"
        in result.columns
    ):

        now = pd.Timestamp.now(
            tz="UTC"
        )

        age = (
            now
            - result[
                "timeend"
            ]
        ).dt.days

        result.loc[
            age <= 7,
            "_recent_score",
        ] = 100

        result.loc[
            (age > 7)
            & (age <= 30),
            "_recent_score",
        ] = 80

        result.loc[
            (age > 30)
            & (age <= 180),
            "_recent_score",
        ] = 50

        result.loc[
            (age > 180)
            & (age <= 365),
            "_recent_score",
        ] = 25

    # --------------------------------------------------------
    # Solapamiento con rango solicitado
    # --------------------------------------------------------

    result[
        "_overlap_score"
    ] = 0

    start_dt = pd.to_datetime(
        start,
        errors="coerce",
        utc=True,
    )

    end_dt = pd.to_datetime(
        end,
        errors="coerce",
        utc=True,
    )

    if (
        pd.notna(
            start_dt
        )
        and pd.notna(
            end_dt
        )
        and "timestart"
        in result.columns
        and "timeend"
        in result.columns
    ):

        overlap = (
            (
                result[
                    "timestart"
                ].isna()
            )
            |
            (
                result[
                    "timestart"
                ]
                <= end_dt
            )
        ) & (
            (
                result[
                    "timeend"
                ].isna()
            )
            |
            (
                result[
                    "timeend"
                ]
                >= start_dt
            )
        )

        result.loc[
            overlap,
            "_overlap_score",
        ] = 150

    result[
        "_total_score"
    ] = (
        result[
            "_station_score"
        ]
        + result[
            "_priority_score"
        ]
        + result[
            "_proc_score"
        ]
        + result[
            "_count_score"
        ]
        + result[
            "_recent_score"
        ]
        + result[
            "_overlap_score"
        ]
    )

    result = (
        result
        .sort_values(
            "_total_score",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "series_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# EXTRAER OBSERVACIONES DEL JSON
# ============================================================

def _extract_observation_records(
    data,
):

    if isinstance(
        data,
        list,
    ):

        if (
            data
            and all(
                isinstance(
                    item,
                    dict,
                )
                for item in data
            )
        ):

            keys = set()

            for item in data[:5]:

                keys.update(
                    item.keys()
                )

            if keys.intersection(
                {
                    "timestart",
                    "valor",
                    "value",
                    "datetime",
                    "fecha",
                    "time",
                }
            ):

                return data

        for item in data:

            records = (
                _extract_observation_records(
                    item
                )
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
            "datos",
            "records",
            "values",
            "result",
        ]:

            if key in data:

                records = (
                    _extract_observation_records(
                        data[
                            key
                        ]
                    )
                )

                if records:

                    return records

        for value in data.values():

            records = (
                _extract_observation_records(
                    value
                )
            )

            if records:

                return records

    return []


# ============================================================
# NORMALIZAR CAUDAL
# ============================================================

def _normalizar_caudal_response(
    data,
):

    records = (
        _extract_observation_records(
            data
        )
    )

    if not records:

        return pd.DataFrame(
            columns=[
                "datetime",
                "caudal_m3s",
            ]
        )

    rows = []

    date_fields = [
        "timestart",
        "datetime",
        "timestamp",
        "fecha",
        "date",
        "time",
        "obs_date",
    ]

    value_fields = [
        "valor",
        "value",
        "caudal",
        "obs_value",
    ]

    for record in records:

        date_value = None
        flow_value = None

        for field in date_fields:

            if (
                field in record
                and record[
                    field
                ] is not None
            ):

                date_value = record[
                    field
                ]

                break

        for field in value_fields:

            if (
                field in record
                and record[
                    field
                ] is not None
            ):

                flow_value = record[
                    field
                ]

                break

        dt = pd.to_datetime(
            date_value,
            errors="coerce",
            utc=True,
        )

        flow = pd.to_numeric(
            flow_value,
            errors="coerce",
        )

        if (
            pd.isna(
                dt
            )
            or pd.isna(
                flow
            )
        ):

            continue

        flow = float(
            flow
        )

        # Rango amplio de control de calidad.
        if (
            flow < 0
            or flow > 200000
        ):

            continue

        rows.append(
            {
                "datetime":
                    dt,

                "caudal_m3s":
                    flow,
            }
        )

    if not rows:

        return pd.DataFrame(
            columns=[
                "datetime",
                "caudal_m3s",
            ]
        )

    df = pd.DataFrame(
        rows
    )

    df[
        "datetime"
    ] = _to_datetime_naive(
        df[
            "datetime"
        ]
    )

    df[
        "datetime"
    ] = (
        df[
            "datetime"
        ]
        .dt
        .normalize()
    )

    # Varias observaciones en un mismo día:
    # utilizar mediana diaria.
    df = (
        df
        .groupby(
            "datetime",
            as_index=False,
        )[
            "caudal_m3s"
        ]
        .median()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# CONSULTAR UNA SERIE DE CAUDAL
# ============================================================

def _query_caudal_series(
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
            _normalizar_fecha(
                start
            ),

        "timeend":
            _normalizar_fecha(
                end
            ),
    }

    data = _request_json(
        INA_OBSERVATIONS_URL,
        params=params,
    )

    return (
        _normalizar_caudal_response(
            data
        )
    )


# ============================================================
# SELECCIONAR MEJOR SERIE REAL
# ============================================================

def find_best_caudal_series(
    start=None,
    end=None,
):

    candidates = (
        _find_caudal_candidates(
            start,
            end,
        )
    )

    if candidates.empty:

        return None

    requested_end = pd.to_datetime(
        end,
        errors="coerce",
        utc=True,
    )

    if pd.isna(
        requested_end
    ):

        requested_end = (
            pd.Timestamp.now(
                tz="UTC"
            )
        )

    for index, row in (
        candidates
        .head(20)
        .iterrows()
    ):

        series_id = _safe_int(
            row.get(
                "series_id"
            )
        )

        if series_id is None:

            continue

        catalog_start = pd.to_datetime(
            row.get(
                "timestart"
            ),
            errors="coerce",
            utc=True,
        )

        catalog_end = pd.to_datetime(
            row.get(
                "timeend"
            ),
            errors="coerce",
            utc=True,
        )

        validation_end = (
            min(
                catalog_end,
                requested_end,
            )
            if pd.notna(
                catalog_end
            )
            else requested_end
        )

        validation_start = (
            validation_end
            - pd.Timedelta(
                days=180
            )
        )

        if pd.notna(
            catalog_start
        ):

            validation_start = max(
                validation_start,
                catalog_start,
            )

        if (
            validation_start
            > validation_end
        ):

            continue

        try:

            test = _query_caudal_series(
                series_id,
                validation_start,
                validation_end,
            )

        except Exception:

            continue

        if test.empty:

            continue

        return {

            "series_id":
                series_id,

            "station":
                row.get(
                    "_station"
                )
                or row.get(
                    "nombre"
                ),

            "series_name":
                row.get(
                    "nombre"
                ),

            "var_id":
                _safe_int(
                    row.get(
                        "var_id"
                    )
                ),

            "variable":
                row.get(
                    "var_nombre"
                ),

            "proc_id":
                _safe_int(
                    row.get(
                        "proc_id"
                    )
                ),

            "unit_id":
                _safe_int(
                    row.get(
                        "unit_id"
                    )
                ),

            "source":
                row.get(
                    "fuente"
                ),

            "availability":
                row.get(
                    "data_availability"
                ),

            "catalog_count":
                _safe_int(
                    row.get(
                        "count"
                    ),
                    0,
                ),

            "catalog_start":
                (
                    catalog_start.strftime(
                        "%Y-%m-%d"
                    )
                    if pd.notna(
                        catalog_start
                    )
                    else None
                ),

            "catalog_end":
                (
                    catalog_end.strftime(
                        "%Y-%m-%d"
                    )
                    if pd.notna(
                        catalog_end
                    )
                    else None
                ),

            "validation_records":
                int(
                    len(
                        test
                    )
                ),

            "candidate_number":
                int(
                    index + 1
                ),
        }

    return None


# ============================================================
# CAUDAL HISTÓRICO
# ============================================================

def get_caudal_history(
    start,
    end,
):

    empty = pd.DataFrame(
        columns=[
            "datetime",
            "caudal_m3s",
        ]
    )

    info = find_best_caudal_series(
        start,
        end,
    )

    if info is None:

        return (
            empty,
            {
                "status":
                    "sin_serie",

                "message":
                    "INA no devolvió una serie de caudal "
                    "validada para las estaciones analizadas.",
            },
        )

    try:

        df = _query_caudal_series(
            info[
                "series_id"
            ],
            start,
            end,
        )

    except Exception as exc:

        info = info.copy()

        info[
            "status"
        ] = "error_consulta"

        info[
            "error"
        ] = str(
            exc
        )

        return (
            empty,
            info,
        )

    info = info.copy()

    if df.empty:

        info[
            "status"
        ] = "sin_observaciones"

        info[
            "records"
        ] = 0

        return (
            empty,
            info,
        )

    info[
        "status"
    ] = "ok"

    info[
        "records"
    ] = int(
        len(
            df
        )
    )

    info[
        "first_date"
    ] = (
        df[
            "datetime"
        ]
        .min()
        .strftime(
            "%Y-%m-%d"
        )
    )

    info[
        "last_date"
    ] = (
        df[
            "datetime"
        ]
        .max()
        .strftime(
            "%Y-%m-%d"
        )
    )

    info[
        "current_flow"
    ] = float(
        df[
            "caudal_m3s"
        ]
        .dropna()
        .iloc[-1]
    )

    return (
        df,
        info,
    )


# ============================================================
# PROYECCIÓN DE CAUDAL
# ============================================================

def project_caudal(
    history,
    future_dates,
):

    dates = pd.to_datetime(
        future_dates,
        errors="coerce",
    )

    output = pd.DataFrame(
        {
            "datetime":
                dates,
        }
    )

    output[
        "caudal_m3s"
    ] = np.nan

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

        return output

    q = history.copy()

    q[
        "caudal_m3s"
    ] = pd.to_numeric(
        q[
            "caudal_m3s"
        ],
        errors="coerce",
    )

    values = (
        q[
            "caudal_m3s"
        ]
        .dropna()
        .tail(
            14
        )
    )

    if values.empty:

        return output

    current = float(
        values.iloc[-1]
    )

    if len(
        values
    ) >= 5:

        y = values.to_numpy(
            dtype=float
        )

        x = np.arange(
            len(
                y
            ),
            dtype=float,
        )

        slope = float(
            np.polyfit(
                x,
                y,
                1,
            )[0]
        )

        # Evitar proyecciones absurdas.
        max_daily = max(
            current * 0.025,
            50.0,
        )

        slope = float(
            np.clip(
                slope,
                -max_daily,
                max_daily,
            )
        )

    else:

        slope = 0.0

    projected = []

    level = current

    for step in range(
        1,
        len(
            output
        )
        + 1,
    ):

        damping = np.exp(
            -step
            / 12.0
        )

        daily_change = (
            slope
            * damping
        )

        level = max(
            0.0,
            level
            + daily_change,
        )

        projected.append(
            level
        )

    output[
        "caudal_m3s"
    ] = projected

    return output


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def get_exogenous_data(
    start,
    end,
    forecast_days=15,
):

    start_text = _normalizar_fecha(
        start
    )

    end_text = _normalizar_fecha(
        end
    )

    if (
        start_text is None
        or end_text is None
    ):

        raise ValueError(
            "Fechas inválidas para variables exógenas."
        )

    # ========================================================
    # LLUVIA HISTÓRICA
    # ========================================================

    rain_history = get_rain_history(
        start_text,
        end_text,
    )

    # ========================================================
    # LLUVIA FUTURA
    # ========================================================

    rain_future = get_rain_forecast(
        forecast_days,
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

    if rain_history.empty:

        history = pd.DataFrame(
            {
                "datetime":
                    pd.date_range(
                        start=start_text,
                        end=end_text,
                        freq="D",
                    ),

                "precip_mm":
                    0.0,
            }
        )

    else:

        history = (
            rain_history.copy()
        )

    history[
        "datetime"
    ] = pd.to_datetime(
        history[
            "datetime"
        ],
        errors="coerce",
    )

    if (
        caudal_history is not None
        and isinstance(
            caudal_history,
            pd.DataFrame,
        )
        and not caudal_history.empty
    ):

        flow = (
            caudal_history.copy()
        )

        flow[
            "datetime"
        ] = pd.to_datetime(
            flow[
                "datetime"
            ],
            errors="coerce",
        )

        history = history.merge(
            flow,
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

    history = (
        history
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

    # ========================================================
    # FUTURO
    # ========================================================

    forecast_days = max(
        1,
        int(
            forecast_days
        ),
    )

    # Fechas futuras respecto a la última fecha seleccionada.
    end_dt = pd.to_datetime(
        end_text
    )

    future_dates = pd.date_range(
        start=
            end_dt
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

    # --------------------------------------------------------
    # Lluvia prevista únicamente puede corresponder a fechas
    # actuales/futuras reales de Open-Meteo.
    # --------------------------------------------------------

    if (
        rain_future is not None
        and isinstance(
            rain_future,
            pd.DataFrame,
        )
        and not rain_future.empty
    ):

        rain_future = (
            rain_future.copy()
        )

        rain_future[
            "datetime"
        ] = pd.to_datetime(
            rain_future[
                "datetime"
            ],
            errors="coerce",
        )

        future = (
            future
            .drop(
                columns=[
                    "precip_mm"
                ]
            )
            .merge(
                rain_future[
                    [
                        "datetime",
                        "precip_mm",
                    ]
                ],
                on="datetime",
                how="left",
            )
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

    # --------------------------------------------------------
    # Proyección de caudal
    # --------------------------------------------------------

    caudal_future = project_caudal(
        caudal_history,
        future[
            "datetime"
        ],
    )

    future = future.merge(
        caudal_future,
        on="datetime",
        how="left",
    )

    future[
        "datetime"
    ] = pd.to_datetime(
        future[
            "datetime"
        ],
        errors="coerce",
    )

    # ========================================================
    # META
    # ========================================================

    valid_flow = int(
        history[
            "caudal_m3s"
        ]
        .notna()
        .sum()
    )

    meta = {

        "version":
            VERSION,

        "rain_source":
            "Open-Meteo",

        "rain_points":
            list(
                RAIN_POINTS.keys()
            ),

        "rain_history_records":
            int(
                history[
                    "precip_mm"
                ]
                .notna()
                .sum()
            ),

        "rain_history_max_mm":
            (
                float(
                    history[
                        "precip_mm"
                    ]
                    .max()
                )
                if not history.empty
                else 0.0
            ),

        "caudal_source":
            "INA A5",

        "caudal_series":
            caudal_info,

        "caudal_records":
            valid_flow,

        "uses_caudal":
            bool(
                valid_flow > 0
            ),

        "historical_start":
            start_text,

        "historical_end":
            end_text,

        "forecast_days":
            forecast_days,

        "datetime_timezone":
            "naive",
    }

    return (
        history,
        future,
        meta,
    )


# ============================================================
# DIAGNÓSTICO
# ============================================================

def diagnostic(
    start,
    end,
):

    report = {

        "version":
            VERSION,

        "ina_catalog":
            INA_SERIES_GEOJSON_URL,

        "ina_observations":
            INA_OBSERVATIONS_URL,

        "caudal_var_id":
            VAR_ID_CAUDAL,

        "start":
            _normalizar_fecha(
                start
            ),

        "end":
            _normalizar_fecha(
                end
            ),

        "catalog_records":
            0,

        "caudal_candidates":
            0,

        "selected_series":
            None,
    }

    try:

        catalog = _get_ina_catalog()

        report[
            "catalog_records"
        ] = int(
            len(
                catalog
            )
        )

    except Exception as exc:

        report[
            "catalog_error"
        ] = str(
            exc
        )

        return report

    try:

        candidates = (
            _find_caudal_candidates(
                start,
                end,
            )
        )

        report[
            "caudal_candidates"
        ] = int(
            len(
                candidates
            )
        )

        preview = []

        for _, row in (
            candidates
            .head(10)
            .iterrows()
        ):

            preview.append(
                {
                    "series_id":
                        _safe_int(
                            row.get(
                                "series_id"
                            )
                        ),

                    "station":
                        row.get(
                            "_station"
                        ),

                    "name":
                        row.get(
                            "nombre"
                        ),

                    "var_id":
                        _safe_int(
                            row.get(
                                "var_id"
                            )
                        ),

                    "proc_id":
                        _safe_int(
                            row.get(
                                "proc_id"
                            )
                        ),

                    "count":
                        _safe_int(
                            row.get(
                                "count"
                            ),
                            0,
                        ),

                    "score":
                        _safe_float(
                            row.get(
                                "_total_score"
                            ),
                            0.0,
                        ),
                }
            )

        report[
            "candidate_preview"
        ] = preview

    except Exception as exc:

        report[
            "candidate_error"
        ] = str(
            exc
        )

    try:

        selected = (
            find_best_caudal_series(
                start,
                end,
            )
        )

        report[
            "selected_series"
        ] = selected

    except Exception as exc:

        report[
            "selection_error"
        ] = str(
            exc
        )

    return report
