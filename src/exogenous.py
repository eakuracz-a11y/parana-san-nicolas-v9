# ============================================================
# PARANÁ · SAN NICOLÁS
# src/exogenous.py
# V11.9.8 COMPLETO
#
# OBJETIVOS
# ------------------------------------------------------------
# - Lluvia histórica Open-Meteo
# - Lluvia prevista Open-Meteo
# - Caudal INA A5
# - Catálogo oficial GeoJSON A5
# - var_id = 4 = Caudal
# - Buscar automáticamente la mejor serie
# - Priorizar estaciones del río Paraná
# - Validar cada candidato mediante getObservaciones
# - Evitar usar caudales de ríos no relacionados
# - Proyectar caudal hasta 60 días
# - Salida compatible con model.py V11.9.7
#
# SALIDA:
#
# history:
#   datetime
#   precip_mm
#   caudal_m3s
#
# future:
#   datetime
#   precip_mm
#   caudal_m3s
#
# meta:
#   diagnóstico completo
#
# ============================================================


from functools import lru_cache

import unicodedata

import numpy as np
import pandas as pd
import requests


# ============================================================
# VERSIÓN
# ============================================================

VERSION = "V11.9.8"


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
# VARIABLE CAUDAL
# ============================================================

VAR_ID_CAUDAL = 4


# ============================================================
# CONFIGURACIÓN
# ============================================================

REQUEST_TIMEOUT = 45

MAX_FORECAST_DAYS = 60

CAUDAL_MIN = 0.0

CAUDAL_MAX = 200000.0


# ============================================================
# ESTACIONES DE LLUVIA
# ============================================================

RAIN_POINTS = {

    "Corrientes": {
        "latitude": -27.4692,
        "longitude": -58.8306,
    },

    "Goya": {
        "latitude": -29.1400,
        "longitude": -59.2634,
    },

    "La Paz": {
        "latitude": -30.7449,
        "longitude": -59.6457,
    },

    "Paraná": {
        "latitude": -31.7319,
        "longitude": -60.5238,
    },

    "Diamante": {
        "latitude": -32.0664,
        "longitude": -60.6384,
    },

    "Rosario": {
        "latitude": -32.9442,
        "longitude": -60.6505,
    },

    "Villa Constitución": {
        "latitude": -33.2272,
        "longitude": -60.3296,
    },

    "San Nicolás": {
        "latitude": -33.3358,
        "longitude": -60.2252,
    },
}


# ============================================================
# PRIORIDAD DE CAUDAL
#
# El caudal no necesariamente existe en San Nicolás.
# Por eso se busca de abajo hacia arriba.
# ============================================================

CAUDAL_STATION_PRIORITY = [

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

CAUDAL_ALIASES = {

    "San Nicolás": [
        "san nicolas",
        "san nicolás",
    ],

    "Villa Constitución": [
        "villa constitucion",
        "villa constitución",
    ],

    "Rosario": [
        "rosario",
    ],

    "Diamante": [
        "diamante",
    ],

    "Paraná": [
        "parana",
        "paraná",
    ],

    "La Paz": [
        "la paz",
    ],

    "Goya": [
        "goya",
    ],

    "Corrientes": [
        "corrientes",
    ],
}


# ============================================================
# REQUEST SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
            "Parana-San-Nicolas/11.9.8",

        "Accept":
            "application/json",
    }
)


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
    ).strip().lower()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        c
        for c in text
        if not unicodedata.combining(
            c
        )
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

        raise ValueError(
            f"Fecha inválida: {value}"
        )

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

        result = int(
            float(
                value
            )
        )

        return result

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
):

    response = SESSION.get(

        url,

        params=
            params,

        timeout=
            REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# LLUVIA HISTÓRICA DE UN PUNTO
# ============================================================

def _rain_history_point(
    latitude,
    longitude,
    start,
    end,
):

    params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "start_date":
            _normalizar_fecha(
                start
            ),

        "end_date":
            _normalizar_fecha(
                end
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

        return pd.DataFrame()


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
        or not rain
    ):

        return pd.DataFrame()


    result = pd.DataFrame(
        {
            "datetime":
                dates,

            "precip_mm":
                rain,
        }
    )


    result[
        "datetime"
    ] = _to_datetime_naive(
        result[
            "datetime"
        ]
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


    return result.dropna(
        subset=[
            "datetime"
        ]
    )


# ============================================================
# LLUVIA HISTÓRICA CORREDOR
# ============================================================

def get_rain_history(
    start,
    end,
):

    frames = []


    for (
        station,
        coords,
    ) in RAIN_POINTS.items():

        try:

            frame = (
                _rain_history_point(

                    coords[
                        "latitude"
                    ],

                    coords[
                        "longitude"
                    ],

                    start,
                    end,
                )
            )

        except Exception:

            continue


        if frame.empty:
            continue


        frame = frame.rename(
            columns={
                "precip_mm":
                    (
                        "rain_"
                        + _normalizar_texto(
                            station
                        )
                        .replace(
                            " ",
                            "_",
                        )
                    )
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
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LLUVIA FUTURA POR PUNTO
# ============================================================

def _rain_forecast_point(
    latitude,
    longitude,
    days=15,
):

    api_days = min(
        max(
            int(
                days
            ),
            1,
        ),
        16,
    )


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

        return pd.DataFrame()


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
        or not rain
    ):

        return pd.DataFrame()


    result = pd.DataFrame(
        {
            "datetime":
                dates,

            "precip_mm":
                rain,
        }
    )


    result[
        "datetime"
    ] = _to_datetime_naive(
        result[
            "datetime"
        ]
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


    return result


# ============================================================
# LLUVIA FUTURA CORREDOR
# ============================================================

def get_rain_forecast(
    days=15,
):

    api_days = min(
        max(
            int(
                days
            ),
            1,
        ),
        16,
    )


    frames = []


    for (
        station,
        coords,
    ) in RAIN_POINTS.items():

        try:

            frame = (
                _rain_forecast_point(

                    coords[
                        "latitude"
                    ],

                    coords[
                        "longitude"
                    ],

                    api_days,
                )
            )

        except Exception:

            continue


        if frame.empty:
            continue


        frame = frame.rename(
            columns={
                "precip_mm":
                    (
                        "rain_"
                        + _normalizar_texto(
                            station
                        )
                        .replace(
                            " ",
                            "_",
                        )
                    )
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
            api_days
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CATÁLOGO INA A5
# ============================================================

@lru_cache(
    maxsize=1
)
def _get_ina_catalog():

    try:

        data = _request_json(

            INA_SERIES_GEOJSON_URL,

            params={
                "format":
                    "geojson",
            },
        )

    except Exception:

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


        props = feature.get(
            "properties",
            {},
        )


        if not isinstance(
            props,
            dict,
        ):
            continue


        row = dict(
            props
        )


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


        geometry = feature.get(
            "geometry"
        )


        if isinstance(
            geometry,
            dict,
        ):

            coords = geometry.get(
                "coordinates"
            )


            if (
                isinstance(
                    coords,
                    list,
                )
                and len(
                    coords
                )
                >= 2
            ):

                row[
                    "longitude"
                ] = coords[
                    0
                ]

                row[
                    "latitude"
                ] = coords[
                    1
                ]


        rows.append(
            row
        )


    if not rows:

        return pd.DataFrame()


    catalog = pd.DataFrame(
        rows
    )


    for col in [
        "series_id",
        "var_id",
        "proc_id",
        "unit_id",
        "count",
    ]:

        if col in catalog.columns:

            catalog[
                col
            ] = pd.to_numeric(
                catalog[
                    col
                ],
                errors="coerce",
            )


    for col in [
        "timestart",
        "timeend",
    ]:

        if col in catalog.columns:

            catalog[
                col
            ] = pd.to_datetime(
                catalog[
                    col
                ],
                errors="coerce",
                utc=True,
            )


    if (
        "nombre"
        not in catalog.columns
    ):

        catalog[
            "nombre"
        ] = ""


    if (
        "rio"
        not in catalog.columns
    ):

        catalog[
            "rio"
        ] = ""


    catalog[
        "_nombre_normalizado"
    ] = catalog[
        "nombre"
    ].apply(
        _normalizar_texto
    )


    catalog[
        "_rio_normalizado"
    ] = catalog[
        "rio"
    ].apply(
        _normalizar_texto
    )


    return catalog


# ============================================================
# SCORE DE ESTACIÓN
# ============================================================

def _station_match_score(
    station,
    series_name,
):

    series_name = (
        _normalizar_texto(
            series_name
        )
    )


    aliases = (
        CAUDAL_ALIASES.get(
            station,
            [
                station
            ],
        )
    )


    best = 0


    for alias in aliases:

        alias = (
            _normalizar_texto(
                alias
            )
        )


        if not alias:
            continue


        if (
            series_name
            == alias
        ):

            best = max(
                best,
                1000,
            )


        elif (
            series_name.startswith(
                alias
            )
        ):

            best = max(
                best,
                900,
            )


        elif (
            f" {alias} "
            in
            f" {series_name} "
        ):

            best = max(
                best,
                850,
            )


        elif (
            alias
            in series_name
        ):

            best = max(
                best,
                750,
            )


    return best


# ============================================================
# IDENTIFICAR SI ES PARANÁ
# ============================================================

def _parana_score(
    rio,
    nombre,
):

    rio_text = _normalizar_texto(
        rio
    )

    nombre_text = (
        _normalizar_texto(
            nombre
        )
    )


    score = 0


    if "parana" in rio_text:

        score += 500


    if (
        "rio parana"
        in nombre_text
    ):

        score += 200


    # Penalizaciones para tributarios conocidos
    # cuando el catálogo los explicita.

    tributaries = [
        "paraguay",
        "uruguay",
        "salado",
        "carcarana",
        "gualeguay",
        "gualeguaychu",
        "bermejo",
        "pilcomayo",
        "iguazu",
    ]


    for river in tributaries:

        if (
            river
            in rio_text
            and "parana"
            not in rio_text
        ):

            score -= 500


    return score


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


    if (
        "var_id"
        not in catalog.columns
    ):

        return pd.DataFrame()


    candidates = catalog[
        catalog[
            "var_id"
        ]
        == VAR_ID_CAUDAL
    ].copy()


    if candidates.empty:

        return pd.DataFrame()


    requested_start = pd.to_datetime(
        start,
        errors="coerce",
        utc=True,
    )


    requested_end = pd.to_datetime(
        end,
        errors="coerce",
        utc=True,
    )


    rows = []


    for _, row in candidates.iterrows():

        series_id = _safe_int(
            row.get(
                "series_id"
            )
        )


        if series_id is None:
            continue


        series_name = str(
            row.get(
                "nombre",
                "",
            )
            or ""
        )


        rio = str(
            row.get(
                "rio",
                "",
            )
            or ""
        )


        best_station = None

        station_score = 0

        priority_score = 0


        for (
            priority_index,
            station,
        ) in enumerate(
            CAUDAL_STATION_PRIORITY
        ):

            score = (
                _station_match_score(
                    station,
                    series_name,
                )
            )


            if (
                score
                > station_score
            ):

                station_score = score

                best_station = station

                priority_score = (
                    len(
                        CAUDAL_STATION_PRIORITY
                    )
                    - priority_index
                ) * 20


        parana_score = (
            _parana_score(
                rio,
                series_name,
            )
        )


        # ----------------------------------------------------
        # Si no encontramos estación prioritaria y tampoco
        # está relacionado claramente con Paraná, descartamos.
        # ----------------------------------------------------

        if (
            station_score
            <= 0
            and parana_score
            <= 0
        ):

            continue


        proc_id = _safe_int(
            row.get(
                "proc_id"
            ),
            -1,
        )


        count = _safe_int(
            row.get(
                "count"
            ),
            0,
        )


        catalog_start = row.get(
            "timestart"
        )

        catalog_end = row.get(
            "timeend"
        )


        total_score = (
            station_score
            + priority_score
            + parana_score
        )


        # ----------------------------------------------------
        # PROCEDIMIENTO
        # ----------------------------------------------------

        if proc_id == 1:

            total_score += 100

        elif proc_id == 2:

            total_score += 60


        # ----------------------------------------------------
        # CANTIDAD DE DATOS
        # ----------------------------------------------------

        if count > 10000:

            total_score += 100

        elif count > 1000:

            total_score += 70

        elif count > 100:

            total_score += 40

        elif count > 0:

            total_score += 20


        # ----------------------------------------------------
        # RECENCIA
        # ----------------------------------------------------

        if pd.notna(
            catalog_end
        ):

            now = pd.Timestamp.now(
                tz="UTC"
            )

            age_days = (
                now
                - catalog_end
            ).days


            if age_days <= 7:

                total_score += 150

            elif age_days <= 30:

                total_score += 120

            elif age_days <= 180:

                total_score += 80

            elif age_days <= 365:

                total_score += 40


        # ----------------------------------------------------
        # SOLAPAMIENTO CON PERÍODO SOLICITADO
        # ----------------------------------------------------

        overlap = True


        if (
            pd.notna(
                requested_start
            )
            and pd.notna(
                catalog_end
            )
            and catalog_end
            < requested_start
        ):

            overlap = False


        if (
            pd.notna(
                requested_end
            )
            and pd.notna(
                catalog_start
            )
            and catalog_start
            > requested_end
        ):

            overlap = False


        if overlap:

            total_score += 150

        else:

            total_score -= 500


        rows.append(
            {

                "series_id":
                    series_id,

                "station":
                    best_station,

                "series_name":
                    series_name,

                "rio":
                    rio,

                "var_id":
                    VAR_ID_CAUDAL,

                "proc_id":
                    proc_id,

                "unit_id":
                    _safe_int(
                        row.get(
                            "unit_id"
                        )
                    ),

                "count":
                    count,

                "timestart":
                    catalog_start,

                "timeend":
                    catalog_end,

                "latitude":
                    row.get(
                        "latitude"
                    ),

                "longitude":
                    row.get(
                        "longitude"
                    ),

                "score":
                    total_score,

                "station_score":
                    station_score,

                "parana_score":
                    parana_score,

                "overlap":
                    overlap,
            }
        )


    if not rows:

        return pd.DataFrame()


    result = pd.DataFrame(
        rows
    )


    result = (
        result
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


    return result


# ============================================================
# PARSER OBSERVACIONES INA
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


            for item in data[
                :5
            ]:

                keys.update(
                    item.keys()
                )


            if keys.intersection(
                {
                    "timestart",
                    "datetime",
                    "fecha",
                    "valor",
                    "value",
                }
            ):

                return data


        for item in data:

            result = (
                _extract_observation_records(
                    item
                )
            )

            if result:

                return result


        return []


    if isinstance(
        data,
        dict,
    ):

        preferred = [

            "observaciones",
            "observations",
            "datos",
            "data",
            "records",
            "values",
            "result",
        ]


        for key in preferred:

            if key in data:

                result = (
                    _extract_observation_records(
                        data[
                            key
                        ]
                    )
                )

                if result:

                    return result


        for value in data.values():

            result = (
                _extract_observation_records(
                    value
                )
            )

            if result:

                return result


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
        "valor_num",
    ]


    rows = []


    for record in records:

        if not isinstance(
            record,
            dict,
        ):

            continue


        dt = None

        value = None


        for field in date_fields:

            if (
                field in record
                and record[
                    field
                ]
                is not None
            ):

                dt = record[
                    field
                ]

                break


        for field in value_fields:

            if (
                field in record
                and record[
                    field
                ]
                is not None
            ):

                value = record[
                    field
                ]

                break


        dt = pd.to_datetime(
            dt,
            errors="coerce",
            utc=True,
        )


        value = pd.to_numeric(
            value,
            errors="coerce",
        )


        if (
            pd.isna(
                dt
            )
            or pd.isna(
                value
            )
        ):

            continue


        value = float(
            value
        )


        if (
            value
            < CAUDAL_MIN
            or value
            > CAUDAL_MAX
        ):

            continue


        rows.append(
            {
                "datetime":
                    dt,

                "caudal_m3s":
                    value,
            }
        )


    if not rows:

        return pd.DataFrame(
            columns=[
                "datetime",
                "caudal_m3s",
            ]
        )


    result = pd.DataFrame(
        rows
    )


    result[
        "datetime"
    ] = _to_datetime_naive(
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


    result = (
        result
        .dropna(
            subset=[
                "datetime",
                "caudal_m3s",
            ]
        )
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


    return result


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

        params=
            params,
    )


    return (
        _normalizar_caudal_response(
            data
        )
    )


# ============================================================
# VENTANA DE VALIDACIÓN
# ============================================================

def _candidate_validation_window(
    candidate,
    requested_start,
    requested_end,
):

    req_start = pd.to_datetime(
        requested_start,
        errors="coerce",
        utc=True,
    )


    req_end = pd.to_datetime(
        requested_end,
        errors="coerce",
        utc=True,
    )


    cat_start = candidate.get(
        "timestart"
    )


    cat_end = candidate.get(
        "timeend"
    )


    if pd.isna(
        req_end
    ):

        req_end = pd.Timestamp.now(
            tz="UTC"
        )


    if pd.isna(
        req_start
    ):

        req_start = (
            req_end
            - pd.Timedelta(
                days=365
            )
        )


    overlap_start = req_start

    overlap_end = req_end


    if pd.notna(
        cat_start
    ):

        overlap_start = max(
            overlap_start,
            cat_start,
        )


    if pd.notna(
        cat_end
    ):

        overlap_end = min(
            overlap_end,
            cat_end,
        )


    if (
        overlap_end
        < overlap_start
    ):

        return (
            None,
            None,
        )


    validation_start = max(

        overlap_start,

        overlap_end
        - pd.Timedelta(
            days=180
        ),
    )


    return (

        validation_start
        .tz_localize(
            None
        ),

        overlap_end
        .tz_localize(
            None
        ),
    )


# ============================================================
# BUSCAR Y VALIDAR MEJOR SERIE
# ============================================================

def find_best_caudal_series(
    start=None,
    end=None,
):

    if end is None:

        end = pd.Timestamp.today().strftime(
            "%Y-%m-%d"
        )


    if start is None:

        start = (
            pd.Timestamp.today()
            - pd.Timedelta(
                days=365
            )
        ).strftime(
            "%Y-%m-%d"
        )


    candidates = (
        _find_caudal_candidates(
            start,
            end,
        )
    )


    if candidates.empty:

        return None


    # ========================================================
    # PROBAR LOS MEJORES 30
    # ========================================================

    for index, candidate in (
        candidates
        .head(
            30
        )
        .iterrows()
    ):

        (
            validation_start,
            validation_end,
        ) = (
            _candidate_validation_window(
                candidate,
                start,
                end,
            )
        )


        if (
            validation_start
            is None
            or validation_end
            is None
        ):

            continue


        try:

            observations = (
                _query_caudal_series(

                    int(
                        candidate[
                            "series_id"
                        ]
                    ),

                    validation_start,

                    validation_end,
                )
            )

        except Exception:

            continue


        if (
            observations.empty
            or "caudal_m3s"
            not in observations.columns
        ):

            continue


        valid = (
            observations[
                "caudal_m3s"
            ]
            .dropna()
        )


        if len(
            valid
        ) < 3:

            continue


        current_flow = float(
            valid.iloc[-1]
        )


        # ----------------------------------------------------
        # Validación física básica
        # ----------------------------------------------------

        if (
            current_flow
            <= 0
            or current_flow
            > CAUDAL_MAX
        ):

            continue


        return {

            "series_id":
                int(
                    candidate[
                        "series_id"
                    ]
                ),

            "station":
                candidate.get(
                    "station"
                ),

            "series_name":
                candidate.get(
                    "series_name"
                ),

            "river":
                candidate.get(
                    "rio"
                ),

            "var_id":
                VAR_ID_CAUDAL,

            "variable":
                "Caudal",

            "proc_id":
                _safe_int(
                    candidate.get(
                        "proc_id"
                    )
                ),

            "unit_id":
                _safe_int(
                    candidate.get(
                        "unit_id"
                    )
                ),

            "source":
                "INA A5",

            "availability":
                True,

            "catalog_count":
                _safe_int(
                    candidate.get(
                        "count"
                    ),
                    0,
                ),

            "catalog_start":
                (
                    candidate[
                        "timestart"
                    ].isoformat()
                    if pd.notna(
                        candidate.get(
                            "timestart"
                        )
                    )
                    else None
                ),

            "catalog_end":
                (
                    candidate[
                        "timeend"
                    ].isoformat()
                    if pd.notna(
                        candidate.get(
                            "timeend"
                        )
                    )
                    else None
                ),

            "validation_start":
                str(
                    validation_start.date()
                ),

            "validation_end":
                str(
                    validation_end.date()
                ),

            "validation_records":
                int(
                    len(
                        observations
                    )
                ),

            "validation_current_flow":
                current_flow,

            "candidate_number":
                int(
                    index
                )
                + 1,

            "score":
                _safe_float(
                    candidate.get(
                        "score"
                    )
                ),

            "latitude":
                candidate.get(
                    "latitude"
                ),

            "longitude":
                candidate.get(
                    "longitude"
                ),
        }


    return None


# ============================================================
# OBTENER HISTORIAL DE CAUDAL
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


    try:

        info = (
            find_best_caudal_series(
                start,
                end,
            )
        )

    except Exception as exc:

        return (

            empty,

            {
                "status":
                    "error_busqueda",

                "error":
                    str(
                        exc
                    ),
            },
        )


    if info is None:

        return (

            empty,

            {
                "status":
                    "sin_serie",

                "message":
                    (
                        "No fue posible validar una serie "
                        "de caudal del río Paraná."
                    ),
            },
        )


    try:

        result = (
            _query_caudal_series(

                info[
                    "series_id"
                ],

                start,

                end,
            )
        )

    except Exception as exc:

        return (

            empty,

            {
                **info,

                "status":
                    "error_consulta",

                "error":
                    str(
                        exc
                    ),
            },
        )


    if result.empty:

        return (

            empty,

            {
                **info,

                "status":
                    "sin_observaciones",

                "records":
                    0,
            },
        )


    valid = (
        result[
            "caudal_m3s"
        ]
        .dropna()
    )


    metadata = {

        **info,

        "status":
            "ok",

        "records":
            int(
                len(
                    result
                )
            ),

        "first_date":
            result[
                "datetime"
            ].min(),

        "last_date":
            result[
                "datetime"
            ].max(),

        "current_flow":
            (
                float(
                    valid.iloc[-1]
                )
                if not valid.empty
                else None
            ),

        "mean_flow":
            (
                float(
                    valid.mean()
                )
                if not valid.empty
                else None
            ),

        "max_flow":
            (
                float(
                    valid.max()
                )
                if not valid.empty
                else None
            ),

        "min_flow":
            (
                float(
                    valid.min()
                )
                if not valid.empty
                else None
            ),
    }


    return (
        result,
        metadata,
    )


# ============================================================
# PROYECCIÓN DE CAUDAL
# ============================================================

def project_caudal(
    history,
    future_dates,
):

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

        return pd.Series(
            np.nan,
            index=
                range(
                    len(
                        future_dates
                    )
                ),
            dtype=float,
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
            21
        )
    )


    if values.empty:

        return pd.Series(
            np.nan,
            index=
                range(
                    len(
                        future_dates
                    )
                ),
            dtype=float,
        )


    current = float(
        values.iloc[-1]
    )


    # ========================================================
    # TENDENCIA RECIENTE
    # ========================================================

    if len(
        values
    ) >= 5:

        x = np.arange(
            len(
                values
            ),
            dtype=float,
        )


        try:

            slope = float(
                np.polyfit(
                    x,
                    values.to_numpy(
                        dtype=float
                    ),
                    1,
                )[0]
            )

        except Exception:

            slope = 0.0

    else:

        slope = 0.0


    # ========================================================
    # LIMITAR CAMBIO DIARIO
    # ========================================================

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


    # ========================================================
    # PROYECCIÓN AMORTIGUADA
    # ========================================================

    output = []

    value = current


    for step in range(
        1,
        len(
            future_dates
        )
        + 1,
    ):

        damping = np.exp(
            -step
            / 18.0
        )


        daily_change = (
            slope
            * damping
        )


        value = max(

            0.0,

            value
            + daily_change,
        )


        output.append(
            value
        )


    return pd.Series(
        output,
        dtype=float,
    )


# ============================================================
# COMPLETAR FUTURO DE LLUVIA
#
# Open-Meteo suele aportar ~16 días.
# Para horizontes mayores no inventamos lluvia:
# se completa con 0 y queda explícito en metadata.
# ============================================================

def _build_future_rain(
    base_end,
    days,
):

    end_date = pd.to_datetime(
        base_end,
        errors="coerce",
    )


    if pd.isna(
        end_date
    ):

        end_date = (
            pd.Timestamp.today()
            .normalize()
        )


    future_dates = pd.date_range(

        start=
            end_date
            + pd.Timedelta(
                days=1
            ),

        periods=
            int(
                days
            ),

        freq="D",
    )


    result = pd.DataFrame(
        {
            "datetime":
                future_dates,

            "precip_mm":
                0.0,
        }
    )


    try:

        rain = get_rain_forecast(
            min(
                int(
                    days
                ),
                16,
            )
        )

    except Exception:

        rain = pd.DataFrame()


    if (
        not rain.empty
        and "datetime"
        in rain.columns
    ):

        rain = rain.copy()


        rain[
            "datetime"
        ] = _to_datetime_naive(
            rain[
                "datetime"
            ]
        )


        rain[
            "datetime"
        ] = (
            rain[
                "datetime"
            ]
            .dt
            .normalize()
        )


        rain[
            "precip_mm"
        ] = (
            pd.to_numeric(
                rain[
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


        result = result.merge(

            rain[
                [
                    "datetime",
                    "precip_mm",
                ]
            ],

            on=
                "datetime",

            how=
                "left",

            suffixes=(
                "_default",
                "_forecast",
            ),
        )


        result[
            "precip_mm"
        ] = (
            result[
                "precip_mm_forecast"
            ]
            .fillna(
                result[
                    "precip_mm_default"
                ]
            )
        )


        result = result[
            [
                "datetime",
                "precip_mm",
            ]
        ]


    return result


# ============================================================
# FUNCIÓN PRINCIPAL
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


    # ========================================================
    # LLUVIA HISTÓRICA
    # ========================================================

    try:

        rain_history = (
            get_rain_history(
                start,
                end,
            )
        )

    except Exception:

        rain_history = pd.DataFrame(
            columns=[
                "datetime",
                "precip_mm",
            ]
        )


    # ========================================================
    # CAUDAL HISTÓRICO
    # ========================================================

    caudal_history = pd.DataFrame(
        columns=[
            "datetime",
            "caudal_m3s",
        ]
    )


    caudal_meta = {
        "status":
            "sin_datos",
    }


    try:

        (
            caudal_history,
            caudal_meta,
        ) = get_caudal_history(
            start,
            end,
        )

    except Exception as exc:

        caudal_meta = {

            "status":
                "error",

            "error":
                str(
                    exc
                ),
        }


    # ========================================================
    # HISTORY
    # ========================================================

    history_dates = pd.date_range(

        start=
            pd.to_datetime(
                start
            ),

        end=
            pd.to_datetime(
                end
            ),

        freq="D",
    )


    history = pd.DataFrame(
        {
            "datetime":
                history_dates
        }
    )


    if not rain_history.empty:

        history = history.merge(

            rain_history,

            on=
                "datetime",

            how=
                "left",
        )


    if (
        not caudal_history.empty
    ):

        history = history.merge(

            caudal_history,

            on=
                "datetime",

            how=
                "left",
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


    # --------------------------------------------------------
    # Interpolación limitada de caudal
    # --------------------------------------------------------

    if (
        history[
            "caudal_m3s"
        ]
        .notna()
        .sum()
        >= 5
    ):

        history[
            "caudal_m3s"
        ] = (
            history[
                "caudal_m3s"
            ]
            .interpolate(
                limit=5,
                limit_direction=
                    "both",
            )
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
        _build_future_rain(
            end,
            forecast_days,
        )
    )


    # ========================================================
    # PROYECCIÓN DE CAUDAL
    # ========================================================

    future[
        "caudal_m3s"
    ] = project_caudal(

        caudal_history,

        future[
            "datetime"
        ],
    )


    # ========================================================
    # METADATA
    # ========================================================

    valid_flow = (
        history[
            "caudal_m3s"
        ]
        .dropna()
    )


    meta = {

        "version":
            VERSION,

        # ----------------------------------------------------
        # LLUVIA
        # ----------------------------------------------------

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
                    ].max()
                )
                if not history.empty
                else None
            ),

        "rain_forecast_real_days":
            min(
                forecast_days,
                16,
            ),

        "rain_extended_days_assumption":
            (
                "0 mm después del horizonte "
                "disponible de Open-Meteo"
            ),

        # ----------------------------------------------------
        # CAUDAL
        # ----------------------------------------------------

        "caudal_source":
            "INA A5",

        "caudal_series":
            caudal_meta,

        "caudal_status":
            (
                caudal_meta.get(
                    "status"
                )
                if isinstance(
                    caudal_meta,
                    dict,
                )
                else None
            ),

        "caudal_records":
            int(
                len(
                    caudal_history
                )
            ),

        "uses_caudal":
            bool(
                not valid_flow.empty
            ),

        "current_flow_m3s":
            (
                float(
                    valid_flow.iloc[-1]
                )
                if not valid_flow.empty
                else None
            ),

        "historical_flow_max_m3s":
            (
                float(
                    valid_flow.max()
                )
                if not valid_flow.empty
                else None
            ),

        "historical_flow_min_m3s":
            (
                float(
                    valid_flow.min()
                )
                if not valid_flow.empty
                else None
            ),

        # ----------------------------------------------------
        # GENERAL
        # ----------------------------------------------------

        "historical_start":
            str(
                pd.to_datetime(
                    start
                ).date()
            ),

        "historical_end":
            str(
                pd.to_datetime(
                    end
                ).date()
            ),

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

    result = {

        "version":
            VERSION,

        "ina_catalog_url":
            INA_SERIES_GEOJSON_URL,

        "ina_observations_url":
            INA_OBSERVATIONS_URL,

        "caudal_var_id":
            VAR_ID_CAUDAL,

        "requested_start":
            str(
                start
            ),

        "requested_end":
            str(
                end
            ),
    }


    # ========================================================
    # CATÁLOGO
    # ========================================================

    try:

        catalog = (
            _get_ina_catalog()
        )


        result[
            "catalog_records"
        ] = int(
            len(
                catalog
            )
        )


        if (
            not catalog.empty
            and "var_id"
            in catalog.columns
        ):

            result[
                "catalog_caudal_records"
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
        ] = str(
            exc
        )


    # ========================================================
    # CANDIDATOS
    # ========================================================

    try:

        candidates = (
            _find_caudal_candidates(
                start,
                end,
            )
        )


        result[
            "candidate_count"
        ] = int(
            len(
                candidates
            )
        )


        if not candidates.empty:

            preview_cols = [
                c
                for c in [
                    "series_id",
                    "station",
                    "series_name",
                    "rio",
                    "proc_id",
                    "count",
                    "timestart",
                    "timeend",
                    "score",
                    "station_score",
                    "parana_score",
                    "overlap",
                ]
                if c
                in candidates.columns
            ]


            result[
                "top_candidates"
            ] = (
                candidates[
                    preview_cols
                ]
                .head(
                    15
                )
                .astype(
                    str
                )
                .to_dict(
                    orient="records"
                )
            )


    except Exception as exc:

        result[
            "candidate_error"
        ] = str(
            exc
        )


    # ========================================================
    # SERIE ELEGIDA
    # ========================================================

    try:

        selected = (
            find_best_caudal_series(
                start,
                end,
            )
        )


        result[
            "selected_series"
        ] = selected


    except Exception as exc:

        result[
            "selected_series_error"
        ] = str(
            exc
        )


    # ========================================================
    # DATOS
    # ========================================================

    try:

        (
            flow,
            flow_meta,
        ) = get_caudal_history(
            start,
            end,
        )


        result[
            "flow_records"
        ] = int(
            len(
                flow
            )
        )


        result[
            "flow_meta"
        ] = flow_meta


        if not flow.empty:

            result[
                "first_flow_date"
            ] = str(
                flow[
                    "datetime"
                ].min()
            )


            result[
                "last_flow_date"
            ] = str(
                flow[
                    "datetime"
                ].max()
            )


            result[
                "current_flow_m3s"
            ] = float(
                flow[
                    "caudal_m3s"
                ]
                .dropna()
                .iloc[-1]
            )


    except Exception as exc:

        result[
            "flow_error"
        ] = str(
            exc
        )


    return result
