# ============================================================
# PARANÁ · SAN NICOLÁS
# src/upstream.py
# V11.9.1 COMPLETO
#
# CAMBIOS V11.9.1
# ------------------------------------------------------------
# - Corrige conflicto de timezone:
#       datetime64[us]
#       vs
#       datetime64[us, UTC]
#
# - Todas las fechas entregadas por este módulo quedan
#   normalizadas SIN timezone.
#
# - Mantiene:
#       catálogo A5 GeoJSON
#       detección automática series
#       validación por getObservaciones
#       historial ampliado Corrientes
#       diagnóstico por estación
# ============================================================


import unicodedata

from functools import lru_cache

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURACIÓN INA
# ============================================================

A5_BASE_URL = (
    "https://alerta.ina.gob.ar/a5"
)

A5_SERIES_GEOJSON_URL = (
    A5_BASE_URL
    + "/obs/puntual/series"
)

A5_OBSERVATIONS_URL = (
    A5_BASE_URL
    + "/getObservaciones"
)


# ============================================================
# VARIABLE NIVEL
# ============================================================

VAR_ID_NIVEL = 2


# ============================================================
# HISTORIAL CORRIENTES
# ============================================================

DEFAULT_HISTORY_FLOOR = "1900-01-01"


# ============================================================
# BLOQUES HISTÓRICOS
# ============================================================

HISTORY_BLOCK_YEARS = 5


# ============================================================
# HTTP
# ============================================================

REQUEST_TIMEOUT = 45


# ============================================================
# ESTACIONES
# ============================================================

STATIONS = {
    "Corrientes": {
        "column":
            "nivel_corrientes",

        "aliases": [
            "Corrientes",
        ],
    },

    "Goya": {
        "column":
            "nivel_goya",

        "aliases": [
            "Goya",
        ],
    },

    "La Paz": {
        "column":
            "nivel_la_paz",

        "aliases": [
            "La Paz",
            "La Paz Entre Rios",
            "La Paz Entre Ríos",
        ],
    },

    "Paraná": {
        "column":
            "nivel_parana",

        "aliases": [
            "Paraná",
            "Parana",
        ],
    },

    "Diamante": {
        "column":
            "nivel_diamante",

        "aliases": [
            "Diamante",
        ],
    },

    "Rosario": {
        "column":
            "nivel_rosario",

        "aliases": [
            "Rosario",
        ],
    },

    "Villa Constitución": {
        "column":
            "nivel_villa_constitucion",

        "aliases": [
            "Villa Constitución",
            "Villa Constitucion",
        ],
    },
}


# ============================================================
# SESIÓN HTTP
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
            "Parana-San-Nicolas-Hydrology/11.9.1",

        "Accept":
            "application/json",
    }
)


# ============================================================
# UTILIDADES
# ============================================================

def normalizar_texto(
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
        char
        for char in text
        if unicodedata.category(
            char
        ) != "Mn"
    )

    text = (
        text
        .lower()
        .replace("–", "-")
        .replace("—", "-")
        .replace("_", " ")
    )

    text = " ".join(
        text.split()
    )

    return text


def normalizar_fecha(
    value,
):

    value = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(
        value
    ):
        return None

    return value.strftime(
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


def _safe_datetime(
    value,
):

    result = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(
        result
    ):
        return pd.NaT

    return result


def _to_naive_datetime_series(
    series,
):

    # --------------------------------------------------------
    # CONVERSIÓN UNIFICADA
    #
    # Siempre:
    # 1. interpreta la fecha
    # 2. normaliza temporalmente a UTC
    # 3. elimina timezone
    #
    # Resultado:
    # datetime64[ns]
    # --------------------------------------------------------

    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    ).dt.tz_localize(
        None
    )


# ============================================================
# CATÁLOGO GEOJSON
# ============================================================

@lru_cache(
    maxsize=1
)
def descargar_catalogo_geojson():

    response = SESSION.get(
        A5_SERIES_GEOJSON_URL,
        params={
            "format":
                "geojson",
        },
        timeout=
            REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            "El catálogo INA no devolvió "
            "un objeto GeoJSON válido."
        )

    features = data.get(
        "features",
        [],
    )

    if not isinstance(
        features,
        list,
    ):

        raise ValueError(
            "El catálogo INA no contiene "
            "la lista 'features'."
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

        if row.get(
            "series_id"
        ) is None:

            row[
                "series_id"
            ] = feature.get(
                "id"
            )

        geometry = feature.get(
            "geometry",
            {},
        )

        if isinstance(
            geometry,
            dict,
        ):

            coordinates = geometry.get(
                "coordinates"
            )

            if (
                isinstance(
                    coordinates,
                    list,
                )
                and len(
                    coordinates
                ) >= 2
            ):

                row[
                    "longitude"
                ] = coordinates[0]

                row[
                    "latitude"
                ] = coordinates[1]

        rows.append(
            row
        )

    catalog = pd.DataFrame(
        rows
    )

    if catalog.empty:

        raise ValueError(
            "El catálogo INA está vacío."
        )

    if "series_id" in catalog.columns:

        catalog[
            "series_id"
        ] = pd.to_numeric(
            catalog[
                "series_id"
            ],
            errors="coerce",
        )

    if "var_id" in catalog.columns:

        catalog[
            "var_id"
        ] = pd.to_numeric(
            catalog[
                "var_id"
            ],
            errors="coerce",
        )

    if "proc_id" in catalog.columns:

        catalog[
            "proc_id"
        ] = pd.to_numeric(
            catalog[
                "proc_id"
            ],
            errors="coerce",
        )

    if "count" in catalog.columns:

        catalog[
            "count"
        ] = pd.to_numeric(
            catalog[
                "count"
            ],
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

    if "nombre" in catalog.columns:

        catalog[
            "_nombre_normalizado"
        ] = (
            catalog[
                "nombre"
            ]
            .fillna("")
            .map(
                normalizar_texto
            )
        )

    else:

        catalog[
            "_nombre_normalizado"
        ] = ""

    return catalog


# ============================================================
# SCORE ESTACIÓN
# ============================================================

def _station_match_score(
    station_name,
    aliases,
):

    station_norm = normalizar_texto(
        station_name
    )

    if not station_norm:
        return 0

    best_score = 0

    for alias in aliases:

        alias_norm = normalizar_texto(
            alias
        )

        if not alias_norm:
            continue

        if (
            station_norm
            == alias_norm
        ):

            score = 1000

        elif station_norm.startswith(
            alias_norm
            + " "
        ):

            score = 900

        elif station_norm.startswith(
            alias_norm
            + "-"
        ):

            score = 900

        elif station_norm.endswith(
            " "
            + alias_norm
        ):

            score = 850

        elif (
            " "
            + alias_norm
            + " "
        ) in (
            " "
            + station_norm
            + " "
        ):

            score = 800

        elif alias_norm in station_norm:

            score = 700

        else:

            score = 0

        best_score = max(
            best_score,
            score,
        )

    return best_score


# ============================================================
# PENALIZAR SERIES NO HIDROMÉTRICAS
# ============================================================

def _penalizacion_nombre(
    station_name,
):

    name = normalizar_texto(
        station_name
    )

    penalty = 0

    forbidden_terms = [
        "rmet",
        "meteorologica",
        "meteorologico",
        "precipitacion",
        "inta",
        "escuela",
        "agrometeorologica",
        "aeropuerto",
    ]

    for term in forbidden_terms:

        if term in name:

            penalty -= 500

    return penalty


# ============================================================
# BUSCAR CANDIDATOS
# ============================================================

def buscar_candidatos_estacion(
    catalog,
    station,
):

    config = STATIONS.get(
        station
    )

    if config is None:

        return pd.DataFrame()

    aliases = config[
        "aliases"
    ]

    if (
        catalog is None
        or not isinstance(
            catalog,
            pd.DataFrame,
        )
        or catalog.empty
    ):

        return pd.DataFrame()

    x = catalog.copy()

    # --------------------------------------------------------
    # NIVEL
    # --------------------------------------------------------

    if "var_id" in x.columns:

        x = x[
            x["var_id"]
            == VAR_ID_NIVEL
        ].copy()

    if x.empty:

        return x

    # --------------------------------------------------------
    # SCORE NOMBRE
    # --------------------------------------------------------

    if "nombre" not in x.columns:

        return pd.DataFrame()

    x[
        "_station_score"
    ] = x[
        "nombre"
    ].fillna(
        ""
    ).apply(
        lambda value:
            _station_match_score(
                value,
                aliases,
            )
    )

    x[
        "_name_penalty"
    ] = x[
        "nombre"
    ].fillna(
        ""
    ).apply(
        _penalizacion_nombre
    )

    x[
        "_score"
    ] = (
        x[
            "_station_score"
        ]
        + x[
            "_name_penalty"
        ]
    )

    x = x[
        x["_station_score"]
        >= 700
    ].copy()

    if x.empty:

        return x

    # --------------------------------------------------------
    # PROCEDIMIENTO
    # --------------------------------------------------------

    x[
        "_proc_score"
    ] = 0

    if "proc_id" in x.columns:

        x.loc[
            x["proc_id"] == 1,
            "_proc_score",
        ] = 100

        x.loc[
            x["proc_id"] == 2,
            "_proc_score",
        ] = 50

    # --------------------------------------------------------
    # DISPONIBILIDAD
    # --------------------------------------------------------

    x[
        "_availability_score"
    ] = 0

    if (
        "data_availability"
        in x.columns
    ):

        availability = (
            x[
                "data_availability"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
        )

        x.loc[
            availability == "RT",
            "_availability_score",
        ] = 60

        x.loc[
            availability == "NRT",
            "_availability_score",
        ] = 50

        x.loc[
            availability == "H",
            "_availability_score",
        ] = 30

        x.loc[
            availability == "C",
            "_availability_score",
        ] = 10

    # --------------------------------------------------------
    # CANTIDAD
    # --------------------------------------------------------

    if "count" in x.columns:

        count_values = (
            pd.to_numeric(
                x["count"],
                errors="coerce",
            )
            .fillna(0)
        )

        x[
            "_count_score"
        ] = (
            np.log1p(
                count_values
            )
            * 5
        )

    else:

        x[
            "_count_score"
        ] = 0

    # --------------------------------------------------------
    # RECIENCIA
    # --------------------------------------------------------

    if "timeend" in x.columns:

        now = pd.Timestamp.now(
            tz="UTC"
        )

        age_days = (
            now
            - x["timeend"]
        ).dt.total_seconds() / 86400

        age_days = (
            age_days
            .fillna(
                99999
            )
        )

        x[
            "_recent_score"
        ] = np.where(
            age_days <= 7,
            80,
            np.where(
                age_days <= 30,
                60,
                np.where(
                    age_days <= 365,
                    30,
                    0,
                ),
            ),
        )

    else:

        x[
            "_recent_score"
        ] = 0

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    x[
        "_total_score"
    ] = (
        x[
            "_score"
        ]
        + x[
            "_proc_score"
        ]
        + x[
            "_availability_score"
        ]
        + x[
            "_count_score"
        ]
        + x[
            "_recent_score"
        ]
    )

    sort_columns = [
        "_total_score"
    ]

    ascending = [
        False
    ]

    if "count" in x.columns:

        sort_columns.append(
            "count"
        )

        ascending.append(
            False
        )

    x = x.sort_values(
        sort_columns,
        ascending=
            ascending,
        na_position=
            "last",
    )

    return x.reset_index(
        drop=True
    )


# ============================================================
# BUSCAR OBSERVACIONES EN JSON
# ============================================================

def _buscar_lista_observaciones(
    data,
):

    if isinstance(
        data,
        list,
    ):

        if len(data) == 0:

            return []

        if all(
            isinstance(
                item,
                dict,
            )
            for item in data
        ):

            sample_keys = set()

            for item in data[:5]:

                sample_keys.update(
                    item.keys()
                )

            observation_keys = {
                "timestart",
                "time",
                "fecha",
                "datetime",
                "valor",
                "value",
                "obs_date",
            }

            if (
                sample_keys
                & observation_keys
            ):

                return data

        for item in data:

            result = (
                _buscar_lista_observaciones(
                    item
                )
            )

            if (
                isinstance(
                    result,
                    list,
                )
                and len(
                    result
                ) > 0
            ):

                return result

        return []

    if isinstance(
        data,
        dict,
    ):

        preferred_keys = [
            "observaciones",
            "observations",
            "data",
            "datos",
            "values",
            "records",
            "result",
        ]

        for key in preferred_keys:

            if key in data:

                result = (
                    _buscar_lista_observaciones(
                        data[key]
                    )
                )

                if isinstance(
                    result,
                    list,
                ):

                    return result

        for value in data.values():

            result = (
                _buscar_lista_observaciones(
                    value
                )
            )

            if (
                isinstance(
                    result,
                    list,
                )
                and len(
                    result
                ) > 0
            ):

                return result

    return []


# ============================================================
# NORMALIZAR OBSERVACIONES
# ============================================================

def normalizar_observaciones(
    data,
):

    records = (
        _buscar_lista_observaciones(
            data
        )
    )

    if not records:

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )

    rows = []

    datetime_fields = [
        "timestart",
        "timestamp",
        "datetime",
        "date",
        "fecha",
        "time",
        "obs_date",
        "fechaHora",
        "fecha_hora",
    ]

    value_fields = [
        "valor",
        "value",
        "nivel",
        "obs_value",
        "valor_num",
    ]

    for record in records:

        if not isinstance(
            record,
            dict,
        ):

            continue

        dt = None

        for field in datetime_fields:

            if (
                field in record
                and record[
                    field
                ] is not None
            ):

                dt = record[
                    field
                ]

                break

        value = None

        for field in value_fields:

            if (
                field in record
                and record[
                    field
                ] is not None
            ):

                value = record[
                    field
                ]

                break

        if (
            value is None
            and isinstance(
                record.get(
                    "observation"
                ),
                dict,
            )
        ):

            obs = record[
                "observation"
            ]

            for field in value_fields:

                if field in obs:

                    value = obs[
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

        if pd.isna(
            dt
        ):

            continue

        if pd.isna(
            value
        ):

            continue

        value = float(
            value
        )

        if (
            value < -5
            or value > 20
        ):

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

    result = (
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

    return result


# ============================================================
# CONSULTAR A5
# ============================================================

def consultar_a5(
    series_id,
    start,
    end,
):

    series_id = _safe_int(
        series_id
    )

    if series_id is None:

        raise ValueError(
            "series_id inválido."
        )

    start = normalizar_fecha(
        start
    )

    end = normalizar_fecha(
        end
    )

    if (
        start is None
        or end is None
    ):

        raise ValueError(
            "Rango de fechas inválido."
        )

    response = SESSION.get(
        A5_OBSERVATIONS_URL,
        params={
            "tipo":
                "puntual",

            "series_id":
                series_id,

            "timestart":
                start,

            "timeend":
                end,
        },
        timeout=
            REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    return normalizar_observaciones(
        data
    )


# ============================================================
# VALIDAR SERIE
# ============================================================

def validar_serie(
    row,
    requested_start,
    requested_end,
):

    series_id = _safe_int(
        row.get(
            "series_id"
        )
    )

    if series_id is None:

        return {
            "ok":
                False,

            "records":
                0,

            "error":
                "series_id inválido",
        }

    catalog_start = (
        _safe_datetime(
            row.get(
                "timestart"
            )
        )
    )

    catalog_end = (
        _safe_datetime(
            row.get(
                "timeend"
            )
        )
    )

    req_end = pd.to_datetime(
        requested_end,
        errors="coerce",
        utc=True,
    )

    if pd.isna(
        req_end
    ):

        req_end = pd.Timestamp.now(
            tz="UTC"
        )

    validation_end = (
        catalog_end
        if not pd.isna(
            catalog_end
        )
        else req_end
    )

    validation_end = min(
        validation_end,
        req_end,
    )

    validation_start = (
        validation_end
        - pd.Timedelta(
            days=120
        )
    )

    if (
        not pd.isna(
            catalog_start
        )
    ):

        validation_start = max(
            validation_start,
            catalog_start,
        )

    try:

        df = consultar_a5(
            series_id=
                series_id,
            start=
                validation_start,
            end=
                validation_end,
        )

    except Exception as exc:

        return {
            "ok":
                False,

            "records":
                0,

            "error":
                str(exc),
        }

    return {
        "ok":
            not df.empty,

        "records":
            len(df),

        "first_date":
            (
                df[
                    "datetime"
                ].min()
                if not df.empty
                else None
            ),

        "last_date":
            (
                df[
                    "datetime"
                ].max()
                if not df.empty
                else None
            ),

        "error":
            None,
    }


# ============================================================
# SELECCIONAR SERIE
# ============================================================

def seleccionar_serie(
    catalog,
    station,
    requested_start,
    requested_end,
):

    candidates = (
        buscar_candidatos_estacion(
            catalog,
            station,
        )
    )

    if candidates.empty:

        return (
            None,
            {
                "station":
                    station,

                "status":
                    "sin_candidatos",

                "candidates":
                    0,

                "message":
                    "No se encontró serie de nivel "
                    "compatible en catálogo INA.",
            },
        )

    attempts = []

    for index, row in (
        candidates
        .head(12)
        .iterrows()
    ):

        result = validar_serie(
            row=
                row,
            requested_start=
                requested_start,
            requested_end=
                requested_end,
        )

        attempt = {
            "series_id":
                _safe_int(
                    row.get(
                        "series_id"
                    )
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

            "catalog_count":
                _safe_int(
                    row.get(
                        "count"
                    ),
                    0,
                ),

            "catalog_start":
                row.get(
                    "timestart"
                ),

            "catalog_end":
                row.get(
                    "timeend"
                ),

            "validation_records":
                result.get(
                    "records",
                    0,
                ),

            "validation_ok":
                result.get(
                    "ok",
                    False,
                ),

            "error":
                result.get(
                    "error"
                ),
        }

        attempts.append(
            attempt
        )

        if result.get(
            "ok",
            False,
        ):

            return (
                row,
                {
                    "station":
                        station,

                    "status":
                        "ok",

                    "candidates":
                        len(
                            candidates
                        ),

                    "selected_candidate":
                        index + 1,

                    "attempts":
                        attempts,
                },
            )

    return (
        None,
        {
            "station":
                station,

            "status":
                "sin_serie_validada",

            "candidates":
                len(
                    candidates
                ),

            "attempts":
                attempts,

            "message":
                "Se encontraron candidatos pero ninguno "
                "devolvió observaciones válidas.",
        },
    )


# ============================================================
# CONSULTAR EN BLOQUES
# ============================================================

def consultar_a5_en_bloques(
    series_id,
    start,
    end,
    block_years=
        HISTORY_BLOCK_YEARS,
):

    start = pd.to_datetime(
        start,
        errors="coerce",
        utc=True,
    )

    end = pd.to_datetime(
        end,
        errors="coerce",
        utc=True,
    )

    if (
        pd.isna(
            start
        )
        or pd.isna(
            end
        )
        or start > end
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )

    pieces = []

    cursor = start

    while cursor <= end:

        block_end = (
            cursor
            + pd.DateOffset(
                years=
                    block_years
            )
            - pd.Timedelta(
                days=1
            )
        )

        block_end = min(
            block_end,
            end,
        )

        try:

            part = consultar_a5(
                series_id=
                    series_id,
                start=
                    cursor,
                end=
                    block_end,
            )

            if not part.empty:

                pieces.append(
                    part
                )

        except Exception:

            pass

        cursor = (
            block_end
            + pd.Timedelta(
                days=1
            )
        )

    if not pieces:

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    result = (
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

    return result


# ============================================================
# CONVERTIR A DIARIO
#
# CORRECCIÓN PRINCIPAL V11.9.1:
# datetime queda SIN timezone.
# ============================================================

def convertir_diario(
    df,
    output_column,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                output_column,
            ]
        )

    if (
        "datetime"
        not in df.columns
        or "value"
        not in df.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                output_column,
            ]
        )

    x = df[
        [
            "datetime",
            "value",
        ]
    ].copy()

    # ========================================================
    # IMPORTANTE
    # quitar timezone antes de entregar datos al modelo
    # ========================================================

    x["datetime"] = (
        _to_naive_datetime_series(
            x["datetime"]
        )
    )

    x["value"] = pd.to_numeric(
        x["value"],
        errors="coerce",
    )

    x = x.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )

    if x.empty:

        return pd.DataFrame(
            columns=[
                "datetime",
                output_column,
            ]
        )

    x["datetime"] = (
        x["datetime"]
        .dt.floor(
            "D"
        )
    )

    daily = (
        x
        .groupby(
            "datetime",
            as_index=False,
        )["value"]
        .median()
        .rename(
            columns={
                "value":
                    output_column
            }
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # SEGUNDA PROTECCIÓN
    # --------------------------------------------------------

    daily[
        "datetime"
    ] = pd.to_datetime(
        daily[
            "datetime"
        ],
        errors="coerce",
    )

    return daily


# ============================================================
# OBTENER ESTACIÓN
# ============================================================

def obtener_estacion(
    station,
    catalog,
    requested_start,
    requested_end,
):

    config = STATIONS[
        station
    ]

    output_column = config[
        "column"
    ]

    selected_row, selection_meta = (
        seleccionar_serie(
            catalog=
                catalog,
            station=
                station,
            requested_start=
                requested_start,
            requested_end=
                requested_end,
        )
    )

    if selected_row is None:

        metadata = {
            "station":
                station,

            "column":
                output_column,

            "status":
                selection_meta.get(
                    "status",
                    "sin_serie",
                ),

            "series_id":
                None,

            "series_name":
                None,

            "records":
                0,

            "first_date":
                None,

            "last_date":
                None,

            "catalog_candidates":
                selection_meta.get(
                    "candidates",
                    0,
                ),

            "selection":
                selection_meta,
        }

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    output_column,
                ]
            ),
            metadata,
        )

    series_id = _safe_int(
        selected_row.get(
            "series_id"
        )
    )

    catalog_start = (
        _safe_datetime(
            selected_row.get(
                "timestart"
            )
        )
    )

    catalog_end = (
        _safe_datetime(
            selected_row.get(
                "timeend"
            )
        )
    )

    requested_start_dt = (
        pd.to_datetime(
            requested_start,
            errors="coerce",
            utc=True,
        )
    )

    requested_end_dt = (
        pd.to_datetime(
            requested_end,
            errors="coerce",
            utc=True,
        )
    )

    # --------------------------------------------------------
    # RANGO
    # --------------------------------------------------------

    if station == "Corrientes":

        history_floor = pd.Timestamp(
            DEFAULT_HISTORY_FLOOR,
            tz="UTC",
        )

        if not pd.isna(
            catalog_start
        ):

            query_start = max(
                catalog_start,
                history_floor,
            )

        else:

            query_start = history_floor

        if not pd.isna(
            catalog_end
        ):

            query_end = min(
                catalog_end,
                requested_end_dt,
            )

        else:

            query_end = (
                requested_end_dt
            )

    else:

        query_start = (
            requested_start_dt
        )

        query_end = (
            requested_end_dt
        )

        if not pd.isna(
            catalog_start
        ):

            query_start = max(
                query_start,
                catalog_start,
            )

        if not pd.isna(
            catalog_end
        ):

            query_end = min(
                query_end,
                catalog_end,
            )

    if (
        pd.isna(
            query_start
        )
        or pd.isna(
            query_end
        )
        or query_start > query_end
    ):

        metadata = {
            "station":
                station,

            "column":
                output_column,

            "status":
                "fuera_de_periodo",

            "series_id":
                series_id,

            "series_name":
                selected_row.get(
                    "nombre"
                ),

            "records":
                0,

            "first_date":
                None,

            "last_date":
                None,

            "catalog_start":
                (
                    catalog_start.isoformat()
                    if not pd.isna(
                        catalog_start
                    )
                    else None
                ),

            "catalog_end":
                (
                    catalog_end.isoformat()
                    if not pd.isna(
                        catalog_end
                    )
                    else None
                ),
        }

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    output_column,
                ]
            ),
            metadata,
        )

    # --------------------------------------------------------
    # DESCARGA
    # --------------------------------------------------------

    try:

        if station == "Corrientes":

            raw = (
                consultar_a5_en_bloques(
                    series_id=
                        series_id,
                    start=
                        query_start,
                    end=
                        query_end,
                    block_years=
                        HISTORY_BLOCK_YEARS,
                )
            )

        else:

            raw = consultar_a5(
                series_id=
                    series_id,
                start=
                    query_start,
                end=
                    query_end,
            )

    except Exception as exc:

        metadata = {
            "station":
                station,

            "column":
                output_column,

            "status":
                "error_consulta",

            "series_id":
                series_id,

            "series_name":
                selected_row.get(
                    "nombre"
                ),

            "records":
                0,

            "first_date":
                None,

            "last_date":
                None,

            "error":
                str(exc),
        }

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    output_column,
                ]
            ),
            metadata,
        )

    daily = convertir_diario(
        raw,
        output_column,
    )

    if not daily.empty:

        first_date = (
            daily[
                "datetime"
            ].min()
        )

        last_date = (
            daily[
                "datetime"
            ].max()
        )

    else:

        first_date = None
        last_date = None

    metadata = {
        "station":
            station,

        "column":
            output_column,

        "status":
            (
                "ok"
                if not daily.empty
                else "sin_observaciones"
            ),

        "series_id":
            series_id,

        "series_name":
            selected_row.get(
                "nombre"
            ),

        "station_id":
            _safe_int(
                selected_row.get(
                    "estacion_id"
                )
            ),

        "var_id":
            _safe_int(
                selected_row.get(
                    "var_id"
                )
            ),

        "variable":
            selected_row.get(
                "var_nombre"
            ),

        "proc_id":
            _safe_int(
                selected_row.get(
                    "proc_id"
                )
            ),

        "unit_id":
            _safe_int(
                selected_row.get(
                    "unit_id"
                )
            ),

        "source":
            selected_row.get(
                "fuente"
            ),

        "availability":
            selected_row.get(
                "data_availability"
            ),

        "catalog_count":
            _safe_int(
                selected_row.get(
                    "count"
                ),
                0,
            ),

        "records":
            int(
                len(
                    daily
                )
            ),

        "first_date":
            (
                pd.Timestamp(
                    first_date
                ).strftime(
                    "%Y-%m-%d"
                )
                if first_date is not None
                else None
            ),

        "last_date":
            (
                pd.Timestamp(
                    last_date
                ).strftime(
                    "%Y-%m-%d"
                )
                if last_date is not None
                else None
            ),

        "catalog_start":
            (
                catalog_start.strftime(
                    "%Y-%m-%d"
                )
                if not pd.isna(
                    catalog_start
                )
                else None
            ),

        "catalog_end":
            (
                catalog_end.strftime(
                    "%Y-%m-%d"
                )
                if not pd.isna(
                    catalog_end
                )
                else None
            ),

        "query_start":
            pd.Timestamp(
                query_start
            ).strftime(
                "%Y-%m-%d"
            ),

        "query_end":
            pd.Timestamp(
                query_end
            ).strftime(
                "%Y-%m-%d"
            ),

        "catalog_candidates":
            selection_meta.get(
                "candidates",
                0,
            ),

        "selection":
            selection_meta,
    }

    return (
        daily,
        metadata,
    )


# ============================================================
# COMBINAR ESTACIONES
#
# SEGUNDA CORRECCIÓN PRINCIPAL V11.9.1:
# blindaje completo de datetime antes de cada merge.
# ============================================================

def _merge_station_frames(
    frames,
):

    normalized_frames = []

    for frame in frames:

        if (
            isinstance(
                frame,
                pd.DataFrame,
            )
            and not frame.empty
            and "datetime"
            in frame.columns
        ):

            temp = frame.copy()

            temp[
                "datetime"
            ] = _to_naive_datetime_series(
                temp[
                    "datetime"
                ]
            )

            temp = temp.dropna(
                subset=[
                    "datetime"
                ]
            )

            normalized_frames.append(
                temp
            )

        else:

            normalized_frames.append(
                frame
            )

    frames = normalized_frames

    valid_frames = [
        frame
        for frame in frames
        if (
            isinstance(
                frame,
                pd.DataFrame,
            )
            and not frame.empty
            and "datetime"
            in frame.columns
        )
    ]

    if not valid_frames:

        result = pd.DataFrame(
            columns=[
                "datetime"
            ]
        )

    else:

        result = (
            valid_frames[0]
            .copy()
        )

        # ----------------------------------------------------
        # BLINDAJE PRIMER FRAME
        # ----------------------------------------------------

        result[
            "datetime"
        ] = _to_naive_datetime_series(
            result[
                "datetime"
            ]
        )

        for frame in valid_frames[1:]:

            temp = frame.copy()

            temp[
                "datetime"
            ] = _to_naive_datetime_series(
                temp[
                    "datetime"
                ]
            )

            result = pd.merge(
                result,
                temp,
                on=
                    "datetime",
                how=
                    "outer",
            )

        result = (
            result
            .sort_values(
                "datetime"
            )
            .drop_duplicates(
                subset=[
                    "datetime"
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # GARANTIZAR COLUMNAS
    # --------------------------------------------------------

    for config in STATIONS.values():

        col = config[
            "column"
        ]

        if col not in result.columns:

            result[col] = np.nan

    desired_columns = [
        "datetime",
        "nivel_corrientes",
        "nivel_goya",
        "nivel_la_paz",
        "nivel_parana",
        "nivel_diamante",
        "nivel_rosario",
        "nivel_villa_constitucion",
    ]

    for col in desired_columns:

        if col not in result.columns:

            result[col] = np.nan

    result = result[
        desired_columns
    ]

    # --------------------------------------------------------
    # GARANTÍA FINAL
    #
    # Ningún timezone sale de upstream.py
    # --------------------------------------------------------

    if (
        "datetime"
        in result.columns
    ):

        result[
            "datetime"
        ] = _to_naive_datetime_series(
            result[
                "datetime"
            ]
        )

    return result


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def get_upstream_history(
    start,
    end,
):

    start = normalizar_fecha(
        start
    )

    end = normalizar_fecha(
        end
    )

    if (
        start is None
        or end is None
    ):

        raise ValueError(
            "Fechas inválidas para consulta "
            "de estaciones aguas arriba."
        )

    if (
        pd.Timestamp(
            start
        )
        > pd.Timestamp(
            end
        )
    ):

        raise ValueError(
            "La fecha inicial es posterior "
            "a la fecha final."
        )

    catalog = (
        descargar_catalogo_geojson()
    )

    frames = []

    metadata = {}

    for station in STATIONS.keys():

        try:

            frame, meta = obtener_estacion(
                station=
                    station,
                catalog=
                    catalog,
                requested_start=
                    start,
                requested_end=
                    end,
            )

        except Exception as exc:

            config = STATIONS[
                station
            ]

            frame = pd.DataFrame(
                columns=[
                    "datetime",
                    config[
                        "column"
                    ],
                ]
            )

            meta = {
                "station":
                    station,

                "column":
                    config[
                        "column"
                    ],

                "status":
                    "error",

                "series_id":
                    None,

                "series_name":
                    None,

                "records":
                    0,

                "first_date":
                    None,

                "last_date":
                    None,

                "error":
                    str(exc),
            }

        frames.append(
            frame
        )

        metadata[
            station
        ] = meta

    result = (
        _merge_station_frames(
            frames
        )
    )

    return (
        result,
        metadata,
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
            "V11.9.1",

        "catalog_url":
            A5_SERIES_GEOJSON_URL,

        "observations_url":
            A5_OBSERVATIONS_URL,

        "var_id_nivel":
            VAR_ID_NIVEL,

        "start":
            normalizar_fecha(
                start
            ),

        "end":
            normalizar_fecha(
                end
            ),

        "catalog_records":
            0,

        "stations":
            {},
    }

    try:

        catalog = (
            descargar_catalogo_geojson()
        )

        report[
            "catalog_records"
        ] = len(
            catalog
        )

    except Exception as exc:

        report[
            "catalog_error"
        ] = str(
            exc
        )

        return report

    for station in STATIONS:

        candidates = (
            buscar_candidatos_estacion(
                catalog,
                station,
            )
        )

        candidate_list = []

        if not candidates.empty:

            for _, row in (
                candidates
                .head(10)
                .iterrows()
            ):

                start_value = (
                    row.get(
                        "timestart"
                    )
                )

                end_value = (
                    row.get(
                        "timeend"
                    )
                )

                candidate_list.append(
                    {
                        "series_id":
                            _safe_int(
                                row.get(
                                    "series_id"
                                )
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

                        "records":
                            _safe_int(
                                row.get(
                                    "count"
                                ),
                                0,
                            ),

                        "start":
                            (
                                pd.Timestamp(
                                    start_value
                                ).strftime(
                                    "%Y-%m-%d"
                                )
                                if (
                                    start_value is not None
                                    and not pd.isna(
                                        start_value
                                    )
                                )
                                else None
                            ),

                        "end":
                            (
                                pd.Timestamp(
                                    end_value
                                ).strftime(
                                    "%Y-%m-%d"
                                )
                                if (
                                    end_value is not None
                                    and not pd.isna(
                                        end_value
                                    )
                                )
                                else None
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
            "stations"
        ][station] = {
            "candidate_count":
                len(
                    candidates
                ),

            "candidates":
                candidate_list,
        }

    return report


# ============================================================
# TABLA DIAGNÓSTICA
# ============================================================

def diagnostic_table(
    start,
    end,
):

    try:

        history, metadata = (
            get_upstream_history(
                start,
                end,
            )
        )

    except Exception as exc:

        return pd.DataFrame(
            [
                {
                    "Estación":
                        "GENERAL",

                    "Estado":
                        "ERROR",

                    "Serie":
                        "—",

                    "Nombre INA":
                        "—",

                    "Registros":
                        0,

                    "Desde":
                        "—",

                    "Hasta":
                        "—",

                    "Tipo fecha":
                        "—",

                    "Error":
                        str(exc),
                }
            ]
        )

    rows = []

    datetime_type = (
        str(
            history[
                "datetime"
            ].dtype
        )
        if (
            isinstance(
                history,
                pd.DataFrame,
            )
            and "datetime"
            in history.columns
        )
        else "—"
    )

    for station in STATIONS:

        meta = metadata.get(
            station,
            {}
        )

        rows.append(
            {
                "Estación":
                    station,

                "Estado":
                    meta.get(
                        "status",
                        "sin dato",
                    ),

                "Serie":
                    (
                        meta.get(
                            "series_id"
                        )
                        if meta.get(
                            "series_id"
                        )
                        is not None
                        else "—"
                    ),

                "Nombre INA":
                    (
                        meta.get(
                            "series_name"
                        )
                        or "—"
                    ),

                "Variable":
                    (
                        meta.get(
                            "variable"
                        )
                        or "—"
                    ),

                "Registros":
                    meta.get(
                        "records",
                        0,
                    ),

                "Desde":
                    (
                        meta.get(
                            "first_date"
                        )
                        or "—"
                    ),

                "Hasta":
                    (
                        meta.get(
                            "last_date"
                        )
                        or "—"
                    ),

                "Candidatos":
                    meta.get(
                        "catalog_candidates",
                        0,
                    ),

                "Tipo fecha":
                    datetime_type,

                "Error":
                    (
                        meta.get(
                            "error"
                        )
                        or ""
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )
