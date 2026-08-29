# ============================================================
# PARANÁ · SAN NICOLÁS
# src/upstream.py
# V11.10.1 COMPLETO
#
# NIVELES AGUAS ARRIBA - INA A5
#
# Objetivos:
# ------------------------------------------------------------
# 1. Detectar automáticamente series de NIVEL (var_id = 2)
#    para las estaciones del corredor Paraná.
#
# 2. Validar cada serie antes de utilizarla.
#
# 3. Evitar coincidencias falsas por nombre:
#    - estaciones meteorológicas
#    - INTA
#    - aeropuertos
#    - escuelas
#    - estaciones que no correspondan al corredor
#
# 4. Preferir estaciones asociadas al río Paraná.
#
# 5. Consultar observaciones reales INA A5 antes de aceptar
#    una serie.
#
# 6. Mantener fechas timezone-naive para evitar errores:
#    datetime64[us] vs datetime64[us, UTC]
#
# 7. Mantener historial completo de Corrientes para análisis
#    histórico Corrientes -> San Nicolás.
#
# 8. Entregar diagnóstico:
#    estación
#    series_id
#    nombre
#    río
#    registros
#    última fecha
#    último nivel
#    estado
#
# API PRINCIPAL:
# ------------------------------------------------------------
# get_upstream_history(start, end)
#
# retorna:
#     dataframe, metadata
#
# COLUMNAS:
#     datetime
#     nivel_corrientes
#     nivel_goya
#     nivel_la_paz
#     nivel_parana
#     nivel_diamante
#     nivel_rosario
#     nivel_villa_constitucion
#
# IMPORTANTE:
# San Nicolás continúa siendo serie 36 y debe seguir
# obteniéndose mediante src/ina.py.
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

VERSION = "V11.10.1"


# ============================================================
# INA A5
# ============================================================

A5_BASE_URL = "https://alerta.ina.gob.ar/a5"

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
# CONFIGURACIÓN
# ============================================================

REQUEST_TIMEOUT = 45

DEFAULT_HISTORY_FLOOR = "1900-01-01"

HISTORY_BLOCK_YEARS = 5

VALIDATION_DAYS = 180

MIN_VALID_OBSERVATIONS = 3

LEVEL_MIN_VALID = -5.0

LEVEL_MAX_VALID = 20.0


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
]


# ============================================================
# COLUMNAS
# ============================================================

STATION_COLUMNS = {

    "Corrientes":
        "nivel_corrientes",

    "Goya":
        "nivel_goya",

    "La Paz":
        "nivel_la_paz",

    "Paraná":
        "nivel_parana",

    "Diamante":
        "nivel_diamante",

    "Rosario":
        "nivel_rosario",

    "Villa Constitución":
        "nivel_villa_constitucion",
}


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
        "puerto villa constitucion",
        "puerto villa constitución",
    ],
}


# ============================================================
# TÉRMINOS NO DESEADOS
# ============================================================

BAD_NAME_TERMS = [
    "meteorologica",
    "meteorologico",
    "agrometeorologica",
    "aeropuerto",
    "aerodromo",
    "inta",
    "escuela",
    "pluviometrica",
    "precipitacion",
    "lluvia",
    "temperatura",
]


# ============================================================
# CURSOS SECUNDARIOS / OTROS RÍOS
#
# No significa que esas estaciones sean inválidas en INA.
# Significa que no deben utilizarse como nivel del Paraná
# troncal para este modelo.
# ============================================================

NON_PARANA_TERMS = [
    "arroyo",
    "canal",
    "riacho",
    "laguna",
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
    ).strip().lower()

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


def _to_naive_datetime_series(series):

    return (
        pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
    )


def _safe_timestamp_naive(value):

    dt = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(
        dt
    ):

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
# CATÁLOGO INA A5
# ============================================================

@lru_cache(maxsize=1)
def descargar_catalogo_geojson():

    response = requests.get(

        A5_SERIES_GEOJSON_URL,

        params={
            "format":
                "geojson"
        },

        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    features = payload.get(
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

        properties = feature.get(
            "properties",
            {}
        )

        if not isinstance(
            properties,
            dict,
        ):
            continue

        geometry = feature.get(
            "geometry",
            {}
        )

        coordinates = []

        if isinstance(
            geometry,
            dict,
        ):

            coordinates = (
                geometry.get(
                    "coordinates",
                    []
                )
                or []
            )

        longitude = np.nan
        latitude = np.nan

        if (
            isinstance(
                coordinates,
                list,
            )
            and len(
                coordinates
            ) >= 2
        ):

            longitude = _safe_float(
                coordinates[0]
            )

            latitude = _safe_float(
                coordinates[1]
            )

        rows.append(
            {
                "id":
                    properties.get(
                        "id"
                    ),

                "series_id":
                    _safe_int(
                        properties.get(
                            "series_id"
                        ),
                        0,
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
                    _safe_int(
                        properties.get(
                            "var_id"
                        ),
                        0,
                    ),

                "proc_id":
                    _safe_int(
                        properties.get(
                            "proc_id"
                        ),
                        0,
                    ),

                "unit_id":
                    properties.get(
                        "unit_id"
                    ),

                "var_nombre":
                    properties.get(
                        "var_nombre"
                    ),

                "GeneralCategory":
                    properties.get(
                        "GeneralCategory"
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
                    _safe_int(
                        properties.get(
                            "count"
                        ),
                        0,
                    ),

                "forecast_date":
                    properties.get(
                        "forecast_date"
                    ),

                "data_availability":
                    properties.get(
                        "data_availability"
                    ),

                "fuente":
                    properties.get(
                        "fuente"
                    ),

                "id_externo":
                    properties.get(
                        "id_externo"
                    ),

                "public":
                    properties.get(
                        "public"
                    ),

                "longitude":
                    longitude,

                "latitude":
                    latitude,
            }
        )

    catalog = pd.DataFrame(
        rows
    )

    if catalog.empty:

        raise RuntimeError(
            "El catálogo INA A5 no devolvió series."
        )

    return catalog


# ============================================================
# SCORE DEL NOMBRE
# ============================================================

def _station_name_score(
    station,
    station_name,
):

    name = _normalize_text(
        station_name
    )

    if not name:

        return -1000

    aliases = STATION_ALIASES.get(
        station,
        [
            station
        ],
    )

    score = -1000

    for alias in aliases:

        alias = _normalize_text(
            alias
        )

        if not alias:
            continue

        if name == alias:

            score = max(
                score,
                150,
            )

        elif name.startswith(
            alias
        ):

            score = max(
                score,
                125,
            )

        elif re.search(
            r"\b"
            + re.escape(
                alias
            )
            + r"\b",
            name,
        ):

            score = max(
                score,
                105,
            )

        elif alias in name:

            score = max(
                score,
                75,
            )

    for bad_term in BAD_NAME_TERMS:

        if (
            _normalize_text(
                bad_term
            )
            in name
        ):

            score -= 150

    return score


# ============================================================
# SCORE DEL RÍO
# ============================================================

def _river_score(
    river,
):

    river_name = _normalize_text(
        river
    )

    if not river_name:

        return 0

    score = 0

    if "parana" in river_name:

        score += 90

    for term in NON_PARANA_TERMS:

        term = _normalize_text(
            term
        )

        if term in river_name:

            score -= 120

    return score


# ============================================================
# ¿CORRESPONDE AL PARANÁ?
# ============================================================

def _is_parana_candidate(
    river,
    station_name=None,
):

    river_name = _normalize_text(
        river
    )

    station_name = _normalize_text(
        station_name
    )

    combined = (
        river_name
        + " "
        + station_name
    )

    for term in NON_PARANA_TERMS:

        term = _normalize_text(
            term
        )

        if term in combined:

            return False

    # --------------------------------------------------------
    # Si INA declara explícitamente el río, exigimos Paraná.
    # --------------------------------------------------------

    if river_name:

        return (
            "parana"
            in river_name
        )

    # --------------------------------------------------------
    # Si INA no tiene campo río, no descartamos automáticamente
    # una coincidencia fuerte por nombre. La validación con
    # observaciones reales decidirá después.
    # --------------------------------------------------------

    return True


# ============================================================
# CANDIDATOS DE ESTACIÓN
# ============================================================

def buscar_candidatos_estacion(
    station,
    start=None,
    end=None,
):

    catalog = descargar_catalogo_geojson()

    candidates = catalog[
        catalog[
            "var_id"
        ]
        == VAR_ID_NIVEL
    ].copy()

    if candidates.empty:

        return pd.DataFrame()

    candidates[
        "station_score"
    ] = candidates[
        "nombre"
    ].apply(
        lambda value:
            _station_name_score(
                station,
                value,
            )
    )

    # --------------------------------------------------------
    # Exigir coincidencia de estación.
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
        "river_score"
    ] = candidates[
        "rio"
    ].apply(
        _river_score
    )

    candidates[
        "is_parana_candidate"
    ] = candidates.apply(
        lambda row:
            _is_parana_candidate(
                row.get(
                    "rio"
                ),
                row.get(
                    "nombre"
                ),
            ),
        axis=1,
    )

    candidates[
        "parana_score"
    ] = np.where(
        candidates[
            "is_parana_candidate"
        ],
        50,
        -200,
    )

    # --------------------------------------------------------
    # Disponibilidad temporal
    # --------------------------------------------------------

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
    ] = np.where(
        pd.to_numeric(
            candidates[
                "proc_id"
            ],
            errors="coerce",
        ).fillna(
            0
        ) > 0,
        10,
        0,
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
            30,
            np.where(
                age_days <= 365,
                20,
                np.where(
                    age_days <= 365 * 5,
                    8,
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
            25,
            -100,
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
            "parana_score"
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
# PARSER RECURSIVO DE OBSERVACIONES
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
                    dict,
                    list,
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

        dt = _safe_timestamp_naive(
            record.get(
                key
            )
        )

        if not pd.isna(
            dt
        ):

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
# CONSULTA A5
# ============================================================

def consultar_a5(
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

        A5_OBSERVATIONS_URL,

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

        if pd.isna(
            dt
        ):

            continue

        if not np.isfinite(
            value
        ):

            continue

        # ----------------------------------------------------
        # Filtro de seguridad.
        # No usamos 0-7 aquí porque los ceros de escala
        # hidrométrica son diferentes entre estaciones.
        # --------------------------------------------------------

        if (
            value < LEVEL_MIN_VALID
            or value > LEVEL_MAX_VALID
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

    result[
        "datetime"
    ] = _to_naive_datetime_series(
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
# VENTANA DE VALIDACIÓN
# ============================================================

def _validation_window(
    candidate,
    start,
    end,
):

    request_start = pd.to_datetime(
        start
    ).normalize()

    request_end = pd.to_datetime(
        end
    ).normalize()

    catalog_start = _safe_timestamp_naive(
        candidate.get(
            "timestart"
        )
    )

    catalog_end = _safe_timestamp_naive(
        candidate.get(
            "timeend"
        )
    )

    if pd.isna(
        catalog_start
    ):

        catalog_start = request_start

    else:

        catalog_start = (
            pd.Timestamp(
                catalog_start
            )
            .normalize()
        )

    if pd.isna(
        catalog_end
    ):

        catalog_end = request_end

    else:

        catalog_end = (
            pd.Timestamp(
                catalog_end
            )
            .normalize()
        )

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
                VALIDATION_DAYS
        ),
    )

    return (
        validation_start,
        overlap_end,
    )


# ============================================================
# VALIDACIÓN DE SERIE
# ============================================================

def validar_serie(
    station,
    candidate,
    data,
):

    reasons = []

    valid = True

    if (
        data is None
        or data.empty
    ):

        return {
            "valid":
                False,

            "records":
                0,

            "reasons":
                [
                    "sin_observaciones"
                ],
        }

    values = (
        pd.to_numeric(
            data[
                "value"
            ],
            errors="coerce",
        )
        .dropna()
    )

    if (
        len(
            values
        )
        < MIN_VALID_OBSERVATIONS
    ):

        valid = False

        reasons.append(
            "pocas_observaciones"
        )

    # --------------------------------------------------------
    # Nombre de estación.
    # --------------------------------------------------------

    station_score = (
        _station_name_score(
            station,
            candidate.get(
                "nombre"
            ),
        )
    )

    if station_score <= 0:

        valid = False

        reasons.append(
            "nombre_no_coincide"
        )

    # --------------------------------------------------------
    # Río Paraná.
    # --------------------------------------------------------

    if not _is_parana_candidate(
        candidate.get(
            "rio"
        ),
        candidate.get(
            "nombre"
        ),
    ):

        valid = False

        reasons.append(
            "no_corresponde_parana"
        )

    # --------------------------------------------------------
    # Rango.
    # --------------------------------------------------------

    if not values.empty:

        if (
            values.min()
            < LEVEL_MIN_VALID
            or values.max()
            > LEVEL_MAX_VALID
        ):

            valid = False

            reasons.append(
                "nivel_fuera_rango"
            )

    # --------------------------------------------------------
    # Evitar series completamente constantes.
    # --------------------------------------------------------

    if (
        values.nunique()
        <= 1
        and len(
            values
        ) >= 5
    ):

        valid = False

        reasons.append(
            "serie_constante"
        )

    # --------------------------------------------------------
    # Estadísticas.
    # --------------------------------------------------------

    last_level = (
        float(
            values.iloc[
                -1
            ]
        )
        if not values.empty
        else np.nan
    )

    mean_level = (
        float(
            values.mean()
        )
        if not values.empty
        else np.nan
    )

    min_level = (
        float(
            values.min()
        )
        if not values.empty
        else np.nan
    )

    max_level = (
        float(
            values.max()
        )
        if not values.empty
        else np.nan
    )

    last_date = (
        data[
            "datetime"
        ].max()
        if (
            "datetime"
            in data.columns
            and not data.empty
        )
        else pd.NaT
    )

    return {

        "valid":
            bool(
                valid
            ),

        "records":
            int(
                len(
                    values
                )
            ),

        "last_level":
            last_level,

        "mean_level":
            mean_level,

        "min_level":
            min_level,

        "max_level":
            max_level,

        "last_date":
            last_date,

        "station_score":
            station_score,

        "river_score":
            _river_score(
                candidate.get(
                    "rio"
                )
            ),

        "reasons":
            reasons,
    }


# ============================================================
# SELECCIONAR SERIE
# ============================================================

def seleccionar_serie(
    station,
    start,
    end,
):

    candidates = buscar_candidatos_estacion(
        station,
        start,
        end,
    )

    metadata = {

        "station":
            station,

        "status":
            "sin_serie",

        "candidate_count":
            int(
                len(
                    candidates
                )
            ),

        "tested":
            [],
    }

    if candidates.empty:

        return (
            None,
            metadata,
        )

    # --------------------------------------------------------
    # Validamos candidatos con observaciones reales.
    # --------------------------------------------------------

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

        window = _validation_window(
            candidate,
            start,
            end,
        )

        if window is None:
            continue

        validation_start, validation_end = (
            window
        )

        try:

            observations = consultar_a5(

                series_id,

                validation_start,

                validation_end,
            )

            validation = validar_serie(

                station,

                candidate,

                observations,
            )

            tested = {

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
                tested
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

                    "records_validation":
                        validation.get(
                            "records",
                            0,
                        ),

                    "last_level_validation":
                        validation.get(
                            "last_level"
                        ),

                    "last_date_validation":
                        validation.get(
                            "last_date"
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

                    "nombre":
                        candidate.get(
                            "nombre"
                        ),

                    "rio":
                        candidate.get(
                            "rio"
                        ),

                    "valid":
                        False,

                    "reasons":
                        [
                            "error_consulta"
                        ],

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
# CONSULTA EN BLOQUES
# ============================================================

def _consultar_historia_bloques(
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

    if end_dt < start_dt:

        return pd.DataFrame()

    frames = []

    block_start = start_dt

    while block_start <= end_dt:

        block_end = min(

            block_start
            + pd.DateOffset(
                years=
                    HISTORY_BLOCK_YEARS
            )
            - pd.Timedelta(
                days=1
            ),

            end_dt,
        )

        try:

            part = consultar_a5(

                series_id,

                block_start,

                block_end,
            )

            if (
                part is not None
                and not part.empty
            ):

                frames.append(
                    part
                )

        except Exception:

            # Un bloque sin datos no invalida todo el historial.
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

    result[
        "datetime"
    ] = _to_naive_datetime_series(
        result[
            "datetime"
        ]
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
# DESCARGAR HISTORIA DE SERIE
# ============================================================

def _download_selected_history(
    selected,
    requested_start,
    requested_end,
    force_full_history=False,
):

    series_id = _safe_int(
        selected.get(
            "series_id"
        ),
        0,
    )

    if series_id <= 0:

        return pd.DataFrame()

    requested_start = pd.to_datetime(
        requested_start
    ).normalize()

    requested_end = pd.to_datetime(
        requested_end
    ).normalize()

    catalog_start = _safe_timestamp_naive(
        selected.get(
            "timestart"
        )
    )

    catalog_end = _safe_timestamp_naive(
        selected.get(
            "timeend"
        )
    )

    # --------------------------------------------------------
    # Inicio
    # --------------------------------------------------------

    if force_full_history:

        if not pd.isna(
            catalog_start
        ):

            start_dt = max(
                pd.to_datetime(
                    DEFAULT_HISTORY_FLOOR
                ).normalize(),
                pd.Timestamp(
                    catalog_start
                ).normalize(),
            )

        else:

            start_dt = pd.to_datetime(
                DEFAULT_HISTORY_FLOOR
            ).normalize()

    else:

        start_dt = requested_start

        if not pd.isna(
            catalog_start
        ):

            start_dt = max(
                start_dt,
                pd.Timestamp(
                    catalog_start
                ).normalize(),
            )

    # --------------------------------------------------------
    # Fin
    # --------------------------------------------------------

    end_dt = requested_end

    if not pd.isna(
        catalog_end
    ):

        end_dt = min(
            end_dt,
            pd.Timestamp(
                catalog_end
            ).normalize(),
        )

    if end_dt < start_dt:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Consulta corta.
    # --------------------------------------------------------

    total_days = (
        end_dt
        - start_dt
    ).days

    if (
        total_days
        <= 365 * HISTORY_BLOCK_YEARS
        and not force_full_history
    ):

        try:

            data = consultar_a5(
                series_id,
                start_dt,
                end_dt,
            )

            if not data.empty:

                return data

        except Exception:

            pass

    # --------------------------------------------------------
    # Historial largo.
    # --------------------------------------------------------

    return _consultar_historia_bloques(
        series_id,
        start_dt,
        end_dt,
    )


# ============================================================
# CONVERTIR A DIARIO
# ============================================================

def convertir_diario(
    df,
    output_column,
):

    if (
        df is None
        or df.empty
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                output_column,
            ]
        )

    x = df.copy()

    x[
        "datetime"
    ] = _to_naive_datetime_series(
        x[
            "datetime"
        ]
    )

    x[
        "value"
    ] = pd.to_numeric(
        x[
            "value"
        ],
        errors="coerce",
    )

    x = x.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )

    x = x[
        (
            x[
                "value"
            ]
            >= LEVEL_MIN_VALID
        )
        &
        (
            x[
                "value"
            ]
            <= LEVEL_MAX_VALID
        )
    ]

    if x.empty:

        return pd.DataFrame(
            columns=[
                "datetime",
                output_column,
            ]
        )

    x[
        "datetime"
    ] = (
        x[
            "datetime"
        ]
        .dt
        .normalize()
    )

    result = (
        x
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

    result[
        "datetime"
    ] = _to_naive_datetime_series(
        result[
            "datetime"
        ]
    )

    return result


# ============================================================
# DESCARGAR ESTACIÓN
# ============================================================

def get_station_history(
    station,
    start,
    end,
    full_history=False,
):

    output_column = STATION_COLUMNS[
        station
    ]

    selected, metadata = seleccionar_serie(
        station,
        start,
        end,
    )

    if selected is None:

        metadata[
            "records"
        ] = 0

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    output_column,
                ]
            ),
            metadata,
        )

    try:

        raw = _download_selected_history(

            selected,

            start,

            end,

            force_full_history=
                full_history,
        )

        daily = convertir_diario(
            raw,
            output_column,
        )

        # ----------------------------------------------------
        # Segunda validación de la historia descargada.
        # --------------------------------------------------------

        if daily.empty:

            metadata[
                "status"
            ] = "sin_observaciones"

            metadata[
                "records"
            ] = 0

            return (
                daily,
                metadata,
            )

        values = pd.to_numeric(
            daily[
                output_column
            ],
            errors="coerce",
        )

        valid_values = (
            values
            .dropna()
        )

        if (
            len(
                valid_values
            )
            < MIN_VALID_OBSERVATIONS
        ):

            metadata[
                "status"
            ] = "pocas_observaciones"

            metadata[
                "records"
            ] = int(
                len(
                    valid_values
                )
            )

            return (
                pd.DataFrame(
                    columns=[
                        "datetime",
                        output_column,
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
            len(
                daily
            )
        )

        metadata[
            "first_date"
        ] = daily[
            "datetime"
        ].min()

        metadata[
            "last_date"
        ] = daily[
            "datetime"
        ].max()

        metadata[
            "last_level"
        ] = float(
            valid_values.iloc[
                -1
            ]
        )

        metadata[
            "min_level"
        ] = float(
            valid_values.min()
        )

        metadata[
            "max_level"
        ] = float(
            valid_values.max()
        )

        metadata[
            "mean_level"
        ] = float(
            valid_values.mean()
        )

        metadata[
            "full_history"
        ] = bool(
            full_history
        )

        return (
            daily,
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
                    output_column,
                ]
            ),
            metadata,
        )


# ============================================================
# MERGE DEFENSIVO
# ============================================================

def _merge_station_frames(
    frames,
):

    valid_frames = []

    for frame in frames:

        if (
            frame is None
            or not isinstance(
                frame,
                pd.DataFrame,
            )
            or frame.empty
            or "datetime"
            not in frame.columns
        ):

            continue

        x = frame.copy()

        # ----------------------------------------------------
        # CRÍTICO:
        # Todas las fechas quedan timezone-naive ANTES
        # del merge.
        # --------------------------------------------------------

        x[
            "datetime"
        ] = _to_naive_datetime_series(
            x[
                "datetime"
            ]
        )

        x = x.dropna(
            subset=[
                "datetime"
            ]
        )

        x[
            "datetime"
        ] = (
            x[
                "datetime"
            ]
            .dt
            .normalize()
        )

        x = (
            x
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

        valid_frames.append(
            x
        )

    if not valid_frames:

        return pd.DataFrame(
            columns=[
                "datetime"
            ]
        )

    result = valid_frames[0]

    for frame in valid_frames[
        1:
    ]:

        # Segunda normalización defensiva.
        result[
            "datetime"
        ] = _to_naive_datetime_series(
            result[
                "datetime"
            ]
        )

        frame[
            "datetime"
        ] = _to_naive_datetime_series(
            frame[
                "datetime"
            ]
        )

        result = result.merge(
            frame,
            on="datetime",
            how="outer",
        )

    result[
        "datetime"
    ] = _to_naive_datetime_series(
        result[
            "datetime"
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
# FUNCIÓN PRINCIPAL
# ============================================================

def get_upstream_history(
    start,
    end,
):

    start = _normalize_date(
        start
    )

    end = _normalize_date(
        end
    )

    frames = []

    station_metadata = {}

    for station in STATIONS:

        # ----------------------------------------------------
        # Corrientes:
        # conservar TODO el historial disponible.
        #
        # Resto:
        # rango solicitado para entrenamiento operativo.
        # --------------------------------------------------------

        full_history = (
            station
            == "Corrientes"
        )

        try:

            station_df, metadata = (
                get_station_history(

                    station,

                    start,

                    end,

                    full_history=
                        full_history,
                )
            )

            station_metadata[
                station
            ] = metadata

            if (
                station_df is not None
                and not station_df.empty
            ):

                frames.append(
                    station_df
                )

        except Exception as exc:

            station_metadata[
                station
            ] = {

                "station":
                    station,

                "status":
                    "error",

                "records":
                    0,

                "error":
                    str(
                        exc
                    ),
            }

    result = _merge_station_frames(
        frames
    )

    # ========================================================
    # GARANTIZAR COLUMNAS
    # ========================================================

    for station in STATIONS:

        col = STATION_COLUMNS[
            station
        ]

        if col not in result.columns:

            result[
                col
            ] = np.nan

    # ========================================================
    # NORMALIZACIÓN FINAL
    # ========================================================

    if not result.empty:

        result[
            "datetime"
        ] = _to_naive_datetime_series(
            result[
                "datetime"
            ]
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

    # ========================================================
    # COBERTURA
    # ========================================================

    coverage = {}

    available_stations = []

    for station in STATIONS:

        col = STATION_COLUMNS[
            station
        ]

        count = 0

        if (
            col in result.columns
            and not result.empty
        ):

            count = int(
                pd.to_numeric(
                    result[
                        col
                    ],
                    errors="coerce",
                )
                .notna()
                .sum()
            )

        coverage[
            col
        ] = count

        if count >= MIN_VALID_OBSERVATIONS:

            available_stations.append(
                station
            )

    metadata = {

        "version":
            VERSION,

        "status":
            (
                "ok"
                if available_stations
                else "sin_datos"
            ),

        "source":
            "INA A5",

        "variable":
            "Nivel",

        "var_id":
            VAR_ID_NIVEL,

        "stations":
            station_metadata,

        "available_stations":
            available_stations,

        "station_count":
            len(
                available_stations
            ),

        "coverage":
            coverage,

        "records":
            int(
                len(
                    result
                )
            ),

        "start":
            (
                result[
                    "datetime"
                ].min()
                if not result.empty
                else None
            ),

        "end":
            (
                result[
                    "datetime"
                ].max()
                if not result.empty
                else None
            ),

        "corrientes_full_history":
            True,
    }

    return (
        result,
        metadata,
    )


# ============================================================
# ESTADO ACTUAL DE UNA ESTACIÓN
# ============================================================

def _station_current_state(
    history,
    station,
):

    col = STATION_COLUMNS[
        station
    ]

    if (
        history is None
        or history.empty
        or col not in history.columns
    ):

        return None

    data = history[
        [
            "datetime",
            col,
        ]
    ].copy()

    data[
        col
    ] = pd.to_numeric(
        data[
            col
        ],
        errors="coerce",
    )

    data = (
        data
        .dropna(
            subset=[
                "datetime",
                col,
            ]
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    if data.empty:

        return None

    current = float(
        data.iloc[
            -1
        ][
            col
        ]
    )

    current_date = data.iloc[
        -1
    ][
        "datetime"
    ]

    delta_1 = np.nan
    delta_3 = np.nan
    delta_7 = np.nan

    if len(
        data
    ) >= 2:

        delta_1 = (
            current
            -
            float(
                data.iloc[
                    -2
                ][
                    col
                ]
            )
        )

    if len(
        data
    ) >= 4:

        delta_3 = (
            current
            -
            float(
                data.iloc[
                    -4
                ][
                    col
                ]
            )
        )

    if len(
        data
    ) >= 8:

        delta_7 = (
            current
            -
            float(
                data.iloc[
                    -8
                ][
                    col
                ]
            )
        )

    if np.isfinite(
        delta_3
    ):

        if delta_3 > 0.03:

            trend = "Creciendo"

            arrow = "↑"

        elif delta_3 < -0.03:

            trend = "Bajando"

            arrow = "↓"

        else:

            trend = "Estable"

            arrow = "→"

    elif np.isfinite(
        delta_1
    ):

        if delta_1 > 0.02:

            trend = "Creciendo"

            arrow = "↑"

        elif delta_1 < -0.02:

            trend = "Bajando"

            arrow = "↓"

        else:

            trend = "Estable"

            arrow = "→"

    else:

        trend = "Sin tendencia"

        arrow = "—"

    return {

        "station":
            station,

        "datetime":
            current_date,

        "level":
            current,

        "delta_1":
            (
                float(
                    delta_1
                )
                if np.isfinite(
                    delta_1
                )
                else np.nan
            ),

        "delta_3":
            (
                float(
                    delta_3
                )
                if np.isfinite(
                    delta_3
                )
                else np.nan
            ),

        "delta_7":
            (
                float(
                    delta_7
                )
                if np.isfinite(
                    delta_7
                )
                else np.nan
            ),

        "trend":
            trend,

        "arrow":
            arrow,
    }


# ============================================================
# RESUMEN ACTUAL DEL CORREDOR
# ============================================================

def current_state_table(
    history,
):

    rows = []

    for station in STATIONS:

        state = _station_current_state(
            history,
            station,
        )

        if state is None:

            rows.append(
                {
                    "Estación":
                        station,

                    "Nivel":
                        np.nan,

                    "Δ 1 día":
                        np.nan,

                    "Δ 3 días":
                        np.nan,

                    "Δ 7 días":
                        np.nan,

                    "Estado":
                        "Sin datos",
                }
            )

            continue

        rows.append(
            {
                "Estación":
                    station,

                "Nivel":
                    state[
                        "level"
                    ],

                "Δ 1 día":
                    state[
                        "delta_1"
                    ],

                "Δ 3 días":
                    state[
                        "delta_3"
                    ],

                "Δ 7 días":
                    state[
                        "delta_7"
                    ],

                "Estado":
                    (
                        state[
                            "arrow"
                        ]
                        + " "
                        + state[
                            "trend"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COMPATIBILIDAD CON APP / MODEL
# ============================================================

def resumen_niveles_estaciones(
    history,
):

    return current_state_table(
        history
    )


# ============================================================
# TABLA DE DIAGNÓSTICO
# ============================================================

def diagnostic_table(
    start,
    end,
):

    _, metadata = get_upstream_history(
        start,
        end,
    )

    rows = []

    station_meta = metadata.get(
        "stations",
        {}
    )

    for station in STATIONS:

        info = station_meta.get(
            station,
            {}
        )

        rows.append(
            {
                "Estación":
                    station,

                "Estado":
                    info.get(
                        "status",
                        "sin_datos",
                    ),

                "Serie ID":
                    info.get(
                        "series_id"
                    ),

                "Serie":
                    info.get(
                        "series_name"
                    ),

                "Río":
                    info.get(
                        "river"
                    ),

                "Registros":
                    info.get(
                        "records",
                        0,
                    ),

                "Primer dato":
                    info.get(
                        "first_date"
                    ),

                "Último dato":
                    info.get(
                        "last_date"
                    ),

                "Último nivel":
                    info.get(
                        "last_level"
                    ),

                "Nivel mínimo":
                    info.get(
                        "min_level"
                    ),

                "Nivel máximo":
                    info.get(
                        "max_level"
                    ),

                "Historial completo":
                    info.get(
                        "full_history",
                        False,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DIAGNÓSTICO DETALLADO
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

        catalog = descargar_catalogo_geojson()

        level_catalog = catalog[
            catalog[
                "var_id"
            ]
            == VAR_ID_NIVEL
        ]

        result[
            "catalog_records"
        ] = int(
            len(
                catalog
            )
        )

        result[
            "level_series_records"
        ] = int(
            len(
                level_catalog
            )
        )

    except Exception as exc:

        result[
            "catalog_error"
        ] = str(
            exc
        )

    # ========================================================
    # ESTACIONES
    # ========================================================

    station_results = {}

    for station in STATIONS:

        station_result = {}

        try:

            candidates = (
                buscar_candidatos_estacion(
                    station,
                    start,
                    end,
                )
            )

            station_result[
                "candidate_count"
            ] = int(
                len(
                    candidates
                )
            )

            if not candidates.empty:

                preview_columns = [
                    col
                    for col in [
                        "series_id",
                        "nombre",
                        "rio",
                        "var_id",
                        "proc_id",
                        "timestart",
                        "timeend",
                        "count",
                        "is_parana_candidate",
                        "score",
                    ]
                    if col
                    in candidates.columns
                ]

                station_result[
                    "top_candidates"
                ] = (
                    candidates[
                        preview_columns
                    ]
                    .head(
                        5
                    )
                    .to_dict(
                        orient="records"
                    )
                )

            selected, selection_meta = (
                seleccionar_serie(
                    station,
                    start,
                    end,
                )
            )

            station_result[
                "selection"
            ] = selection_meta

            if selected is not None:

                station_result[
                    "selected_series_id"
                ] = selected.get(
                    "series_id"
                )

                station_result[
                    "selected_name"
                ] = selected.get(
                    "nombre"
                )

                station_result[
                    "selected_river"
                ] = selected.get(
                    "rio"
                )

        except Exception as exc:

            station_result[
                "error"
            ] = str(
                exc
            )

        station_results[
            station
        ] = station_result

    result[
        "stations"
    ] = station_results

    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    try:

        history, metadata = (
            get_upstream_history(
                start,
                end,
            )
        )

        result[
            "status"
        ] = metadata.get(
            "status",
            "ok",
        )

        result[
            "records"
        ] = int(
            len(
                history
            )
        )

        result[
            "available_stations"
        ] = metadata.get(
            "available_stations",
            []
        )

        result[
            "station_count"
        ] = metadata.get(
            "station_count",
            0,
        )

        result[
            "coverage"
        ] = metadata.get(
            "coverage",
            {}
        )

        result[
            "history_start"
        ] = metadata.get(
            "start"
        )

        result[
            "history_end"
        ] = metadata.get(
            "end"
        )

        result[
            "corrientes_full_history"
        ] = metadata.get(
            "corrientes_full_history",
            False,
        )

        result[
            "diagnostic_table"
        ] = diagnostic_table_from_metadata(
            metadata
        ).to_dict(
            orient="records"
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


# ============================================================
# DIAGNÓSTICO DESDE METADATA
# ============================================================

def diagnostic_table_from_metadata(
    metadata,
):

    rows = []

    station_meta = metadata.get(
        "stations",
        {}
    )

    for station in STATIONS:

        info = station_meta.get(
            station,
            {}
        )

        rows.append(
            {
                "Estación":
                    station,

                "Estado":
                    info.get(
                        "status",
                        "sin_datos",
                    ),

                "Serie ID":
                    info.get(
                        "series_id"
                    ),

                "Serie":
                    info.get(
                        "series_name"
                    ),

                "Río":
                    info.get(
                        "river"
                    ),

                "Registros":
                    info.get(
                        "records",
                        0,
                    ),

                "Primer dato":
                    info.get(
                        "first_date"
                    ),

                "Último dato":
                    info.get(
                        "last_date"
                    ),

                "Último nivel":
                    info.get(
                        "last_level"
                    ),

                "Mínimo":
                    info.get(
                        "min_level"
                    ),

                "Máximo":
                    info.get(
                        "max_level"
                    ),

                "Historial completo":
                    info.get(
                        "full_history",
                        False,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )
