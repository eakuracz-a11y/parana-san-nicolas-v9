# ============================================================
# PARANÁ · SAN NICOLÁS
# src/hydrology.py
# V11.10 COMPLETO
#
# PROPAGACIÓN HIDROLÓGICA MULTIESTACIÓN
#
# OBJETIVOS
# ------------------------------------------------------------
# - Analizar niveles históricos aguas arriba
# - Incorporar caudales históricos por estación
# - Incorporar lluvias históricas por estación
# - Calcular retardos entre estaciones
# - Calcular retardo Corrientes -> San Nicolás
# - Detectar eventos históricos de creciente
# - Relacionar:
#       nivel
#       variación de nivel
#       caudal
#       variación de caudal
#       lluvia acumulada
#       tiempo de propagación
#       respuesta en San Nicolás
# - Buscar eventos históricos similares al estado actual
# - Crear features para model.py
# - Mantener compatibilidad con app.py actual
#
# IMPORTANTE
# ------------------------------------------------------------
# Los niveles de distintas estaciones tienen ceros hidrométricos
# diferentes. El análisis se basa principalmente en variaciones,
# tendencias y eventos, no en asumir que 4 m en Corrientes
# equivalen físicamente a 4 m en San Nicolás.
# ============================================================


from functools import lru_cache

import numpy as np
import pandas as pd
import requests


# ============================================================
# VERSIÓN
# ============================================================

VERSION = "V11.10"


# ============================================================
# INA
# ============================================================

INA_BASE_URL = "https://alerta.ina.gob.ar/a5"

INA_OBSERVATIONS_URL = (
    INA_BASE_URL
    + "/getObservaciones"
)

INA_SERIES_GEOJSON_URL = (
    INA_BASE_URL
    + "/obs/puntual/series"
)


# ============================================================
# SAN NICOLÁS
# ============================================================

SAN_NICOLAS_SERIES_ID = 36


# ============================================================
# CONFIGURACIÓN
# ============================================================

REQUEST_TIMEOUT = 45

DEFAULT_HISTORY_START = "1900-01-01"

HISTORY_BLOCK_YEARS = 5

MAX_LAG_DAYS = 35

DEFAULT_LAG_ANALYSIS_DAYS = 30

MIN_CORRELATION_SAMPLES = 30

MIN_EVENT_DISTANCE_DAYS = 8

EVENT_QUANTILE = 0.85

EVENT_BASELINE_DAYS = 7

EVENT_SEARCH_BEFORE = 3

EVENT_SEARCH_AFTER = 30

MIN_EVENT_RISE = 0.08

CURRENT_LOOKBACK_DAYS = 7

SIMILAR_EVENTS_DEFAULT = 8


# ============================================================
# CORREDOR HIDROLÓGICO
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


UPSTREAM_STATIONS = [
    "Corrientes",
    "Goya",
    "La Paz",
    "Paraná",
    "Diamante",
    "Rosario",
    "Villa Constitución",
]


# ============================================================
# NOMBRES DE COLUMNAS
# ============================================================

LEVEL_COLUMNS = {

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

    "San Nicolás":
        "nivel",
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
# TRAMOS
# ============================================================

SEGMENTS = [

    (
        "Corrientes",
        "Goya",
    ),

    (
        "Goya",
        "La Paz",
    ),

    (
        "La Paz",
        "Paraná",
    ),

    (
        "Paraná",
        "Diamante",
    ),

    (
        "Diamante",
        "Rosario",
    ),

    (
        "Rosario",
        "Villa Constitución",
    ),

    (
        "Villa Constitución",
        "San Nicolás",
    ),
]


# ============================================================
# REQUEST SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
            "Parana-San-Nicolas-Hydrology/11.10",

        "Accept":
            "application/json",
    }
)


# ============================================================
# UTILIDADES
# ============================================================

def _normalizar_datetime(values):

    return (
        pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
    )


def _safe_datetime(value):

    dt = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(dt):
        return None

    return dt.tz_localize(
        None
    )


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


def _to_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _safe_corr(
    a,
    b,
):

    data = pd.DataFrame(
        {
            "a":
                _to_numeric(
                    a
                ),

            "b":
                _to_numeric(
                    b
                ),
        }
    ).dropna()


    if len(data) < 5:

        return np.nan


    if (
        data["a"].std()
        == 0
        or data["b"].std()
        == 0
    ):

        return np.nan


    return float(
        data["a"].corr(
            data["b"]
        )
    )


def _slug(
    station,
):

    replacements = {

        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
    }


    text = station.lower()


    for old, new in (
        replacements.items()
    ):

        text = text.replace(
            old,
            new,
        )


    return (
        text
        .replace(
            " ",
            "_",
        )
    )


def _normalizar_diario(
    df,
    value_cols=None,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):

        return pd.DataFrame()


    work = df.copy()


    if "datetime" not in work.columns:

        return pd.DataFrame()


    work["datetime"] = (
        _normalizar_datetime(
            work["datetime"]
        )
    )


    work = work.dropna(
        subset=[
            "datetime"
        ]
    )


    work["datetime"] = (
        work["datetime"]
        .dt
        .normalize()
    )


    if value_cols is None:

        value_cols = [
            c
            for c in work.columns
            if c != "datetime"
        ]


    available = []


    for col in value_cols:

        if col not in work.columns:
            continue


        work[col] = (
            _to_numeric(
                work[col]
            )
        )


        available.append(
            col
        )


    if not available:

        return (
            work[
                ["datetime"]
            ]
            .drop_duplicates()
            .sort_values(
                "datetime"
            )
            .reset_index(
                drop=True
            )
        )


    return (
        work[
            [
                "datetime",
                *available,
            ]
        ]
        .groupby(
            "datetime",
            as_index=False,
        )[available]
        .mean()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PARSER INA
# ============================================================

def _extract_records(
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
                    x,
                    dict,
                )
                for x in data
            )
        ):

            sample_keys = set()


            for row in data[
                :5
            ]:

                sample_keys.update(
                    row.keys()
                )


            if sample_keys.intersection(
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

            result = _extract_records(
                item
            )

            if result:
                return result


    elif isinstance(
        data,
        dict,
    ):

        for key in [

            "observaciones",
            "observations",
            "datos",
            "data",
            "records",
            "values",
            "result",

        ]:

            if key in data:

                result = _extract_records(
                    data[
                        key
                    ]
                )

                if result:
                    return result


        for value in data.values():

            result = _extract_records(
                value
            )

            if result:
                return result


    return []


def _normalizar_respuesta_ina(
    data,
):

    records = _extract_records(
        data
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
        "nivel",
        "obs_value",
        "valor_num",

    ]


    for record in records:

        dt = None
        value = None


        for field in date_fields:

            if (
                record.get(
                    field
                )
                is not None
            ):

                dt = record[
                    field
                ]

                break


        for field in value_fields:

            if (
                record.get(
                    field
                )
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


        value = _safe_float(
            value
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


        if (
            value < -5
            or value > 20
        ):

            continue


        rows.append(
            {
                "datetime":
                    dt,

                "nivel":
                    value,
            }
        )


    if not rows:

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel",
            ]
        )


    result = pd.DataFrame(
        rows
    )


    result[
        "datetime"
    ] = _normalizar_datetime(
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


    return (
        result
        .groupby(
            "datetime",
            as_index=False,
        )["nivel"]
        .median()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CATÁLOGO SAN NICOLÁS
# ============================================================

@lru_cache(
    maxsize=1
)
def _get_san_nicolas_catalog_dates():

    fallback_start = pd.Timestamp(
        DEFAULT_HISTORY_START
    )

    fallback_end = (
        pd.Timestamp.today()
        .normalize()
    )


    try:

        response = SESSION.get(

            INA_SERIES_GEOJSON_URL,

            params={
                "format":
                    "geojson",
            },

            timeout=
                REQUEST_TIMEOUT,
        )


        response.raise_for_status()

        data = response.json()


        for feature in data.get(
            "features",
            [],
        ):

            props = feature.get(
                "properties",
                {},
            )


            series_id = _safe_int(
                props.get(
                    "series_id"
                )
                or feature.get(
                    "id"
                )
            )


            if (
                series_id
                != SAN_NICOLAS_SERIES_ID
            ):

                continue


            start = _safe_datetime(
                props.get(
                    "timestart"
                )
            )


            end = _safe_datetime(
                props.get(
                    "timeend"
                )
            )


            if start is None:
                start = fallback_start


            if end is None:
                end = fallback_end


            return (
                start.normalize(),
                end.normalize(),
            )


    except Exception:
        pass


    return (
        fallback_start,
        fallback_end,
    )


# ============================================================
# CONSULTAR SERIE 36
# ============================================================

def _query_san_nicolas(
    start,
    end,
):

    params = {

        "tipo":
            "puntual",

        "series_id":
            SAN_NICOLAS_SERIES_ID,

        "timestart":
            pd.to_datetime(
                start
            ).strftime(
                "%Y-%m-%d"
            ),

        "timeend":
            pd.to_datetime(
                end
            ).strftime(
                "%Y-%m-%d"
            ),
    }


    response = SESSION.get(

        INA_OBSERVATIONS_URL,

        params=
            params,

        timeout=
            REQUEST_TIMEOUT,
    )


    response.raise_for_status()


    return _normalizar_respuesta_ina(
        response.json()
    )


# ============================================================
# HISTORIAL COMPLETO SAN NICOLÁS
# ============================================================

@lru_cache(
    maxsize=8
)
def get_san_nicolas_full_history(
    end_date=None,
):

    (
        catalog_start,
        catalog_end,
    ) = (
        _get_san_nicolas_catalog_dates()
    )


    if end_date is not None:

        requested_end = pd.to_datetime(
            end_date,
            errors="coerce",
        )


        if pd.notna(
            requested_end
        ):

            catalog_end = min(
                catalog_end,
                requested_end,
            )


    frames = []


    current = catalog_start


    while current <= catalog_end:

        block_end = min(

            current
            + pd.DateOffset(
                years=
                    HISTORY_BLOCK_YEARS
            )
            - pd.Timedelta(
                days=1
            ),

            catalog_end,
        )


        try:

            frame = _query_san_nicolas(
                current,
                block_end,
            )

        except Exception:

            frame = pd.DataFrame()


        if not frame.empty:

            frames.append(
                frame
            )


        current = (
            block_end
            + pd.Timedelta(
                days=1
            )
        )


    if not frames:

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel_san_nicolas",
            ]
        )


    result = pd.concat(
        frames,
        ignore_index=True,
    )


    result = (
        result
        .drop_duplicates(
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


    result = result.rename(
        columns={
            "nivel":
                "nivel_san_nicolas"
        }
    )


    return result


# ============================================================
# PREPARAR NIVEL SAN NICOLÁS
# ============================================================

def preparar_san_nicolas(
    df,
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
                "nivel",
            ]
        )


    work = df.copy()


    if "datetime" not in work.columns:

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel",
            ]
        )


    level_col = None


    for col in [
        "nivel",
        "value",
        "nivel_san_nicolas",
    ]:

        if col in work.columns:

            level_col = col
            break


    if level_col is None:

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel",
            ]
        )


    work = work[
        [
            "datetime",
            level_col,
        ]
    ].copy()


    work = work.rename(
        columns={
            level_col:
                "nivel"
        }
    )


    return _normalizar_diario(
        work,
        [
            "nivel"
        ],
    )


# ============================================================
# PREPARAR AGUAS ARRIBA
# ============================================================

def preparar_upstream(
    upstream_history,
):

    if (
        upstream_history is None
        or not isinstance(
            upstream_history,
            pd.DataFrame,
        )
        or upstream_history.empty
    ):

        return pd.DataFrame()


    cols = [
        c
        for c in (
            LEVEL_COLUMNS.values()
        )
        if (
            c != "nivel"
            and c
            in upstream_history.columns
        )
    ]


    if not cols:

        return pd.DataFrame()


    return _normalizar_diario(
        upstream_history,
        cols,
    )


# ============================================================
# PREPARAR EXÓGENAS
# ============================================================

def preparar_exogenas(
    exog_history,
):

    if (
        exog_history is None
        or not isinstance(
            exog_history,
            pd.DataFrame,
        )
        or exog_history.empty
    ):

        return pd.DataFrame()


    wanted = []


    for col in exog_history.columns:

        if col == "datetime":

            continue


        if (
            col.startswith(
                "q_"
            )
            or col.startswith(
                "rain_"
            )
            or col in [
                "caudal_m3s",
                "precip_mm",
            ]
        ):

            wanted.append(
                col
            )


    return _normalizar_diario(
        exog_history,
        wanted,
    )


# ============================================================
# CONSTRUIR DATASET HIDROLÓGICO COMPLETO
# ============================================================

def construir_dataset_hidrologico(
    df,
    upstream_history=None,
    exog_history=None,
    usar_historial_completo=True,
):

    local = preparar_san_nicolas(
        df
    )


    if local.empty:

        return pd.DataFrame()


    # ========================================================
    # HISTORIAL COMPLETO DE SAN NICOLÁS
    # ========================================================

    if usar_historial_completo:

        try:

            full_sn = (
                get_san_nicolas_full_history(
                    str(
                        local[
                            "datetime"
                        ].max().date()
                    )
                )
                .copy()
            )

        except Exception:

            full_sn = pd.DataFrame()


        if not full_sn.empty:

            full_sn = full_sn.rename(
                columns={
                    "nivel_san_nicolas":
                        "nivel"
                }
            )


            local = (
                pd.concat(
                    [
                        full_sn[
                            [
                                "datetime",
                                "nivel",
                            ]
                        ],
                        local[
                            [
                                "datetime",
                                "nivel",
                            ]
                        ],
                    ],
                    ignore_index=True,
                )
                .drop_duplicates(
                    subset=[
                        "datetime"
                    ],
                    keep="last",
                )
                .sort_values(
                    "datetime"
                )
                .reset_index(
                    drop=True
                )
            )


    result = local.copy()


    # ========================================================
    # UPSTREAM
    # ========================================================

    upstream = preparar_upstream(
        upstream_history
    )


    if not upstream.empty:

        result = result.merge(
            upstream,
            on="datetime",
            how="outer",
        )


    # ========================================================
    # EXÓGENAS
    # ========================================================

    exog = preparar_exogenas(
        exog_history
    )


    if not exog.empty:

        result = result.merge(
            exog,
            on="datetime",
            how="left",
        )


    result[
        "datetime"
    ] = _normalizar_datetime(
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
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CORRELACIÓN POR LAG
# ============================================================

def calcular_lag_entre_series(
    df,
    upstream_col,
    downstream_col,
    max_lag=MAX_LAG_DAYS,
):

    if (
        df is None
        or df.empty
        or upstream_col
        not in df.columns
        or downstream_col
        not in df.columns
    ):

        return {
            "best_lag_days":
                None,

            "correlation":
                None,

            "samples":
                0,

            "lag_table":
                pd.DataFrame(),
        }


    work = df[
        [
            "datetime",
            upstream_col,
            downstream_col,
        ]
    ].copy()


    work[upstream_col] = (
        _to_numeric(
            work[
                upstream_col
            ]
        )
    )


    work[downstream_col] = (
        _to_numeric(
            work[
                downstream_col
            ]
        )
    )


    # ========================================================
    # Usamos variaciones de 3 días
    # para reducir sesgo por estacionalidad
    # ========================================================

    work[
        "up_change"
    ] = (
        work[
            upstream_col
        ].diff(
            3
        )
    )


    work[
        "down_change"
    ] = (
        work[
            downstream_col
        ].diff(
            3
        )
    )


    rows = []


    for lag in range(
        0,
        int(
            max_lag
        )
        + 1,
    ):

        # ----------------------------------------------------
        # upstream en t debe correlacionar con downstream
        # en t + lag
        # ----------------------------------------------------

        shifted_down = (
            work[
                "down_change"
            ]
            .shift(
                -lag
            )
        )


        temp = pd.DataFrame(
            {
                "up":
                    work[
                        "up_change"
                    ],

                "down":
                    shifted_down,
            }
        ).dropna()


        if (
            len(
                temp
            )
            < MIN_CORRELATION_SAMPLES
        ):

            continue


        corr = _safe_corr(
            temp[
                "up"
            ],
            temp[
                "down"
            ],
        )


        if not np.isfinite(
            corr
        ):

            continue


        rows.append(
            {
                "lag_days":
                    lag,

                "correlation":
                    corr,

                "abs_correlation":
                    abs(
                        corr
                    ),

                "samples":
                    len(
                        temp
                    ),
            }
        )


    if not rows:

        return {
            "best_lag_days":
                None,

            "correlation":
                None,

            "samples":
                0,

            "lag_table":
                pd.DataFrame(),
        }


    table = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "abs_correlation",
                "samples",
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


    best = table.iloc[
        0
    ]


    return {
        "best_lag_days":
            int(
                best[
                    "lag_days"
                ]
            ),

        "correlation":
            float(
                best[
                    "correlation"
                ]
            ),

        "samples":
            int(
                best[
                    "samples"
                ]
            ),

        "lag_table":
            table,
    }


# ============================================================
# LAGS ENTRE TODOS LOS TRAMOS
# ============================================================

def calcular_retardos_corredor(
    dataset,
):

    results = []


    for (
        upstream_station,
        downstream_station,
    ) in SEGMENTS:

        up_col = LEVEL_COLUMNS[
            upstream_station
        ]

        down_col = LEVEL_COLUMNS[
            downstream_station
        ]


        if (
            up_col
            not in dataset.columns
            or down_col
            not in dataset.columns
        ):

            results.append(
                {
                    "upstream":
                        upstream_station,

                    "downstream":
                        downstream_station,

                    "lag_days":
                        None,

                    "correlation":
                        None,

                    "samples":
                        0,
                }
            )

            continue


        lag = calcular_lag_entre_series(

            dataset,

            up_col,

            down_col,

            max_lag=
                DEFAULT_LAG_ANALYSIS_DAYS,
        )


        results.append(
            {
                "upstream":
                    upstream_station,

                "downstream":
                    downstream_station,

                "lag_days":
                    lag.get(
                        "best_lag_days"
                    ),

                "correlation":
                    lag.get(
                        "correlation"
                    ),

                "samples":
                    lag.get(
                        "samples"
                    ),
            }
        )


    return pd.DataFrame(
        results
    )


# ============================================================
# LAG CORRIENTES -> SAN NICOLÁS
# ============================================================

def calcular_lag_corrientes_san_nicolas(
    dataset,
):

    return calcular_lag_entre_series(

        dataset,

        "nivel_corrientes",

        "nivel",

        max_lag=
            MAX_LAG_DAYS,
    )


# ============================================================
# DETECTAR MÁXIMOS LOCALES
# ============================================================

def detectar_maximos_locales(
    df,
    column,
    quantile=EVENT_QUANTILE,
    min_distance=MIN_EVENT_DISTANCE_DAYS,
):

    if (
        df is None
        or df.empty
        or column not in df.columns
    ):

        return pd.DataFrame()


    values = _to_numeric(
        df[
            column
        ]
    )


    valid = values.dropna()


    if len(
        valid
    ) < 30:

        return pd.DataFrame()


    threshold = valid.quantile(
        quantile
    )


    candidate_indexes = []


    for i in range(
        1,
        len(
            values
        )
        - 1,
    ):

        current = values.iloc[
            i
        ]


        if not np.isfinite(
            current
        ):

            continue


        if current < threshold:
            continue


        prev_value = values.iloc[
            i - 1
        ]

        next_value = values.iloc[
            i + 1
        ]


        if (
            np.isfinite(
                prev_value
            )
            and np.isfinite(
                next_value
            )
            and current
            >= prev_value
            and current
            >= next_value
        ):

            candidate_indexes.append(
                i
            )


    if not candidate_indexes:

        return pd.DataFrame()


    selected = []


    last_date = None


    for idx in candidate_indexes:

        date = df[
            "datetime"
        ].iloc[
            idx
        ]


        if (
            last_date is None
            or (
                date
                - last_date
            ).days
            >= min_distance
        ):

            selected.append(
                idx
            )

            last_date = date


        else:

            previous_idx = selected[
                -1
            ]


            if (
                values.iloc[
                    idx
                ]
                >
                values.iloc[
                    previous_idx
                ]
            ):

                selected[
                    -1
                ] = idx

                last_date = date


    return (
        df.iloc[
            selected
        ][
            [
                "datetime",
                column,
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# VALOR MEDIO EN VENTANA
# ============================================================

def _window_mean(
    dataset,
    column,
    start,
    end,
):

    if (
        column
        not in dataset.columns
    ):

        return np.nan


    mask = (
        (
            dataset[
                "datetime"
            ]
            >= start
        )
        &
        (
            dataset[
                "datetime"
            ]
            <= end
        )
    )


    values = (
        _to_numeric(
            dataset.loc[
                mask,
                column,
            ]
        )
        .dropna()
    )


    if values.empty:

        return np.nan


    return float(
        values.mean()
    )


# ============================================================
# VALOR MÁXIMO EN VENTANA
# ============================================================

def _window_max(
    dataset,
    column,
    start,
    end,
):

    if (
        column
        not in dataset.columns
    ):

        return np.nan


    mask = (
        (
            dataset[
                "datetime"
            ]
            >= start
        )
        &
        (
            dataset[
                "datetime"
            ]
            <= end
        )
    )


    values = (
        _to_numeric(
            dataset.loc[
                mask,
                column,
            ]
        )
        .dropna()
    )


    if values.empty:

        return np.nan


    return float(
        values.max()
    )


# ============================================================
# LLUVIA ACUMULADA
# ============================================================

def _rain_sum(
    dataset,
    column,
    start,
    end,
):

    if column not in dataset.columns:

        return np.nan


    mask = (
        (
            dataset[
                "datetime"
            ]
            >= start
        )
        &
        (
            dataset[
                "datetime"
            ]
            <= end
        )
    )


    values = (
        _to_numeric(
            dataset.loc[
                mask,
                column,
            ]
        )
        .fillna(
            0.0
        )
    )


    if values.empty:

        return np.nan


    return float(
        values.sum()
    )


# ============================================================
# CONSTRUIR EVENTOS CORRIENTES -> SAN NICOLÁS
# ============================================================

def construir_eventos_corrientes_san_nicolas(
    dataset,
    expected_lag=None,
):

    required = [
        "datetime",
        "nivel_corrientes",
        "nivel",
    ]


    if any(
        c not in dataset.columns
        for c in required
    ):

        return pd.DataFrame()


    peaks = detectar_maximos_locales(
        dataset,
        "nivel_corrientes",
    )


    if peaks.empty:

        return pd.DataFrame()


    if expected_lag is None:

        expected_lag = 15


    events = []


    for _, peak in peaks.iterrows():

        corr_date = peak[
            "datetime"
        ]

        corr_peak = _safe_float(
            peak[
                "nivel_corrientes"
            ]
        )


        baseline_start = (
            corr_date
            - pd.Timedelta(
                days=
                    EVENT_BASELINE_DAYS
            )
        )


        baseline_end = (
            corr_date
            - pd.Timedelta(
                days=1
            )
        )


        corr_base = _window_mean(

            dataset,

            "nivel_corrientes",

            baseline_start,

            baseline_end,
        )


        if not np.isfinite(
            corr_base
        ):

            continue


        corr_rise = (
            corr_peak
            - corr_base
        )


        if (
            corr_rise
            < MIN_EVENT_RISE
        ):

            continue


        corr_speed = (
            corr_rise
            / max(
                EVENT_BASELINE_DAYS,
                1,
            )
        )


        target_date = (
            corr_date
            + pd.Timedelta(
                days=
                    expected_lag
            )
        )


        search_start = (
            target_date
            - pd.Timedelta(
                days=
                    EVENT_SEARCH_BEFORE
            )
        )


        search_end = (
            target_date
            + pd.Timedelta(
                days=
                    EVENT_SEARCH_AFTER
            )
        )


        mask = (
            (
                dataset[
                    "datetime"
                ]
                >= search_start
            )
            &
            (
                dataset[
                    "datetime"
                ]
                <= search_end
            )
        )


        sn_window = dataset.loc[
            mask,
            [
                "datetime",
                "nivel",
            ],
        ].dropna()


        if sn_window.empty:

            continue


        max_idx = (
            sn_window[
                "nivel"
            ]
            .idxmax()
        )


        sn_date = dataset.loc[
            max_idx,
            "datetime",
        ]


        sn_peak = _safe_float(
            dataset.loc[
                max_idx,
                "nivel",
            ]
        )


        lag_real = (
            sn_date
            - corr_date
        ).days


        if (
            lag_real < 0
            or lag_real > 40
        ):

            continue


        sn_baseline_start = (
            corr_date
            - pd.Timedelta(
                days=
                    EVENT_BASELINE_DAYS
            )
        )


        sn_baseline_end = (
            corr_date
            + pd.Timedelta(
                days=
                    max(
                        lag_real
                        - 2,
                        0,
                    )
            )
        )


        sn_base = _window_mean(

            dataset,

            "nivel",

            sn_baseline_start,

            sn_baseline_end,
        )


        if not np.isfinite(
            sn_base
        ):

            continue


        sn_response = (
            sn_peak
            - sn_base
        )


        # ----------------------------------------------------
        # evitar asociaciones sin respuesta positiva
        # ----------------------------------------------------

        if sn_response <= 0:

            continue


        relative_response = (
            sn_response
            / corr_rise
            if abs(
                corr_rise
            ) > 0.01
            else np.nan
        )


        # ====================================================
        # CAUDAL POR ESTACIÓN
        # ====================================================

        event = {

            "fecha_max_corrientes":
                corr_date,

            "max_corrientes_m":
                corr_peak,

            "nivel_base_corrientes_m":
                corr_base,

            "crecida_corrientes_m":
                corr_rise,

            "velocidad_corrientes_m_dia":
                corr_speed,

            "fecha_max_san_nicolas":
                sn_date,

            "max_san_nicolas_m":
                sn_peak,

            "nivel_base_san_nicolas_m":
                sn_base,

            "respuesta_san_nicolas_m":
                sn_response,

            "respuesta_relativa":
                relative_response,

            "lag_real_dias":
                lag_real,
        }


        # ====================================================
        # CAUDALES PREVIOS AL EVENTO
        # ====================================================

        for station in STATIONS:

            slug = _slug(
                station
            )


            q_col = FLOW_COLUMNS[
                station
            ]


            if q_col in dataset.columns:

                q_mean = _window_mean(

                    dataset,

                    q_col,

                    corr_date
                    - pd.Timedelta(
                        days=7
                    ),

                    corr_date,
                )


                q_max = _window_max(

                    dataset,

                    q_col,

                    corr_date
                    - pd.Timedelta(
                        days=7
                    ),

                    corr_date,
                )


                event[
                    f"q_{slug}_7d_mean"
                ] = q_mean


                event[
                    f"q_{slug}_7d_max"
                ] = q_max


            rain_col = RAIN_COLUMNS[
                station
            ]


            if rain_col in dataset.columns:

                rain_7d = _rain_sum(

                    dataset,

                    rain_col,

                    corr_date
                    - pd.Timedelta(
                        days=6
                    ),

                    corr_date,
                )


                rain_15d = _rain_sum(

                    dataset,

                    rain_col,

                    corr_date
                    - pd.Timedelta(
                        days=14
                    ),

                    corr_date,
                )


                rain_30d = _rain_sum(

                    dataset,

                    rain_col,

                    corr_date
                    - pd.Timedelta(
                        days=29
                    ),

                    corr_date,
                )


                event[
                    f"rain_{slug}_7d"
                ] = rain_7d


                event[
                    f"rain_{slug}_15d"
                ] = rain_15d


                event[
                    f"rain_{slug}_30d"
                ] = rain_30d


        events.append(
            event
        )


    if not events:

        return pd.DataFrame()


    return (
        pd.DataFrame(
            events
        )
        .sort_values(
            "fecha_max_corrientes"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# ESTADÍSTICAS DE PROPAGACIÓN
# ============================================================

def estadisticas_propagacion(
    events,
):

    if (
        events is None
        or events.empty
    ):

        return {
            "event_count":
                0,

            "median_lag_days":
                None,

            "mean_lag_days":
                None,

            "min_lag_days":
                None,

            "max_lag_days":
                None,

            "median_response_m":
                None,

            "mean_response_m":
                None,

            "median_response_ratio":
                None,

            "maxima_correlation":
                None,
        }


    lag = _to_numeric(
        events[
            "lag_real_dias"
        ]
    ).dropna()


    response = _to_numeric(
        events[
            "respuesta_san_nicolas_m"
        ]
    ).dropna()


    ratio = _to_numeric(
        events[
            "respuesta_relativa"
        ]
    ).dropna()


    maxima_corr = _safe_corr(

        events[
            "max_corrientes_m"
        ],

        events[
            "max_san_nicolas_m"
        ],
    )


    return {

        "event_count":
            int(
                len(
                    events
                )
            ),

        "median_lag_days":
            (
                float(
                    lag.median()
                )
                if not lag.empty
                else None
            ),

        "mean_lag_days":
            (
                float(
                    lag.mean()
                )
                if not lag.empty
                else None
            ),

        "min_lag_days":
            (
                int(
                    lag.min()
                )
                if not lag.empty
                else None
            ),

        "max_lag_days":
            (
                int(
                    lag.max()
                )
                if not lag.empty
                else None
            ),

        "median_response_m":
            (
                float(
                    response.median()
                )
                if not response.empty
                else None
            ),

        "mean_response_m":
            (
                float(
                    response.mean()
                )
                if not response.empty
                else None
            ),

        "median_response_ratio":
            (
                float(
                    ratio.median()
                )
                if not ratio.empty
                else None
            ),

        "maxima_correlation":
            maxima_corr,
    }


# ============================================================
# ESTADO ACTUAL POR ESTACIÓN
# ============================================================

def obtener_estado_actual_estacion(
    dataset,
    station,
):

    level_col = LEVEL_COLUMNS[
        station
    ]


    if (
        level_col
        not in dataset.columns
    ):

        return {}


    valid = dataset[
        [
            "datetime",
            level_col,
        ]
    ].dropna()


    if valid.empty:

        return {}


    latest = valid.iloc[
        -1
    ]


    current_date = latest[
        "datetime"
    ]


    current_level = _safe_float(
        latest[
            level_col
        ]
    )


    before_3 = valid[
        valid[
            "datetime"
        ]
        <= (
            current_date
            - pd.Timedelta(
                days=3
            )
        )
    ]


    before_7 = valid[
        valid[
            "datetime"
        ]
        <= (
            current_date
            - pd.Timedelta(
                days=7
            )
        )
    ]


    delta_3 = np.nan
    delta_7 = np.nan


    if not before_3.empty:

        delta_3 = (
            current_level
            - _safe_float(
                before_3.iloc[
                    -1
                ][
                    level_col
                ]
            )
        )


    if not before_7.empty:

        delta_7 = (
            current_level
            - _safe_float(
                before_7.iloc[
                    -1
                ][
                    level_col
                ]
            )
        )


    speed_7 = (
        delta_7 / 7.0
        if np.isfinite(
            delta_7
        )
        else np.nan
    )


    state = "estable"


    if np.isfinite(
        delta_7
    ):

        if delta_7 >= 0.08:

            state = "creciente"

        elif delta_7 <= -0.08:

            state = "decreciente"


    result = {

        "station":
            station,

        "date":
            current_date,

        "level":
            current_level,

        "delta_3d":
            delta_3,

        "delta_7d":
            delta_7,

        "speed_7d":
            speed_7,

        "state":
            state,
    }


    # ========================================================
    # CAUDAL
    # ========================================================

    q_col = FLOW_COLUMNS[
        station
    ]


    if q_col in dataset.columns:

        q_valid = dataset[
            [
                "datetime",
                q_col,
            ]
        ].dropna()


        if not q_valid.empty:

            result[
                "flow_m3s"
            ] = _safe_float(
                q_valid.iloc[
                    -1
                ][
                    q_col
                ]
            )


            q_recent = q_valid[
                q_valid[
                    "datetime"
                ]
                >= (
                    current_date
                    - pd.Timedelta(
                        days=7
                    )
                )
            ]


            if len(
                q_recent
            ) >= 2:

                result[
                    "flow_delta_7d"
                ] = (
                    _safe_float(
                        q_recent.iloc[
                            -1
                        ][
                            q_col
                        ]
                    )
                    -
                    _safe_float(
                        q_recent.iloc[
                            0
                        ][
                            q_col
                        ]
                    )
                )


    # ========================================================
    # LLUVIA
    # ========================================================

    rain_col = RAIN_COLUMNS[
        station
    ]


    if rain_col in dataset.columns:

        result[
            "rain_7d"
        ] = _rain_sum(

            dataset,

            rain_col,

            current_date
            - pd.Timedelta(
                days=6
            ),

            current_date,
        )


        result[
            "rain_15d"
        ] = _rain_sum(

            dataset,

            rain_col,

            current_date
            - pd.Timedelta(
                days=14
            ),

            current_date,
        )


        result[
            "rain_30d"
        ] = _rain_sum(

            dataset,

            rain_col,

            current_date
            - pd.Timedelta(
                days=29
            ),

            current_date,
        )


    return result


# ============================================================
# ESTADO ACTUAL DEL CORREDOR
# ============================================================

def obtener_estado_actual_corredor(
    dataset,
):

    rows = []


    for station in STATIONS:

        state = (
            obtener_estado_actual_estacion(
                dataset,
                station,
            )
        )


        if state:

            rows.append(
                state
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# ESTADO ACTUAL CORRIENTES
# compatibilidad
# ============================================================

def obtener_estado_actual_corrientes(
    dataset,
):

    return obtener_estado_actual_estacion(
        dataset,
        "Corrientes",
    )


# ============================================================
# DISTANCIA NORMALIZADA
# ============================================================

def _normalized_distance(
    historical,
    current,
):

    historical = _to_numeric(
        historical
    )


    valid = historical.dropna()


    if (
        valid.empty
        or not np.isfinite(
            current
        )
    ):

        return pd.Series(
            0.0,
            index=
                historical.index,
        )


    std = valid.std()


    if (
        not np.isfinite(
            std
        )
        or std < 1e-9
    ):

        std = 1.0


    return (
        abs(
            historical
            - current
        )
        / std
    )


# ============================================================
# BUSCAR EVENTOS SIMILARES
# ============================================================

def buscar_eventos_similares(
    events,
    current_state,
    top_n=SIMILAR_EVENTS_DEFAULT,
):

    if (
        events is None
        or events.empty
        or not current_state
    ):

        return pd.DataFrame()


    work = events.copy()


    distance = pd.Series(
        0.0,
        index=
            work.index,
    )


    total_weight = 0.0


    # ========================================================
    # NIVEL CORRIENTES
    # ========================================================

    current_level = _safe_float(
        current_state.get(
            "level"
        )
    )


    if (
        np.isfinite(
            current_level
        )
        and "max_corrientes_m"
        in work.columns
    ):

        weight = 1.0


        distance += (
            _normalized_distance(

                work[
                    "max_corrientes_m"
                ],

                current_level,
            )
            * weight
        )


        total_weight += weight


    # ========================================================
    # CRECIDA RECIENTE
    # ========================================================

    current_rise = _safe_float(
        current_state.get(
            "delta_7d"
        )
    )


    if (
        np.isfinite(
            current_rise
        )
        and "crecida_corrientes_m"
        in work.columns
    ):

        weight = 1.6


        distance += (
            _normalized_distance(

                work[
                    "crecida_corrientes_m"
                ],

                max(
                    current_rise,
                    0.0,
                ),
            )
            * weight
        )


        total_weight += weight


    # ========================================================
    # VELOCIDAD
    # ========================================================

    speed = _safe_float(
        current_state.get(
            "speed_7d"
        )
    )


    if (
        np.isfinite(
            speed
        )
        and "velocidad_corrientes_m_dia"
        in work.columns
    ):

        weight = 1.8


        distance += (
            _normalized_distance(

                work[
                    "velocidad_corrientes_m_dia"
                ],

                max(
                    speed,
                    0.0,
                ),
            )
            * weight
        )


        total_weight += weight


    # ========================================================
    # CAUDAL CORRIENTES
    # ========================================================

    flow = _safe_float(
        current_state.get(
            "flow_m3s"
        )
    )


    flow_col = (
        "q_corrientes_7d_mean"
    )


    if (
        np.isfinite(
            flow
        )
        and flow_col
        in work.columns
    ):

        weight = 1.2


        distance += (
            _normalized_distance(

                work[
                    flow_col
                ],

                flow,
            )
            * weight
        )


        total_weight += weight


    # ========================================================
    # LLUVIA CORRIENTES
    # ========================================================

    rain = _safe_float(
        current_state.get(
            "rain_15d"
        )
    )


    rain_col = (
        "rain_corrientes_15d"
    )


    if (
        np.isfinite(
            rain
        )
        and rain_col
        in work.columns
    ):

        weight = 0.8


        distance += (
            _normalized_distance(

                work[
                    rain_col
                ],

                rain,
            )
            * weight
        )


        total_weight += weight


    if total_weight <= 0:

        return pd.DataFrame()


    distance = (
        distance
        / total_weight
    )


    work[
        "similarity_distance"
    ] = distance


    work[
        "similarity_score"
    ] = (
        1.0
        / (
            1.0
            + distance
        )
    )


    return (
        work
        .sort_values(
            "similarity_distance"
        )
        .head(
            int(
                top_n
            )
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# ESTIMAR PROPAGACIÓN ACTUAL
# ============================================================

def estimar_demora_actual(
    events,
    current_state,
):

    similar = (
        buscar_eventos_similares(
            events,
            current_state,
            top_n=
                SIMILAR_EVENTS_DEFAULT,
        )
    )


    if similar.empty:

        return {
            "delay_days":
                None,

            "delay_min_days":
                None,

            "delay_max_days":
                None,

            "response_m":
                None,

            "response_ratio":
                None,

            "impact_date":
                None,

            "impact_from":
                None,

            "impact_to":
                None,

            "similar_events":
                similar,
        }


    lag = _to_numeric(
        similar[
            "lag_real_dias"
        ]
    ).dropna()


    response = _to_numeric(
        similar[
            "respuesta_san_nicolas_m"
        ]
    ).dropna()


    ratio = _to_numeric(
        similar[
            "respuesta_relativa"
        ]
    ).dropna()


    delay = (
        float(
            lag.median()
        )
        if not lag.empty
        else None
    )


    delay_min = (
        float(
            lag.quantile(
                0.25
            )
        )
        if not lag.empty
        else None
    )


    delay_max = (
        float(
            lag.quantile(
                0.75
            )
        )
        if not lag.empty
        else None
    )


    response_m = (
        float(
            response.median()
        )
        if not response.empty
        else None
    )


    response_ratio = (
        float(
            ratio.median()
        )
        if not ratio.empty
        else None
    )


    current_date = (
        current_state.get(
            "date"
        )
    )


    impact_date = None
    impact_from = None
    impact_to = None


    if current_date is not None:

        current_date = pd.to_datetime(
            current_date
        )


        if delay is not None:

            impact_date = (
                current_date
                + pd.Timedelta(
                    days=
                        int(
                            round(
                                delay
                            )
                        )
                )
            )


        if delay_min is not None:

            impact_from = (
                current_date
                + pd.Timedelta(
                    days=
                        int(
                            round(
                                delay_min
                            )
                        )
                )
            )


        if delay_max is not None:

            impact_to = (
                current_date
                + pd.Timedelta(
                    days=
                        int(
                            round(
                                delay_max
                            )
                        )
                )
            )


    return {

        "delay_days":
            delay,

        "delay_min_days":
            delay_min,

        "delay_max_days":
            delay_max,

        "response_m":
            response_m,

        "response_ratio":
            response_ratio,

        "impact_date":
            impact_date,

        "impact_from":
            impact_from,

        "impact_to":
            impact_to,

        "similar_events":
            similar,
    }


# ============================================================
# CREAR FEATURES HIDROLÓGICAS
# ============================================================

def crear_features_hidrologicas(
    dataset,
    corridor_lags=None,
):

    if (
        dataset is None
        or dataset.empty
    ):

        return pd.DataFrame()


    result = dataset.copy()


    result[
        "datetime"
    ] = _normalizar_datetime(
        result[
            "datetime"
        ]
    )


    # ========================================================
    # NIVEL SAN NICOLÁS
    # ========================================================

    if "nivel" in result.columns:

        nivel = _to_numeric(
            result[
                "nivel"
            ]
        )


        for lag in [
            1,
            2,
            3,
            5,
            7,
            10,
            14,
            21,
            30,
        ]:

            result[
                f"nivel_lag_{lag}"
            ] = nivel.shift(
                lag
            )


        result[
            "nivel_diff_1"
        ] = nivel.diff(
            1
        )


        result[
            "nivel_diff_3"
        ] = nivel.diff(
            3
        )


        result[
            "nivel_diff_7"
        ] = nivel.diff(
            7
        )


        result[
            "nivel_mean_7"
        ] = (
            nivel
            .rolling(
                7,
                min_periods=2,
            )
            .mean()
        )


        result[
            "nivel_mean_14"
        ] = (
            nivel
            .rolling(
                14,
                min_periods=3,
            )
            .mean()
        )


        result[
            "nivel_max_14"
        ] = (
            nivel
            .rolling(
                14,
                min_periods=3,
            )
            .max()
        )


        result[
            "nivel_min_14"
        ] = (
            nivel
            .rolling(
                14,
                min_periods=3,
            )
            .min()
        )


    # ========================================================
    # NIVELES AGUAS ARRIBA
    # ========================================================

    for station in UPSTREAM_STATIONS:

        col = LEVEL_COLUMNS[
            station
        ]


        if col not in result.columns:

            continue


        slug = _slug(
            station
        )


        values = _to_numeric(
            result[
                col
            ]
        )


        result[
            f"{col}_diff_1"
        ] = values.diff(
            1
        )


        result[
            f"{col}_diff_3"
        ] = values.diff(
            3
        )


        result[
            f"{col}_diff_7"
        ] = values.diff(
            7
        )


        result[
            f"{col}_mean_7"
        ] = (
            values
            .rolling(
                7,
                min_periods=2,
            )
            .mean()
        )


        result[
            f"{col}_trend_7"
        ] = (
            values
            - values.shift(
                7
            )
        )


        result[
            f"{col}_max_14"
        ] = (
            values
            .rolling(
                14,
                min_periods=3,
            )
            .max()
        )


        # ----------------------------------------------------
        # retardos generales
        # ----------------------------------------------------

        for lag in [
            1,
            3,
            5,
            7,
            10,
            14,
            21,
            30,
        ]:

            result[
                f"{col}_lag_{lag}"
            ] = values.shift(
                lag
            )


    # ========================================================
    # LAG APRENDIDO POR TRAMO
    # ========================================================

    if (
        corridor_lags is not None
        and isinstance(
            corridor_lags,
            pd.DataFrame,
        )
        and not corridor_lags.empty
    ):

        cumulative = 0


        for _, row in (
            corridor_lags.iterrows()
        ):

            station = row.get(
                "upstream"
            )


            lag = _safe_int(
                row.get(
                    "lag_days"
                ),
                0,
            )


            if (
                station is None
                or lag is None
            ):

                continue


            cumulative += max(
                lag,
                0,
            )


            if station not in LEVEL_COLUMNS:

                continue


            col = LEVEL_COLUMNS[
                station
            ]


            if col not in result.columns:

                continue


            cumulative_lag = int(
                min(
                    cumulative,
                    MAX_LAG_DAYS,
                )
            )


            result[
                f"{col}_lag_propagacion"
            ] = (
                _to_numeric(
                    result[
                        col
                    ]
                )
                .shift(
                    cumulative_lag
                )
            )


    # ========================================================
    # CAUDALES
    # ========================================================

    for station in STATIONS:

        q_col = FLOW_COLUMNS[
            station
        ]


        if q_col not in result.columns:

            continue


        q = _to_numeric(
            result[
                q_col
            ]
        )


        result[
            f"{q_col}_diff_1"
        ] = q.diff(
            1
        )


        result[
            f"{q_col}_diff_3"
        ] = q.diff(
            3
        )


        result[
            f"{q_col}_diff_7"
        ] = q.diff(
            7
        )


        result[
            f"{q_col}_mean_3"
        ] = (
            q
            .rolling(
                3,
                min_periods=1,
            )
            .mean()
        )


        result[
            f"{q_col}_mean_7"
        ] = (
            q
            .rolling(
                7,
                min_periods=2,
            )
            .mean()
        )


        result[
            f"{q_col}_mean_14"
        ] = (
            q
            .rolling(
                14,
                min_periods=3,
            )
            .mean()
        )


        result[
            f"{q_col}_trend_7"
        ] = (
            q
            - q.shift(
                7
            )
        )


        for lag in [
            1,
            3,
            5,
            7,
            10,
            14,
            21,
            30,
        ]:

            result[
                f"{q_col}_lag_{lag}"
            ] = q.shift(
                lag
            )


    # ========================================================
    # LLUVIAS
    # ========================================================

    for station in STATIONS:

        rain_col = RAIN_COLUMNS[
            station
        ]


        if rain_col not in result.columns:

            continue


        rain = (
            _to_numeric(
                result[
                    rain_col
                ]
            )
            .fillna(
                0.0
            )
        )


        result[
            f"{rain_col}_3d"
        ] = (
            rain
            .rolling(
                3,
                min_periods=1,
            )
            .sum()
        )


        result[
            f"{rain_col}_7d"
        ] = (
            rain
            .rolling(
                7,
                min_periods=1,
            )
            .sum()
        )


        result[
            f"{rain_col}_15d"
        ] = (
            rain
            .rolling(
                15,
                min_periods=1,
            )
            .sum()
        )


        result[
            f"{rain_col}_30d"
        ] = (
            rain
            .rolling(
                30,
                min_periods=1,
            )
            .sum()
        )


    # ========================================================
    # SEÑAL INTEGRADA AGUAS ARRIBA
    # ========================================================

    trend_cols = [
        c
        for c in result.columns
        if (
            c.startswith(
                "nivel_"
            )
            and c.endswith(
                "_diff_7"
            )
        )
    ]


    if trend_cols:

        result[
            "upstream_level_signal"
        ] = (
            result[
                trend_cols
            ]
            .mean(
                axis=1,
                skipna=True,
            )
        )


    flow_trend_cols = [
        c
        for c in result.columns
        if (
            c.startswith(
                "q_"
            )
            and c.endswith(
                "_trend_7"
            )
        )
    ]


    if flow_trend_cols:

        normalized = []


        for col in flow_trend_cols:

            q_base_col = (
                col.replace(
                    "_trend_7",
                    "",
                )
            )


            if q_base_col not in result.columns:

                continue


            ratio = (
                _to_numeric(
                    result[
                        col
                    ]
                )
                /
                _to_numeric(
                    result[
                        q_base_col
                    ]
                )
                .replace(
                    0,
                    np.nan,
                )
            )


            normalized.append(
                ratio
            )


        if normalized:

            result[
                "upstream_flow_signal"
            ] = pd.concat(
                normalized,
                axis=1,
            ).mean(
                axis=1,
                skipna=True,
            )


    rain_15_cols = [
        f"{RAIN_COLUMNS[s]}_15d"
        for s in STATIONS
        if (
            f"{RAIN_COLUMNS[s]}_15d"
            in result.columns
        )
    ]


    if rain_15_cols:

        result[
            "corridor_rain_15d"
        ] = (
            result[
                rain_15_cols
            ]
            .mean(
                axis=1,
                skipna=True,
            )
        )


        result[
            "corridor_rain_15d_max"
        ] = (
            result[
                rain_15_cols
            ]
            .max(
                axis=1,
                skipna=True,
            )
        )


    return result


# ============================================================
# RESUMEN DE MÁXIMOS
# ============================================================

def resumen_eventos_maximos(
    events,
    n=10,
):

    if (
        events is None
        or events.empty
    ):

        return pd.DataFrame()


    cols = [

        "fecha_max_corrientes",
        "max_corrientes_m",
        "crecida_corrientes_m",
        "fecha_max_san_nicolas",
        "max_san_nicolas_m",
        "respuesta_san_nicolas_m",
        "lag_real_dias",

    ]


    cols = [
        c
        for c in cols
        if c in events.columns
    ]


    return (
        events
        .sort_values(
            "max_corrientes_m",
            ascending=False,
        )
        .head(
            int(
                n
            )
        )[
            cols
        ]
        .reset_index(
            drop=True
        )
    )


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def analizar_corrientes_san_nicolas(
    df,
    upstream_history=None,
    exog_history=None,
    usar_historial_completo=True,
):

    # ========================================================
    # DATASET
    # ========================================================

    dataset = (
        construir_dataset_hidrologico(

            df,

            upstream_history=
                upstream_history,

            exog_history=
                exog_history,

            usar_historial_completo=
                usar_historial_completo,
        )
    )


    if dataset.empty:

        return {

            "version":
                VERSION,

            "status":
                "sin_datos",

            "history_source":
                None,

            "features":
                pd.DataFrame(),

            "events":
                pd.DataFrame(),

            "statistics":
                {},

            "lag":
                {},

            "corridor_lags":
                pd.DataFrame(),

            "current_state":
                pd.DataFrame(),

            "current_estimate":
                {},

            "similar_events":
                pd.DataFrame(),
        }


    # ========================================================
    # HISTORIAL
    # ========================================================

    valid_sn = dataset[
        [
            "datetime",
            "nivel",
        ]
    ].dropna()


    history_start = (
        valid_sn[
            "datetime"
        ].min()
        if not valid_sn.empty
        else None
    )


    history_end = (
        valid_sn[
            "datetime"
        ].max()
        if not valid_sn.empty
        else None
    )


    historical_years = None


    if (
        history_start is not None
        and history_end is not None
    ):

        historical_years = (
            (
                history_end
                - history_start
            ).days
            / 365.25
        )


    # ========================================================
    # RETARDOS POR TRAMO
    # ========================================================

    corridor_lags = (
        calcular_retardos_corredor(
            dataset
        )
    )


    # ========================================================
    # CORRIENTES -> SAN NICOLÁS
    # ========================================================

    lag = (
        calcular_lag_corrientes_san_nicolas(
            dataset
        )
    )


    expected_lag = lag.get(
        "best_lag_days"
    )


    if expected_lag is None:

        # ----------------------------------------------------
        # intentar sumar retardos por tramo
        # ----------------------------------------------------

        valid_segment_lags = (
            _to_numeric(
                corridor_lags[
                    "lag_days"
                ]
            )
            .dropna()
            if (
                not corridor_lags.empty
                and "lag_days"
                in corridor_lags.columns
            )
            else pd.Series(
                dtype=float
            )
        )


        if not valid_segment_lags.empty:

            expected_lag = int(
                min(
                    valid_segment_lags.sum(),
                    MAX_LAG_DAYS,
                )
            )

        else:

            expected_lag = 15


    # ========================================================
    # EVENTOS HISTÓRICOS
    # ========================================================

    events = (
        construir_eventos_corrientes_san_nicolas(

            dataset,

            expected_lag=
                expected_lag,
        )
    )


    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

    statistics = (
        estadisticas_propagacion(
            events
        )
    )


    # ========================================================
    # ESTADO ACTUAL
    # ========================================================

    current_state = (
        obtener_estado_actual_corredor(
            dataset
        )
    )


    current_corrientes = (
        obtener_estado_actual_corrientes(
            dataset
        )
    )


    # ========================================================
    # ESTIMACIÓN ACTUAL
    # ========================================================

    current_estimate = (
        estimar_demora_actual(

            events,

            current_corrientes,
        )
    )


    similar_events = (
        current_estimate.get(
            "similar_events",
            pd.DataFrame(),
        )
    )


    # ========================================================
    # FEATURES
    # ========================================================

    features = (
        crear_features_hidrologicas(

            dataset,

            corridor_lags=
                corridor_lags,
        )
    )


    # ========================================================
    # TOP EVENTOS
    # ========================================================

    top_events = (
        resumen_eventos_maximos(
            events,
            10,
        )
    )


    # ========================================================
    # COMPATIBILIDAD CON APP ANTERIOR
    # ========================================================

    return {

        "version":
            VERSION,

        "status":
            "ok",

        "history_source":
            (
                "INA A5 serie 36 + "
                "estaciones aguas arriba + "
                "lluvias + caudales"
            ),

        "historical_start":
            history_start,

        "historical_end":
            history_end,

        "historical_years":
            historical_years,

        "san_nicolas_records":
            int(
                dataset[
                    "nivel"
                ]
                .notna()
                .sum()
            )
            if "nivel"
            in dataset.columns
            else 0,

        "corrientes_records":
            int(
                dataset[
                    "nivel_corrientes"
                ]
                .notna()
                .sum()
            )
            if "nivel_corrientes"
            in dataset.columns
            else 0,

        # ----------------------------------------------------
        # Datasets
        # ----------------------------------------------------

        "dataset":
            dataset,

        "features":
            features,

        # ----------------------------------------------------
        # Retardos
        # ----------------------------------------------------

        "lag":
            lag,

        "corridor_lags":
            corridor_lags,

        # ----------------------------------------------------
        # Eventos
        # ----------------------------------------------------

        "events":
            events,

        "statistics":
            statistics,

        "top_events":
            top_events,

        # ----------------------------------------------------
        # Estado actual
        # ----------------------------------------------------

        "current_state":
            current_state,

        "current_corrientes":
            current_corrientes,

        "current_estimate":
            current_estimate,

        "similar_events":
            similar_events,

        # ----------------------------------------------------
        # Compatibilidad app V11.9.x
        # ----------------------------------------------------

        "demora_probable_dias":
            current_estimate.get(
                "delay_days"
            ),

        "demora_min_dias":
            current_estimate.get(
                "delay_min_days"
            ),

        "demora_max_dias":
            current_estimate.get(
                "delay_max_days"
            ),

        "respuesta_probable_m":
            current_estimate.get(
                "response_m"
            ),

        "respuesta_relativa":
            current_estimate.get(
                "response_ratio"
            ),

        "fecha_impacto_probable":
            current_estimate.get(
                "impact_date"
            ),

        "fecha_impacto_desde":
            current_estimate.get(
                "impact_from"
            ),

        "fecha_impacto_hasta":
            current_estimate.get(
                "impact_to"
            ),
    }


# ============================================================
# ALIAS GENERAL
# ============================================================

def analyze_hydrology(
    df,
    upstream_history=None,
    exog_history=None,
    usar_historial_completo=True,
):

    return analizar_corrientes_san_nicolas(

        df,

        upstream_history=
            upstream_history,

        exog_history=
            exog_history,

        usar_historial_completo=
            usar_historial_completo,
    )


# ============================================================
# DIAGNÓSTICO
# ============================================================

def diagnostic(
    df,
    upstream_history=None,
    exog_history=None,
):

    try:

        result = (
            analizar_corrientes_san_nicolas(

                df,

                upstream_history=
                    upstream_history,

                exog_history=
                    exog_history,

                usar_historial_completo=
                    True,
            )
        )

    except Exception as exc:

        return {

            "version":
                VERSION,

            "status":
                "error",

            "error":
                str(
                    exc
                ),
        }


    corridor_lags = result.get(
        "corridor_lags",
        pd.DataFrame(),
    )


    lag_rows = []


    if isinstance(
        corridor_lags,
        pd.DataFrame,
    ):

        for _, row in (
            corridor_lags.iterrows()
        ):

            lag_rows.append(
                {
                    "upstream":
                        row.get(
                            "upstream"
                        ),

                    "downstream":
                        row.get(
                            "downstream"
                        ),

                    "lag_days":
                        row.get(
                            "lag_days"
                        ),

                    "correlation":
                        row.get(
                            "correlation"
                        ),

                    "samples":
                        row.get(
                            "samples"
                        ),
                }
            )


    stats = result.get(
        "statistics",
        {},
    )


    current = result.get(
        "current_state",
        pd.DataFrame(),
    )


    current_rows = []


    if isinstance(
        current,
        pd.DataFrame,
    ):

        current_rows = (
            current
            .astype(
                str
            )
            .to_dict(
                orient="records"
            )
        )


    return {

        "version":
            VERSION,

        "status":
            result.get(
                "status"
            ),

        "history_source":
            result.get(
                "history_source"
            ),

        "historical_start":
            str(
                result.get(
                    "historical_start"
                )
            ),

        "historical_end":
            str(
                result.get(
                    "historical_end"
                )
            ),

        "historical_years":
            result.get(
                "historical_years"
            ),

        "san_nicolas_records":
            result.get(
                "san_nicolas_records"
            ),

        "corrientes_records":
            result.get(
                "corrientes_records"
            ),

        "corridor_lags":
            lag_rows,

        "corrientes_san_nicolas_lag":
            result.get(
                "lag",
                {},
            ).get(
                "best_lag_days"
            ),

        "corrientes_san_nicolas_correlation":
            result.get(
                "lag",
                {},
            ).get(
                "correlation"
            ),

        "event_count":
            stats.get(
                "event_count"
            ),

        "median_event_delay":
            stats.get(
                "median_lag_days"
            ),

        "median_response_m":
            stats.get(
                "median_response_m"
            ),

        "demora_probable_dias":
            result.get(
                "demora_probable_dias"
            ),

        "demora_min_dias":
            result.get(
                "demora_min_dias"
            ),

        "demora_max_dias":
            result.get(
                "demora_max_dias"
            ),

        "respuesta_probable_m":
            result.get(
                "respuesta_probable_m"
            ),

        "fecha_impacto_probable":
            str(
                result.get(
                    "fecha_impacto_probable"
                )
            ),

        "current_state":
            current_rows,

        "feature_count":
            (
                len(
                    result[
                        "features"
                    ].columns
                )
                if (
                    isinstance(
                        result.get(
                            "features"
                        ),
                        pd.DataFrame,
                    )
                    and not result[
                        "features"
                    ].empty
                )
                else 0
            ),
    }
