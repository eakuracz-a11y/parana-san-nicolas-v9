# ============================================================
# PARANÁ · SAN NICOLÁS
# src/hydrology.py
# V11.9.6 COMPLETO
#
# OBJETIVO
# ------------------------------------------------------------
# Analizar propagación hidrológica:
#
#       CORRIENTES  --->  SAN NICOLÁS
#
# utilizando:
#
# - todo el historial disponible
# - nivel diario Corrientes
# - nivel diario San Nicolás
# - máximos históricos
# - velocidad de creciente
# - demora entre máximos
# - respuesta de San Nicolás
# - caudal
# - precipitación
#
# Devuelve:
#
# - desfase histórico global
# - eventos históricos de creciente
# - demora media / mediana
# - demora mínima / máxima
# - eventos similares al estado actual
# - demora actual probable
# - rango probable de demora
# - respuesta probable en San Nicolás
#
# IMPORTANTE
# ------------------------------------------------------------
# Este archivo NO modifica src/ina.py.
#
# San Nicolás:
#     INA A5
#     series_id = 36
#
# Corrientes:
#     se recibe desde upstream_history
#
# ============================================================


from functools import lru_cache

import numpy as np
import pandas as pd
import requests


# ============================================================
# VERSIÓN
# ============================================================

VERSION = "V11.9.6"


# ============================================================
# INA
# ============================================================

INA_BASE_URL = (
    "https://alerta.ina.gob.ar/a5"
)

INA_OBSERVATIONS_URL = (
    INA_BASE_URL
    + "/getObservaciones"
)

INA_SERIES_GEOJSON_URL = (
    INA_BASE_URL
    + "/obs/puntual/series"
)

SAN_NICOLAS_SERIES_ID = 36


# ============================================================
# CONFIGURACIÓN
# ============================================================

REQUEST_TIMEOUT = 45

DEFAULT_HISTORY_START = (
    "1900-01-01"
)

HISTORY_BLOCK_YEARS = 5

DEFAULT_MAX_LAG = 20

MIN_EVENT_DISTANCE_DAYS = 8

EVENT_QUANTILE = 0.85

DEFAULT_EVENT_WINDOW_BEFORE = 2

DEFAULT_EVENT_WINDOW_AFTER = 20

MIN_EVENT_RISE = 0.10

CURRENT_LOOKBACK_DAYS = 7


# ============================================================
# REQUEST
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
            "Parana-San-Nicolas-Hydrology/11.9.6",

        "Accept":
            "application/json",
    }
)


# ============================================================
# UTILIDADES
# ============================================================

def _to_datetime_naive(
    series,
):

    return (
        pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(
            None
        )
    )


def _safe_datetime(
    value,
):

    dt = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(
        dt
    ):
        return pd.NaT

    return dt


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


def _to_numeric(
    series,
):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _safe_corr(
    a,
    b,
    min_samples=10,
):

    temp = pd.DataFrame(
        {
            "a":
                pd.to_numeric(
                    a,
                    errors="coerce",
                ),

            "b":
                pd.to_numeric(
                    b,
                    errors="coerce",
                ),
        }
    ).dropna()

    if len(
        temp
    ) < min_samples:

        return np.nan

    if (
        temp["a"].std()
        == 0
        or temp["b"].std()
        == 0
    ):

        return np.nan

    return float(
        temp[
            "a"
        ].corr(
            temp[
                "b"
            ]
        )
    )


# ============================================================
# NORMALIZACIÓN DIARIA
# ============================================================

def _normalize_daily(
    df,
    datetime_col="datetime",
    value_col="value",
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or datetime_col
        not in df.columns
        or value_col
        not in df.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )

    x = df[
        [
            datetime_col,
            value_col,
        ]
    ].copy()

    x.columns = [
        "datetime",
        "value",
    ]

    x[
        "datetime"
    ] = _to_datetime_naive(
        x[
            "datetime"
        ]
    )

    x[
        "value"
    ] = _to_numeric(
        x[
            "value"
        ]
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
                "value",
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
        .groupby(
            "datetime",
            as_index=False,
        )[
            "value"
        ]
        .median()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return x


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
                }
            ):

                return data

        for item in data:

            result = (
                _extract_records(
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
                    _extract_records(
                        data[
                            key
                        ]
                    )
                )

                if result:

                    return result

        for value in data.values():

            result = (
                _extract_records(
                    value
                )
            )

            if result:

                return result

    return []


def _normalizar_respuesta_ina(
    data,
):

    records = (
        _extract_records(
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

    date_fields = [
        "timestart",
        "datetime",
        "timestamp",
        "date",
        "fecha",
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
                ] is not None
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
                ] is not None
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

    result[
        "datetime"
    ] = _to_datetime_naive(
        result[
            "datetime"
        ]
    )

    result = (
        result
        .dropna()
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
# CATÁLOGO SAN NICOLÁS
# ============================================================

@lru_cache(
    maxsize=1
)
def _get_san_nicolas_catalog_dates():

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

    except Exception:

        return (
            pd.Timestamp(
                DEFAULT_HISTORY_START
            ),
            pd.Timestamp.today()
            .normalize(),
        )

    features = data.get(
        "features",
        [],
    )

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

        series_id = props.get(
            "series_id"
        )

        if series_id is None:

            series_id = feature.get(
                "id"
            )

        try:

            series_id = int(
                float(
                    series_id
                )
            )

        except Exception:

            continue

        if (
            series_id
            != SAN_NICOLAS_SERIES_ID
        ):

            continue

        start = pd.to_datetime(
            props.get(
                "timestart"
            ),
            errors="coerce",
            utc=True,
        )

        end = pd.to_datetime(
            props.get(
                "timeend"
            ),
            errors="coerce",
            utc=True,
        )

        if pd.isna(
            start
        ):

            start = pd.Timestamp(
                DEFAULT_HISTORY_START,
                tz="UTC",
            )

        if pd.isna(
            end
        ):

            end = pd.Timestamp.now(
                tz="UTC"
            )

        return (
            start.tz_localize(
                None
            ),
            end.tz_localize(
                None
            ),
        )

    return (
        pd.Timestamp(
            DEFAULT_HISTORY_START
        ),
        pd.Timestamp.today()
        .normalize(),
    )


# ============================================================
# CONSULTAR SAN NICOLÁS
# ============================================================

def _query_san_nicolas(
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

    if (
        pd.isna(
            start_dt
        )
        or pd.isna(
            end_dt
        )
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )

    response = SESSION.get(
        INA_OBSERVATIONS_URL,
        params={
            "tipo":
                "puntual",

            "series_id":
                SAN_NICOLAS_SERIES_ID,

            "timestart":
                start_dt.strftime(
                    "%Y-%m-%d"
                ),

            "timeend":
                end_dt.strftime(
                    "%Y-%m-%d"
                ),
        },
        timeout=
            REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return (
        _normalizar_respuesta_ina(
            response.json()
        )
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

    catalog_start, catalog_end = (
        _get_san_nicolas_catalog_dates()
    )

    if end_date is None:

        requested_end = (
            pd.Timestamp.today()
            .normalize()
        )

    else:

        requested_end = pd.to_datetime(
            end_date,
            errors="coerce",
        )

        if pd.isna(
            requested_end
        ):

            requested_end = (
                pd.Timestamp.today()
                .normalize()
            )

    end = min(
        catalog_end,
        requested_end,
    )

    start = catalog_start

    frames = []

    cursor = start

    while cursor <= end:

        block_end = (
            cursor
            + pd.DateOffset(
                years=
                    HISTORY_BLOCK_YEARS
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

            part = _query_san_nicolas(
                cursor,
                block_end,
            )

            if not part.empty:

                frames.append(
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

    result = _normalize_daily(
        result,
        datetime_col=
            "datetime",
        value_col=
            "value",
    )

    result = result.rename(
        columns={
            "value":
                "nivel_san_nicolas"
        }
    )

    return result


# ============================================================
# PREPARAR SAN NICOLÁS
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
                "nivel_san_nicolas",
            ]
        )

    if "nivel_san_nicolas" in df.columns:

        value_col = (
            "nivel_san_nicolas"
        )

    elif "nivel" in df.columns:

        value_col = (
            "nivel"
        )

    elif "value" in df.columns:

        value_col = (
            "value"
        )

    else:

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel_san_nicolas",
            ]
        )

    result = _normalize_daily(
        df,
        datetime_col=
            "datetime",
        value_col=
            value_col,
    )

    return result.rename(
        columns={
            "value":
                "nivel_san_nicolas"
        }
    )


# ============================================================
# PREPARAR CORRIENTES
# ============================================================

def preparar_corrientes(
    upstream_history,
):

    if (
        upstream_history is None
        or not isinstance(
            upstream_history,
            pd.DataFrame,
        )
        or upstream_history.empty
        or "datetime"
        not in upstream_history.columns
        or "nivel_corrientes"
        not in upstream_history.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel_corrientes",
            ]
        )

    result = _normalize_daily(
        upstream_history,
        datetime_col=
            "datetime",
        value_col=
            "nivel_corrientes",
    )

    return result.rename(
        columns={
            "value":
                "nivel_corrientes"
        }
    )


# ============================================================
# RELACIÓN BASE
# ============================================================

def construir_relacion_base(
    san_nicolas,
    upstream_history,
):

    sn = preparar_san_nicolas(
        san_nicolas
    )

    corr = preparar_corrientes(
        upstream_history
    )

    if (
        sn.empty
        or corr.empty
    ):

        return pd.DataFrame()

    result = corr.merge(
        sn,
        on="datetime",
        how="inner",
    )

    result = result.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # CAMBIOS
    # --------------------------------------------------------

    result[
        "delta_corrientes_1d"
    ] = (
        result[
            "nivel_corrientes"
        ].diff()
    )

    result[
        "delta_san_nicolas_1d"
    ] = (
        result[
            "nivel_san_nicolas"
        ].diff()
    )

    result[
        "corrientes_change_3d"
    ] = (
        result[
            "nivel_corrientes"
        ]
        - result[
            "nivel_corrientes"
        ].shift(3)
    )

    result[
        "corrientes_change_7d"
    ] = (
        result[
            "nivel_corrientes"
        ]
        - result[
            "nivel_corrientes"
        ].shift(7)
    )

    result[
        "san_nicolas_change_3d"
    ] = (
        result[
            "nivel_san_nicolas"
        ]
        - result[
            "nivel_san_nicolas"
        ].shift(3)
    )

    result[
        "san_nicolas_change_7d"
    ] = (
        result[
            "nivel_san_nicolas"
        ]
        - result[
            "nivel_san_nicolas"
        ].shift(7)
    )

    return result


# ============================================================
# LAG GLOBAL
#
# Se usa CAMBIO de nivel y no únicamente nivel absoluto.
#
# Esto reduce la influencia de:
# - estacionalidad
# - tendencias de largo plazo
# ============================================================

def calcular_lag_corrientes_san_nicolas(
    san_nicolas,
    upstream_history,
    max_lag=
        DEFAULT_MAX_LAG,
):

    sn = preparar_san_nicolas(
        san_nicolas
    )

    corr = preparar_corrientes(
        upstream_history
    )

    if (
        sn.empty
        or corr.empty
    ):

        return {
            "best_lag_days":
                np.nan,

            "correlation":
                np.nan,

            "samples":
                0,

            "lag_table":
                pd.DataFrame(),
        }

    merged = corr.merge(
        sn,
        on="datetime",
        how="inner",
    )

    if merged.empty:

        return {
            "best_lag_days":
                np.nan,

            "correlation":
                np.nan,

            "samples":
                0,

            "lag_table":
                pd.DataFrame(),
        }

    merged = merged.sort_values(
        "datetime"
    )

    # --------------------------------------------------------
    # CAMBIOS 3 DÍAS
    # --------------------------------------------------------

    merged[
        "corr_change"
    ] = (
        merged[
            "nivel_corrientes"
        ]
        - merged[
            "nivel_corrientes"
        ].shift(3)
    )

    merged[
        "sn_change"
    ] = (
        merged[
            "nivel_san_nicolas"
        ]
        - merged[
            "nivel_san_nicolas"
        ].shift(3)
    )

    rows = []

    for lag in range(
        0,
        int(
            max_lag
        ) + 1,
    ):

        # Corrientes hoy comparado con
        # San Nicolás "lag" días después.

        shifted_sn = (
            merged[
                "sn_change"
            ].shift(
                -lag
            )
        )

        corr_value = _safe_corr(
            merged[
                "corr_change"
            ],
            shifted_sn,
            min_samples=20,
        )

        valid = pd.DataFrame(
            {
                "a":
                    merged[
                        "corr_change"
                    ],

                "b":
                    shifted_sn,
            }
        ).dropna()

        rows.append(
            {
                "lag_days":
                    lag,

                "correlation":
                    corr_value,

                "samples":
                    len(
                        valid
                    ),
            }
        )

    lag_table = pd.DataFrame(
        rows
    )

    valid_table = lag_table.dropna(
        subset=[
            "correlation"
        ]
    )

    if valid_table.empty:

        return {
            "best_lag_days":
                np.nan,

            "correlation":
                np.nan,

            "samples":
                0,

            "lag_table":
                lag_table,
        }

    best_index = (
        valid_table[
            "correlation"
        ]
        .abs()
        .idxmax()
    )

    best = valid_table.loc[
        best_index
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
            lag_table,
    }


# ============================================================
# DETECTAR MÁXIMOS LOCALES
# ============================================================

def detectar_maximos_locales(
    df,
    value_col,
    min_distance_days=
        MIN_EVENT_DISTANCE_DAYS,
    quantile=
        EVENT_QUANTILE,
):

    if (
        df is None
        or df.empty
        or value_col
        not in df.columns
    ):

        return pd.DataFrame()

    x = df[
        [
            "datetime",
            value_col,
        ]
    ].copy()

    x[
        value_col
    ] = pd.to_numeric(
        x[
            value_col
        ],
        errors="coerce",
    )

    x = (
        x
        .dropna()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        x
    ) < 5:

        return pd.DataFrame()

    threshold = float(
        x[
            value_col
        ].quantile(
            quantile
        )
    )

    x[
        "prev"
    ] = (
        x[
            value_col
        ].shift(1)
    )

    x[
        "next"
    ] = (
        x[
            value_col
        ].shift(-1)
    )

    peaks = x[
        (
            x[
                value_col
            ]
            >= threshold
        )
        &
        (
            x[
                value_col
            ]
            >= x[
                "prev"
            ]
        )
        &
        (
            x[
                value_col
            ]
            >= x[
                "next"
            ]
        )
    ].copy()

    if peaks.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # SEPARAR EVENTOS
    # --------------------------------------------------------

    selected = []

    for _, row in peaks.iterrows():

        if not selected:

            selected.append(
                row
            )

            continue

        previous = selected[-1]

        distance = (
            row[
                "datetime"
            ]
            - previous[
                "datetime"
            ]
        ).days

        if (
            distance
            >= min_distance_days
        ):

            selected.append(
                row
            )

        else:

            # En el mismo episodio quedarse
            # con el máximo más alto.

            if (
                row[
                    value_col
                ]
                > previous[
                    value_col
                ]
            ):

                selected[
                    -1
                ] = row

    if not selected:

        return pd.DataFrame()

    result = pd.DataFrame(
        selected
    )

    return (
        result[
            [
                "datetime",
                value_col,
            ]
        ]
        .reset_index(
            drop=True
        )
    )


# ============================================================
# EVENTOS CORRIENTES -> SAN NICOLÁS
# ============================================================

def construir_eventos_corrientes_san_nicolas(
    san_nicolas,
    upstream_history,
    expected_lag=None,
    window_before=
        DEFAULT_EVENT_WINDOW_BEFORE,
    window_after=
        DEFAULT_EVENT_WINDOW_AFTER,
):

    sn = preparar_san_nicolas(
        san_nicolas
    )

    corr = preparar_corrientes(
        upstream_history
    )

    if (
        sn.empty
        or corr.empty
    ):

        return pd.DataFrame()

    if expected_lag is None:

        lag_info = (
            calcular_lag_corrientes_san_nicolas(
                sn,
                corr,
            )
        )

        expected_lag = (
            lag_info.get(
                "best_lag_days"
            )
        )

    if (
        expected_lag is None
        or not np.isfinite(
            _safe_float(
                expected_lag
            )
        )
    ):

        expected_lag = 8

    expected_lag = int(
        expected_lag
    )

    corr_peaks = (
        detectar_maximos_locales(
            corr,
            "nivel_corrientes",
        )
    )

    if corr_peaks.empty:

        return pd.DataFrame()

    events = []

    for _, peak in (
        corr_peaks.iterrows()
    ):

        corr_date = peak[
            "datetime"
        ]

        corr_level = float(
            peak[
                "nivel_corrientes"
            ]
        )

        # ----------------------------------------------------
        # NIVEL BASE CORRIENTES
        # ----------------------------------------------------

        corr_previous = corr[
            (
                corr[
                    "datetime"
                ]
                >= (
                    corr_date
                    - pd.Timedelta(
                        days=7
                    )
                )
            )
            &
            (
                corr[
                    "datetime"
                ]
                < corr_date
            )
        ]

        if corr_previous.empty:

            corr_base = np.nan
            corr_rise = np.nan
            corr_speed = np.nan

        else:

            corr_base = float(
                corr_previous[
                    "nivel_corrientes"
                ].mean()
            )

            corr_rise = (
                corr_level
                - corr_base
            )

            corr_speed = (
                corr_rise
                / max(
                    len(
                        corr_previous
                    ),
                    1,
                )
            )

        # Evitar máximos que no representen
        # un evento de creciente real.

        if (
            np.isfinite(
                corr_rise
            )
            and corr_rise
            < MIN_EVENT_RISE
        ):

            continue

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
                    window_before
            )
        )

        search_end = (
            target_date
            + pd.Timedelta(
                days=
                    window_after
            )
        )

        sn_window = sn[
            (
                sn[
                    "datetime"
                ]
                >= search_start
            )
            &
            (
                sn[
                    "datetime"
                ]
                <= search_end
            )
        ].copy()

        if sn_window.empty:

            continue

        idx = (
            sn_window[
                "nivel_san_nicolas"
            ]
            .idxmax()
        )

        sn_peak = (
            sn_window.loc[
                idx
            ]
        )

        sn_date = sn_peak[
            "datetime"
        ]

        sn_level = float(
            sn_peak[
                "nivel_san_nicolas"
            ]
        )

        real_lag = int(
            (
                sn_date
                - corr_date
            ).days
        )

        # ----------------------------------------------------
        # NIVEL BASE SAN NICOLÁS
        # ----------------------------------------------------

        sn_previous = sn[
            (
                sn[
                    "datetime"
                ]
                >= (
                    corr_date
                    - pd.Timedelta(
                        days=7
                    )
                )
            )
            &
            (
                sn[
                    "datetime"
                ]
                < corr_date
            )
        ]

        if sn_previous.empty:

            sn_base = np.nan
            sn_response = np.nan

        else:

            sn_base = float(
                sn_previous[
                    "nivel_san_nicolas"
                ].mean()
            )

            sn_response = (
                sn_level
                - sn_base
            )

        response_ratio = (
            sn_response
            / corr_rise
            if (
                np.isfinite(
                    sn_response
                )
                and np.isfinite(
                    corr_rise
                )
                and abs(
                    corr_rise
                ) > 0.01
            )
            else np.nan
        )

        events.append(
            {
                "fecha_max_corrientes":
                    corr_date,

                "max_corrientes_m":
                    corr_level,

                "nivel_base_corrientes_m":
                    corr_base,

                "crecida_corrientes_m":
                    corr_rise,

                "velocidad_corrientes_m_dia":
                    corr_speed,

                "fecha_max_san_nicolas":
                    sn_date,

                "max_san_nicolas_m":
                    sn_level,

                "nivel_base_san_nicolas_m":
                    sn_base,

                "respuesta_san_nicolas_m":
                    sn_response,

                "respuesta_relativa":
                    response_ratio,

                "lag_real_dias":
                    real_lag,
            }
        )

    if not events:

        return pd.DataFrame()

    result = pd.DataFrame(
        events
    )

    # --------------------------------------------------------
    # FILTRO DE DEMORAS FÍSICAMENTE ÚTILES
    # --------------------------------------------------------

    result = result[
        (
            result[
                "lag_real_dias"
            ]
            >= 0
        )
        &
        (
            result[
                "lag_real_dias"
            ]
            <= DEFAULT_MAX_LAG
            + DEFAULT_EVENT_WINDOW_AFTER
        )
    ].copy()

    return (
        result
        .sort_values(
            "fecha_max_corrientes"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# EXÓGENAS
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
        or "datetime"
        not in exog_history.columns
    ):

        return pd.DataFrame()

    x = exog_history.copy()

    x[
        "datetime"
    ] = _to_datetime_naive(
        x[
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

    keep = [
        "datetime"
    ]

    if (
        "caudal_m3s"
        in x.columns
    ):

        x[
            "caudal_m3s"
        ] = _to_numeric(
            x[
                "caudal_m3s"
            ]
        )

        keep.append(
            "caudal_m3s"
        )

    if (
        "precip_mm"
        in x.columns
    ):

        x[
            "precip_mm"
        ] = (
            _to_numeric(
                x[
                    "precip_mm"
                ]
            )
            .clip(
                lower=0
            )
        )

        keep.append(
            "precip_mm"
        )

    x = x[
        keep
    ].copy()

    agg = {}

    if (
        "caudal_m3s"
        in x.columns
    ):

        agg[
            "caudal_m3s"
        ] = "mean"

    if (
        "precip_mm"
        in x.columns
    ):

        agg[
            "precip_mm"
        ] = "sum"

    if not agg:

        return pd.DataFrame()

    return (
        x
        .groupby(
            "datetime",
            as_index=False,
        )
        .agg(
            agg
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# ENRIQUECER EVENTOS
# ============================================================

def enriquecer_eventos_con_exogenas(
    events,
    exog_history,
    rain_window_days=15,
    flow_window_days=7,
):

    if (
        events is None
        or not isinstance(
            events,
            pd.DataFrame,
        )
        or events.empty
    ):

        return pd.DataFrame()

    exog = preparar_exogenas(
        exog_history
    )

    result = events.copy()

    if exog.empty:

        result[
            "lluvia_previa_mm"
        ] = np.nan

        result[
            "caudal_medio_m3s"
        ] = np.nan

        result[
            "caudal_max_m3s"
        ] = np.nan

        return result

    rain_values = []
    flow_means = []
    flow_maxs = []

    for _, event in (
        result.iterrows()
    ):

        date = event[
            "fecha_max_corrientes"
        ]

        rain_start = (
            date
            - pd.Timedelta(
                days=
                    rain_window_days
            )
        )

        flow_start = (
            date
            - pd.Timedelta(
                days=
                    flow_window_days
            )
        )

        if (
            "precip_mm"
            in exog.columns
        ):

            rain = exog[
                (
                    exog[
                        "datetime"
                    ]
                    >= rain_start
                )
                &
                (
                    exog[
                        "datetime"
                    ]
                    <= date
                )
            ][
                "precip_mm"
            ]

            rain_values.append(
                float(
                    rain.sum()
                )
                if not rain.empty
                else np.nan
            )

        else:

            rain_values.append(
                np.nan
            )

        if (
            "caudal_m3s"
            in exog.columns
        ):

            flow = exog[
                (
                    exog[
                        "datetime"
                    ]
                    >= flow_start
                )
                &
                (
                    exog[
                        "datetime"
                    ]
                    <= date
                )
            ][
                "caudal_m3s"
            ].dropna()

            flow_means.append(
                float(
                    flow.mean()
                )
                if not flow.empty
                else np.nan
            )

            flow_maxs.append(
                float(
                    flow.max()
                )
                if not flow.empty
                else np.nan
            )

        else:

            flow_means.append(
                np.nan
            )

            flow_maxs.append(
                np.nan
            )

    result[
        "lluvia_previa_mm"
    ] = rain_values

    result[
        "caudal_medio_m3s"
    ] = flow_means

    result[
        "caudal_max_m3s"
    ] = flow_maxs

    return result


# ============================================================
# ESTADÍSTICAS
# ============================================================

def estadisticas_propagacion(
    events,
):

    if (
        events is None
        or not isinstance(
            events,
            pd.DataFrame,
        )
        or events.empty
    ):

        return {
            "eventos":
                0,

            "lag_mediana_dias":
                np.nan,

            "lag_promedio_dias":
                np.nan,

            "lag_min_dias":
                np.nan,

            "lag_max_dias":
                np.nan,

            "respuesta_mediana_m":
                np.nan,

            "respuesta_promedio_m":
                np.nan,

            "respuesta_relativa_mediana":
                np.nan,

            "correlacion_maximos":
                np.nan,
        }

    lag = pd.to_numeric(
        events[
            "lag_real_dias"
        ],
        errors="coerce",
    ).dropna()

    response = pd.to_numeric(
        events[
            "respuesta_san_nicolas_m"
        ],
        errors="coerce",
    ).dropna()

    relative = pd.to_numeric(
        events[
            "respuesta_relativa"
        ],
        errors="coerce",
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    corr = _safe_corr(
        events[
            "max_corrientes_m"
        ],
        events[
            "max_san_nicolas_m"
        ],
        min_samples=5,
    )

    return {
        "eventos":
            int(
                len(
                    events
                )
            ),

        "lag_mediana_dias":
            (
                float(
                    lag.median()
                )
                if not lag.empty
                else np.nan
            ),

        "lag_promedio_dias":
            (
                float(
                    lag.mean()
                )
                if not lag.empty
                else np.nan
            ),

        "lag_min_dias":
            (
                float(
                    lag.min()
                )
                if not lag.empty
                else np.nan
            ),

        "lag_max_dias":
            (
                float(
                    lag.max()
                )
                if not lag.empty
                else np.nan
            ),

        "respuesta_mediana_m":
            (
                float(
                    response.median()
                )
                if not response.empty
                else np.nan
            ),

        "respuesta_promedio_m":
            (
                float(
                    response.mean()
                )
                if not response.empty
                else np.nan
            ),

        "respuesta_relativa_mediana":
            (
                float(
                    relative.median()
                )
                if not relative.empty
                else np.nan
            ),

        "correlacion_maximos":
            corr,
    }


# ============================================================
# ESTADO ACTUAL CORRIENTES
# ============================================================

def obtener_estado_actual_corrientes(
    upstream_history,
    exog_history=None,
):

    corr = preparar_corrientes(
        upstream_history
    )

    result = {
        "fecha":
            None,

        "nivel_corrientes":
            np.nan,

        "variacion_3d":
            np.nan,

        "variacion_7d":
            np.nan,

        "velocidad_7d":
            np.nan,

        "estado":
            "Sin datos",

        "caudal_actual":
            np.nan,

        "lluvia_15d":
            np.nan,
    }

    if corr.empty:

        return result

    latest = corr.iloc[
        -1
    ]

    result[
        "fecha"
    ] = latest[
        "datetime"
    ]

    result[
        "nivel_corrientes"
    ] = float(
        latest[
            "nivel_corrientes"
        ]
    )

    if len(
        corr
    ) >= 4:

        result[
            "variacion_3d"
        ] = (
            float(
                corr[
                    "nivel_corrientes"
                ].iloc[-1]
            )
            - float(
                corr[
                    "nivel_corrientes"
                ].iloc[-4]
            )
        )

    if len(
        corr
    ) >= 8:

        delta7 = (
            float(
                corr[
                    "nivel_corrientes"
                ].iloc[-1]
            )
            - float(
                corr[
                    "nivel_corrientes"
                ].iloc[-8]
            )
        )

        result[
            "variacion_7d"
        ] = delta7

        result[
            "velocidad_7d"
        ] = (
            delta7
            / 7.0
        )

    delta7 = result[
        "variacion_7d"
    ]

    if np.isfinite(
        _safe_float(
            delta7
        )
    ):

        if delta7 > 0.08:

            result[
                "estado"
            ] = "Creciente"

        elif delta7 < -0.08:

            result[
                "estado"
            ] = "Decreciente"

        else:

            result[
                "estado"
            ] = "Estable"

    exog = preparar_exogenas(
        exog_history
    )

    if not exog.empty:

        current_date = result[
            "fecha"
        ]

        if (
            "caudal_m3s"
            in exog.columns
        ):

            flow = exog[
                exog[
                    "datetime"
                ]
                <= current_date
            ][
                "caudal_m3s"
            ].dropna()

            if not flow.empty:

                result[
                    "caudal_actual"
                ] = float(
                    flow.iloc[-1]
                )

        if (
            "precip_mm"
            in exog.columns
        ):

            rain_start = (
                current_date
                - pd.Timedelta(
                    days=15
                )
            )

            rain = exog[
                (
                    exog[
                        "datetime"
                    ]
                    >= rain_start
                )
                &
                (
                    exog[
                        "datetime"
                    ]
                    <= current_date
                )
            ][
                "precip_mm"
            ]

            if not rain.empty:

                result[
                    "lluvia_15d"
                ] = float(
                    rain.sum()
                )

    return result


# ============================================================
# EVENTOS SIMILARES
# ============================================================

def buscar_eventos_similares(
    events,
    current_corrientes=None,
    current_change=None,
    current_speed=None,
    current_flow=None,
    recent_rain=None,
    top_n=10,
):

    if (
        events is None
        or not isinstance(
            events,
            pd.DataFrame,
        )
        or events.empty
    ):

        return pd.DataFrame()

    x = events.copy()

    comparisons = []

    candidates = [
        (
            "max_corrientes_m",
            current_corrientes,
            1.0,
        ),

        (
            "crecida_corrientes_m",
            current_change,
            1.4,
        ),

        (
            "velocidad_corrientes_m_dia",
            current_speed,
            1.5,
        ),

        (
            "caudal_medio_m3s",
            current_flow,
            0.8,
        ),

        (
            "lluvia_previa_mm",
            recent_rain,
            0.5,
        ),
    ]

    for (
        column,
        target,
        weight,
    ) in candidates:

        if (
            column
            not in x.columns
            or target is None
            or not np.isfinite(
                _safe_float(
                    target
                )
            )
        ):

            continue

        values = pd.to_numeric(
            x[
                column
            ],
            errors="coerce",
        )

        valid = values.dropna()

        if len(
            valid
        ) < 3:

            continue

        scale = float(
            valid.std()
        )

        if (
            not np.isfinite(
                scale
            )
            or scale
            < 1e-8
        ):

            scale = max(
                abs(
                    float(
                        valid.mean()
                    )
                )
                * 0.1,
                0.01,
            )

        distance = (
            (
                values
                - float(
                    target
                )
            )
            .abs()
            / scale
        )

        comparisons.append(
            (
                distance,
                weight,
            )
        )

    if not comparisons:

        return pd.DataFrame()

    score = pd.Series(
        0.0,
        index=x.index,
    )

    weight_total = pd.Series(
        0.0,
        index=x.index,
    )

    for (
        distance,
        weight,
    ) in comparisons:

        valid = (
            distance.notna()
        )

        score.loc[
            valid
        ] += (
            distance.loc[
                valid
            ]
            * weight
        )

        weight_total.loc[
            valid
        ] += weight

    score = score / weight_total.replace(
        0,
        np.nan,
    )

    x[
        "similarity_score"
    ] = score

    x = x.dropna(
        subset=[
            "similarity_score"
        ]
    )

    if x.empty:

        return pd.DataFrame()

    return (
        x
        .sort_values(
            "similarity_score"
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
# ESTIMACIÓN DE DEMORA ACTUAL
# ============================================================

def estimar_demora_actual(
    events,
    upstream_history,
    exog_history=None,
    top_n=12,
):

    current = (
        obtener_estado_actual_corrientes(
            upstream_history,
            exog_history,
        )
    )

    current_level = current[
        "nivel_corrientes"
    ]

    current_change = current[
        "variacion_7d"
    ]

    current_speed = current[
        "velocidad_7d"
    ]

    current_flow = current[
        "caudal_actual"
    ]

    current_rain = current[
        "lluvia_15d"
    ]

    similares = (
        buscar_eventos_similares(
            events,
            current_corrientes=
                current_level,
            current_change=
                current_change,
            current_speed=
                current_speed,
            current_flow=
                current_flow,
            recent_rain=
                current_rain,
            top_n=
                top_n,
        )
    )

    # --------------------------------------------------------
    # USAR SIMILARES
    # --------------------------------------------------------

    if (
        similares is not None
        and not similares.empty
    ):

        lags = pd.to_numeric(
            similares[
                "lag_real_dias"
            ],
            errors="coerce",
        ).dropna()

        responses = pd.to_numeric(
            similares[
                "respuesta_san_nicolas_m"
            ],
            errors="coerce",
        ).dropna()

        ratios = pd.to_numeric(
            similares[
                "respuesta_relativa"
            ],
            errors="coerce",
        ).replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).dropna()

    else:

        lags = pd.to_numeric(
            events.get(
                "lag_real_dias",
                pd.Series(
                    dtype=float
                ),
            ),
            errors="coerce",
        ).dropna()

        responses = pd.to_numeric(
            events.get(
                "respuesta_san_nicolas_m",
                pd.Series(
                    dtype=float
                ),
            ),
            errors="coerce",
        ).dropna()

        ratios = pd.to_numeric(
            events.get(
                "respuesta_relativa",
                pd.Series(
                    dtype=float
                ),
            ),
            errors="coerce",
        ).replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).dropna()

    if lags.empty:

        return {
            "estado_corrientes":
                current,

            "eventos_similares":
                similares,

            "demora_probable_dias":
                np.nan,

            "demora_min_dias":
                np.nan,

            "demora_max_dias":
                np.nan,

            "respuesta_probable_m":
                np.nan,

            "respuesta_relativa":
                np.nan,

            "fecha_impacto_probable":
                None,

            "fecha_impacto_desde":
                None,

            "fecha_impacto_hasta":
                None,
        }

    delay_median = float(
        lags.median()
    )

    delay_min = float(
        lags.quantile(
            0.25
        )
    )

    delay_max = float(
        lags.quantile(
            0.75
        )
    )

    response = (
        float(
            responses.median()
        )
        if not responses.empty
        else np.nan
    )

    ratio = (
        float(
            ratios.median()
        )
        if not ratios.empty
        else np.nan
    )

    fecha_corr = current[
        "fecha"
    ]

    if (
        fecha_corr is not None
        and not pd.isna(
            fecha_corr
        )
    ):

        fecha_probable = (
            fecha_corr
            + pd.Timedelta(
                days=
                    round(
                        delay_median
                    )
            )
        )

        fecha_desde = (
            fecha_corr
            + pd.Timedelta(
                days=
                    round(
                        delay_min
                    )
            )
        )

        fecha_hasta = (
            fecha_corr
            + pd.Timedelta(
                days=
                    round(
                        delay_max
                    )
            )
        )

    else:

        fecha_probable = None
        fecha_desde = None
        fecha_hasta = None

    return {
        "estado_corrientes":
            current,

        "eventos_similares":
            similares,

        "demora_probable_dias":
            delay_median,

        "demora_min_dias":
            delay_min,

        "demora_max_dias":
            delay_max,

        "respuesta_probable_m":
            response,

        "respuesta_relativa":
            ratio,

        "fecha_impacto_probable":
            fecha_probable,

        "fecha_impacto_desde":
            fecha_desde,

        "fecha_impacto_hasta":
            fecha_hasta,
    }


# ============================================================
# RESUMEN DE EVENTOS MÁXIMOS
# ============================================================

def resumen_eventos_maximos(
    events,
    top_n=10,
):

    if (
        events is None
        or not isinstance(
            events,
            pd.DataFrame,
        )
        or events.empty
    ):

        return pd.DataFrame()

    columns = [
        "fecha_max_corrientes",
        "max_corrientes_m",
        "fecha_max_san_nicolas",
        "max_san_nicolas_m",
        "lag_real_dias",
        "crecida_corrientes_m",
        "respuesta_san_nicolas_m",
        "respuesta_relativa",
    ]

    available = [
        c
        for c in columns
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
                top_n
            )
        )[
            available
        ]
        .reset_index(
            drop=True
        )
    )


# ============================================================
# FEATURES HIDROLÓGICAS
# ============================================================

def crear_features_hidrologicas(
    san_nicolas,
    upstream_history=None,
    exog_history=None,
):

    sn = preparar_san_nicolas(
        san_nicolas
    )

    if sn.empty:

        return pd.DataFrame()

    x = sn.rename(
        columns={
            "nivel_san_nicolas":
                "nivel"
        }
    )

    corr = preparar_corrientes(
        upstream_history
    )

    if not corr.empty:

        x = x.merge(
            corr,
            on="datetime",
            how="left",
        )

    exog = preparar_exogenas(
        exog_history
    )

    if not exog.empty:

        x = x.merge(
            exog,
            on="datetime",
            how="left",
        )

    x = x.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # SAN NICOLÁS
    # --------------------------------------------------------

    for lag in [
        1,
        2,
        3,
        5,
        7,
        10,
        14,
    ]:

        x[
            f"nivel_sn_lag{lag}"
        ] = (
            x[
                "nivel"
            ].shift(
                lag
            )
        )

    x[
        "nivel_sn_diff1"
    ] = (
        x[
            "nivel"
        ].diff()
    )

    x[
        "nivel_sn_diff3"
    ] = (
        x[
            "nivel"
        ]
        - x[
            "nivel"
        ].shift(3)
    )

    x[
        "nivel_sn_media7"
    ] = (
        x[
            "nivel"
        ]
        .rolling(
            7
        )
        .mean()
    )

    # --------------------------------------------------------
    # CORRIENTES
    # --------------------------------------------------------

    if (
        "nivel_corrientes"
        in x.columns
    ):

        for lag in [
            1,
            3,
            5,
            7,
            10,
            14,
        ]:

            x[
                f"nivel_corrientes_lag{lag}"
            ] = (
                x[
                    "nivel_corrientes"
                ].shift(
                    lag
                )
            )

        x[
            "corrientes_diff1"
        ] = (
            x[
                "nivel_corrientes"
            ].diff()
        )

        x[
            "corrientes_diff3"
        ] = (
            x[
                "nivel_corrientes"
            ]
            - x[
                "nivel_corrientes"
            ].shift(3)
        )

        x[
            "corrientes_diff7"
        ] = (
            x[
                "nivel_corrientes"
            ]
            - x[
                "nivel_corrientes"
            ].shift(7)
        )

        x[
            "corrientes_max7"
        ] = (
            x[
                "nivel_corrientes"
            ]
            .rolling(
                7
            )
            .max()
        )

        x[
            "corrientes_max15"
        ] = (
            x[
                "nivel_corrientes"
            ]
            .rolling(
                15
            )
            .max()
        )

    # --------------------------------------------------------
    # CAUDAL
    # --------------------------------------------------------

    if (
        "caudal_m3s"
        in x.columns
    ):

        x[
            "caudal_diff1"
        ] = (
            x[
                "caudal_m3s"
            ].diff()
        )

        x[
            "caudal_diff3"
        ] = (
            x[
                "caudal_m3s"
            ]
            - x[
                "caudal_m3s"
            ].shift(3)
        )

        x[
            "caudal_media7"
        ] = (
            x[
                "caudal_m3s"
            ]
            .rolling(
                7
            )
            .mean()
        )

    # --------------------------------------------------------
    # LLUVIA
    # --------------------------------------------------------

    if (
        "precip_mm"
        in x.columns
    ):

        x[
            "lluvia_3d"
        ] = (
            x[
                "precip_mm"
            ]
            .rolling(
                3
            )
            .sum()
        )

        x[
            "lluvia_7d"
        ] = (
            x[
                "precip_mm"
            ]
            .rolling(
                7
            )
            .sum()
        )

        x[
            "lluvia_15d"
        ] = (
            x[
                "precip_mm"
            ]
            .rolling(
                15
            )
            .sum()
        )

    return x


# ============================================================
# ANÁLISIS PRINCIPAL
# ============================================================

def analizar_corrientes_san_nicolas(
    san_nicolas,
    upstream_history,
    exog_history=None,
    max_lag=
        DEFAULT_MAX_LAG,
    usar_historial_completo=True,
):

    # ========================================================
    # 1. SAN NICOLÁS
    # ========================================================

    sn_input = preparar_san_nicolas(
        san_nicolas
    )

    sn = sn_input.copy()

    history_source = (
        "periodo_recibido"
    )

    # --------------------------------------------------------
    # Intentar historial completo INA serie 36.
    # --------------------------------------------------------

    if (
        usar_historial_completo
        and not sn_input.empty
    ):

        try:

            end_date = (
                sn_input[
                    "datetime"
                ].max()
            )

            sn_full = (
                get_san_nicolas_full_history(
                    str(
                        end_date.date()
                    )
                )
            )

            if (
                isinstance(
                    sn_full,
                    pd.DataFrame,
                )
                and not sn_full.empty
            ):

                sn = sn_full

                history_source = (
                    "INA_A5_serie_36_completo"
                )

        except Exception:

            pass

    # ========================================================
    # 2. CORRIENTES
    # ========================================================

    corr = preparar_corrientes(
        upstream_history
    )

    # ========================================================
    # SIN DATOS
    # ========================================================

    if (
        sn.empty
        or corr.empty
    ):

        return {
            "version":
                VERSION,

            "status":
                "sin_datos",

            "history_source":
                history_source,

            "san_nicolas_records":
                len(
                    sn
                ),

            "corrientes_records":
                len(
                    corr
                ),

            "lag":
                {},

            "events":
                pd.DataFrame(),

            "statistics":
                {},

            "similar_events":
                pd.DataFrame(),

            "current_estimate":
                {},

            "features":
                pd.DataFrame(),
        }

    # ========================================================
    # 3. RANGO HISTÓRICO COINCIDENTE
    # ========================================================

    overlap_start = max(
        sn[
            "datetime"
        ].min(),
        corr[
            "datetime"
        ].min(),
    )

    overlap_end = min(
        sn[
            "datetime"
        ].max(),
        corr[
            "datetime"
        ].max(),
    )

    # ========================================================
    # 4. LAG GLOBAL
    # ========================================================

    lag = (
        calcular_lag_corrientes_san_nicolas(
            san_nicolas=
                sn,
            upstream_history=
                corr,
            max_lag=
                max_lag,
        )
    )

    expected_lag = lag.get(
        "best_lag_days"
    )

    # ========================================================
    # 5. EVENTOS
    # ========================================================

    events = (
        construir_eventos_corrientes_san_nicolas(
            san_nicolas=
                sn,
            upstream_history=
                corr,
            expected_lag=
                expected_lag,
        )
    )

    # ========================================================
    # 6. CAUDAL + LLUVIA
    # ========================================================

    if (
        not events.empty
        and exog_history is not None
    ):

        events = (
            enriquecer_eventos_con_exogenas(
                events,
                exog_history,
            )
        )

    # ========================================================
    # 7. ESTADÍSTICAS
    # ========================================================

    statistics = (
        estadisticas_propagacion(
            events
        )
    )

    # ========================================================
    # 8. ESTADO ACTUAL + DEMORA
    # ========================================================

    current_estimate = {}

    similar_events = pd.DataFrame()

    if not events.empty:

        current_estimate = (
            estimar_demora_actual(
                events=
                    events,
                upstream_history=
                    corr,
                exog_history=
                    exog_history,
                top_n=12,
            )
        )

        candidate = current_estimate.get(
            "eventos_similares"
        )

        if isinstance(
            candidate,
            pd.DataFrame,
        ):

            similar_events = candidate

    # ========================================================
    # 9. FEATURES
    # ========================================================

    features = (
        crear_features_hidrologicas(
            san_nicolas=
                sn,
            upstream_history=
                corr,
            exog_history=
                exog_history,
        )
    )

    # ========================================================
    # 10. SALIDA COMPATIBLE + NUEVA
    # ========================================================

    return {

        "version":
            VERSION,

        "status":
            "ok",

        "history_source":
            history_source,

        "historical_start":
            overlap_start,

        "historical_end":
            overlap_end,

        "historical_years":
            (
                (
                    overlap_end
                    - overlap_start
                ).days
                / 365.25
            ),

        "san_nicolas_records":
            int(
                len(
                    sn
                )
            ),

        "corrientes_records":
            int(
                len(
                    corr
                )
            ),

        # ----------------------------------------------------
        # Compatibilidad
        # ----------------------------------------------------

        "lag":
            lag,

        "events":
            events,

        "statistics":
            statistics,

        "top_events":
            resumen_eventos_maximos(
                events,
                top_n=15,
            ),

        "features":
            features,

        # ----------------------------------------------------
        # Nuevos resultados V11.9.6
        # ----------------------------------------------------

        "current_estimate":
            current_estimate,

        "similar_events":
            similar_events,

        "demora_probable_dias":
            current_estimate.get(
                "demora_probable_dias",
                np.nan,
            ),

        "demora_min_dias":
            current_estimate.get(
                "demora_min_dias",
                np.nan,
            ),

        "demora_max_dias":
            current_estimate.get(
                "demora_max_dias",
                np.nan,
            ),

        "respuesta_probable_m":
            current_estimate.get(
                "respuesta_probable_m",
                np.nan,
            ),

        "fecha_impacto_probable":
            current_estimate.get(
                "fecha_impacto_probable"
            ),

        "fecha_impacto_desde":
            current_estimate.get(
                "fecha_impacto_desde"
            ),

        "fecha_impacto_hasta":
            current_estimate.get(
                "fecha_impacto_hasta"
            ),
    }
