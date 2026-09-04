# ============================================================
# PARANÁ · SAN NICOLÁS
# src/hydrology.py
# V11.14 COMPLETO
#
# MOTOR HIDROLÓGICO
#
# Funciones:
# - Relación Corrientes -> San Nicolás
# - Retardos de propagación
# - Retardos por tramo
# - Eventos históricos
# - Eventos similares al estado actual
# - Presión hidrológica aguas arriba
# - Escenario probable
# - Escenario adverso
# - Escenario extremo histórico
#
# IMPORTANTE:
# Los escenarios adversos se construyen a partir de eventos
# históricos concurrentes. No se suman máximos independientes
# de lluvia, caudal y nivel pertenecientes a fechas diferentes.
#
# API PRINCIPAL:
#
# analizar_corrientes_san_nicolas(
#     sn_history,
#     upstream_history,
#     exog_history=None,
#     exog_future=None,
#     days=60,
#     **kwargs,
# )
#
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# VERSIÓN
# ============================================================

VERSION = "V11.14"


# ============================================================
# CONFIGURACIÓN
# ============================================================

MAX_SCENARIO_DAYS = 60

MIN_OVERLAP = 30

MIN_LAG_OVERLAP = 15

MIN_EVENT_OVERLAP = 5

MIN_EVENTS_FOR_SCENARIO = 3

MAX_LAG_DAYS = 60

DEFAULT_CORRIENTES_LAG = 20

EVENT_LOOKBACK_DAYS = 30

EVENT_FORWARD_DAYS = 60

SIMILAR_EVENT_LIMIT = 40


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


LEVEL_COLUMNS = {
    "Corrientes": "nivel_corrientes",
    "Goya": "nivel_goya",
    "La Paz": "nivel_la_paz",
    "Paraná": "nivel_parana",
    "Diamante": "nivel_diamante",
    "Rosario": "nivel_rosario",
    "Villa Constitución": "nivel_villa_constitucion",
    "San Nicolás": "nivel_san_nicolas",
}


FLOW_COLUMNS = {
    "Corrientes": "q_corrientes",
    "Goya": "q_goya",
    "La Paz": "q_la_paz",
    "Paraná": "q_parana",
    "Diamante": "q_diamante",
    "Rosario": "q_rosario",
    "Villa Constitución": "q_villa_constitucion",
    "San Nicolás": "q_san_nicolas",
}


RAIN_COLUMNS = {
    "Corrientes": "rain_corrientes",
    "Goya": "rain_goya",
    "La Paz": "rain_la_paz",
    "Paraná": "rain_parana",
    "Diamante": "rain_diamante",
    "Rosario": "rain_rosario",
    "Villa Constitución": "rain_villa_constitucion",
    "San Nicolás": "rain_san_nicolas",
}


CORRIDOR = [
    "Corrientes",
    "Goya",
    "La Paz",
    "Paraná",
    "Diamante",
    "Rosario",
    "Villa Constitución",
    "San Nicolás",
]


DEFAULT_SEGMENT_LAGS = {
    ("Corrientes", "Goya"): 4,
    ("Goya", "La Paz"): 4,
    ("La Paz", "Paraná"): 4,
    ("Paraná", "Diamante"): 2,
    ("Diamante", "Rosario"): 3,
    ("Rosario", "Villa Constitución"): 2,
    ("Villa Constitución", "San Nicolás"): 1,
}


STATION_WEIGHTS = {
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

def _numeric(values):
    return pd.to_numeric(
        values,
        errors="coerce",
    )


def _datetime_naive(values):
    return (
        pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
    )


def _safe_float(value, default=np.nan):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def _safe_int(value, default=None):
    try:
        value = int(round(float(value)))
        return value
    except Exception:
        return default


def _normalize_daily(df):
    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
        or "datetime" not in df.columns
    ):
        return pd.DataFrame()

    x = df.copy()

    x["datetime"] = _datetime_naive(
        x["datetime"]
    )

    x = x.dropna(
        subset=["datetime"]
    )

    x["datetime"] = (
        x["datetime"]
        .dt
        .normalize()
    )

    numeric_columns = []

    for col in x.columns:
        if col == "datetime":
            continue

        if (
            col == "nivel"
            or col == "value"
            or col.startswith("nivel_")
            or col.startswith("q_")
            or col.startswith("rain_")
            or col.startswith("precip_")
            or col.startswith("caudal")
            or col.endswith("_quality")
        ):
            x[col] = _numeric(
                x[col]
            )

            numeric_columns.append(
                col
            )

    if numeric_columns:
        x = (
            x.groupby(
                "datetime",
                as_index=False,
            )[numeric_columns]
            .mean()
        )
    else:
        x = (
            x[
                ["datetime"]
            ]
            .drop_duplicates()
        )

    return (
        x.sort_values("datetime")
        .reset_index(drop=True)
    )


# ============================================================
# PREPARAR SAN NICOLÁS
# ============================================================

def _prepare_sn(sn_history):
    x = _normalize_daily(
        sn_history
    )

    if x.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel_san_nicolas",
            ]
        )

    level_col = None

    for candidate in [
        "nivel_san_nicolas",
        "nivel",
        "value",
        "level",
    ]:
        if candidate in x.columns:
            level_col = candidate
            break

    if level_col is None:
        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel_san_nicolas",
            ]
        )

    result = x[
        [
            "datetime",
            level_col,
        ]
    ].copy()

    result["nivel_san_nicolas"] = _numeric(
        result[level_col]
    )

    result = result[
        (
            result["nivel_san_nicolas"] >= -5
        )
        &
        (
            result["nivel_san_nicolas"] <= 20
        )
    ]

    return (
        result[
            [
                "datetime",
                "nivel_san_nicolas",
            ]
        ]
        .dropna()
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# PREPARAR UPSTREAM
# ============================================================

def _prepare_upstream(upstream_history):
    x = _normalize_daily(
        upstream_history
    )

    if x.empty:
        return pd.DataFrame(
            columns=["datetime"]
        )

    keep = ["datetime"]

    for station in CORRIDOR[:-1]:
        col = LEVEL_COLUMNS[station]

        if col in x.columns:
            x[col] = _numeric(
                x[col]
            )

            keep.append(col)

    return (
        x[keep]
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# PREPARAR EXÓGENAS
# ============================================================

def _prepare_exog(exog):
    x = _normalize_daily(
        exog
    )

    if x.empty:
        return pd.DataFrame(
            columns=["datetime"]
        )

    return x


# ============================================================
# DATASET HIDROLÓGICO
# ============================================================

def build_hydrology_dataset(
    sn_history,
    upstream_history,
    exog_history=None,
):
    sn = _prepare_sn(
        sn_history
    )

    upstream = _prepare_upstream(
        upstream_history
    )

    exog = _prepare_exog(
        exog_history
    )

    if sn.empty:
        return pd.DataFrame()

    result = sn.copy()

    if not upstream.empty:
        result = result.merge(
            upstream,
            on="datetime",
            how="outer",
        )

    if not exog.empty:
        duplicate = [
            col
            for col in exog.columns
            if (
                col != "datetime"
                and col in result.columns
            )
        ]

        exog = exog.drop(
            columns=duplicate,
            errors="ignore",
        )

        result = result.merge(
            exog,
            on="datetime",
            how="outer",
        )

    result = (
        result.sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# ANOMALÍA NORMALIZADA
#
# Evita comparar directamente los ceros hidrométricos
# de distintas estaciones.
# ============================================================

def _normalized_anomaly(
    series,
    window=60,
):
    x = _numeric(
        series
    )

    median = (
        x.rolling(
            window,
            min_periods=max(
                10,
                window // 3,
            ),
        )
        .median()
    )

    scale = (
        x.rolling(
            window,
            min_periods=max(
                10,
                window // 3,
            ),
        )
        .std()
    )

    global_scale = _safe_float(
        x.std(),
        1.0,
    )

    if (
        not np.isfinite(global_scale)
        or global_scale <= 0
    ):
        global_scale = 1.0

    scale = scale.replace(
        0,
        np.nan,
    ).fillna(
        global_scale
    )

    anomaly = (
        x - median
    ) / scale

    return anomaly.clip(
        lower=-5.0,
        upper=5.0,
    )


# ============================================================
# CORRELACIÓN POSITIVA CON LAG
# ============================================================

def _lag_correlation(
    upstream,
    downstream,
    max_lag=MAX_LAG_DAYS,
    min_overlap=MIN_LAG_OVERLAP,
    min_lag=1,
):
    """
    Busca el retardo con mayor relación entre las VARIACIONES
    normalizadas de la estación aguas arriba y San Nicolás.

    V11.12:
    - evita lag=0 para Corrientes -> San Nicolás;
    - permite explorar hasta 60 días;
    - reduce la penalización artificial a retardos largos;
    - conserva todos los candidatos para diagnóstico.
    """
    upstream = _numeric(upstream)
    downstream = _numeric(downstream)

    up_anomaly = _normalized_anomaly(upstream)
    down_anomaly = _normalized_anomaly(downstream)

    up_signal = up_anomaly.diff()
    down_signal = down_anomaly.diff()

    rows = []

    for lag in range(int(min_lag), int(max_lag) + 1):
        shifted = up_signal.shift(lag)

        valid = shifted.notna() & down_signal.notna()
        overlap = int(valid.sum())

        if overlap < min_overlap:
            continue

        corr = shifted[valid].corr(down_signal[valid])
        corr = _safe_float(corr)

        if not np.isfinite(corr) or corr <= 0:
            continue

        # Respuesta física aproximada en m/m usando cambios diarios
        # de nivel en las escalas originales.
        up_delta = upstream.diff().shift(lag)
        down_delta = downstream.diff()
        valid_raw = up_delta.notna() & down_delta.notna()

        response = np.nan
        if int(valid_raw.sum()) >= min_overlap:
            ux = up_delta[valid_raw].to_numpy(dtype=float)
            dy = down_delta[valid_raw].to_numpy(dtype=float)
            var_x = float(np.var(ux))
            if np.isfinite(var_x) and var_x > 1e-10:
                response = float(
                    np.cov(ux, dy, ddof=0)[0, 1] / var_x
                )

        rows.append(
            {
                "lag_days": int(lag),
                "correlation": float(corr),
                "overlap": overlap,
                "response_m_per_m": response,
            }
        )

    if not rows:
        return {
            "lag_days": None,
            "correlation": np.nan,
            "overlap": 0,
            "response_m_per_m": np.nan,
            "table": pd.DataFrame(),
        }

    table = pd.DataFrame(rows)

    # Se prioriza la correlación; sólo se aplica una penalización
    # muy leve para evitar elegir retardos extremos casi empatados.
    table["score"] = (
        table["correlation"]
        - 0.0005 * table["lag_days"]
    )

    best = (
        table.sort_values(
            ["score", "correlation", "overlap"],
            ascending=[False, False, False],
        )
        .iloc[0]
    )

    return {
        "lag_days": int(best["lag_days"]),
        "correlation": float(best["correlation"]),
        "overlap": int(best["overlap"]),
        "response_m_per_m": _safe_float(
            best.get("response_m_per_m")
        ),
        "table": table,
    }


# ============================================================
# CORRIENTES -> SAN NICOLÁS AÑO POR AÑO
# ============================================================

def estimate_corrientes_yearly(
    dataset,
    max_lag=MAX_LAG_DAYS,
):
    """
    Calcula el mejor retardo Corrientes -> San Nicolás para cada año.

    La comparación utiliza variaciones normalizadas para no mezclar
    los distintos ceros hidrométricos de ambas estaciones.
    """
    required = {
        "datetime",
        LEVEL_COLUMNS["Corrientes"],
        LEVEL_COLUMNS["San Nicolás"],
    }

    if (
        dataset is None
        or dataset.empty
        or not required.issubset(dataset.columns)
    ):
        return pd.DataFrame()

    x = dataset[
        [
            "datetime",
            LEVEL_COLUMNS["Corrientes"],
            LEVEL_COLUMNS["San Nicolás"],
        ]
    ].copy()

    x["year"] = pd.to_datetime(
        x["datetime"],
        errors="coerce",
    ).dt.year

    rows = []

    for year in sorted(
        x["year"].dropna().astype(int).unique()
    ):
        annual = x[x["year"] == year].copy()

        if len(annual) < max(40, MIN_LAG_OVERLAP + 10):
            continue

        result = _lag_correlation(
            annual[LEVEL_COLUMNS["Corrientes"]],
            annual[LEVEL_COLUMNS["San Nicolás"]],
            max_lag=max_lag,
            min_overlap=MIN_LAG_OVERLAP,
            min_lag=1,
        )

        lag = result.get("lag_days")
        corr = _safe_float(result.get("correlation"))
        overlap = _safe_int(result.get("overlap"), 0)
        response = _safe_float(
            result.get("response_m_per_m")
        )

        if lag is None or not np.isfinite(corr):
            continue

        rows.append(
            {
                "year": int(year),
                "lag_days": int(lag),
                "correlation": float(corr),
                "response_m_per_m": response,
                "overlap": int(overlap),
            }
        )

    return pd.DataFrame(rows)


def summarize_corrientes_yearly(
    yearly,
):
    """
    Resume la relación anual mediante estadísticas robustas.
    """
    empty = {
        "delay_days": DEFAULT_CORRIENTES_LAG,
        "delay_min": max(1, DEFAULT_CORRIENTES_LAG - 8),
        "delay_max": min(MAX_LAG_DAYS, DEFAULT_CORRIENTES_LAG + 8),
        "correlation": np.nan,
        "response_m_per_m": np.nan,
        "years": 0,
        "quality_years": 0,
    }

    if (
        yearly is None
        or not isinstance(yearly, pd.DataFrame)
        or yearly.empty
    ):
        return empty

    x = yearly.copy()

    for col in [
        "lag_days",
        "correlation",
        "response_m_per_m",
        "overlap",
    ]:
        if col in x.columns:
            x[col] = _numeric(x[col])

    x = x.dropna(
        subset=["lag_days", "correlation"]
    )

    if x.empty:
        return empty

    # Años suficientemente informativos.
    quality = x[
        (x["overlap"] >= MIN_LAG_OVERLAP)
        & (x["correlation"] >= 0.15)
    ].copy()

    if len(quality) < 2:
        quality = x.copy()

    weights = (
        quality["correlation"].clip(lower=0.05)
        * np.sqrt(quality["overlap"].clip(lower=1))
    )

    # Mediana ponderada para evitar que un único año domine.
    order = np.argsort(
        quality["lag_days"].to_numpy(dtype=float)
    )
    lag_sorted = quality["lag_days"].to_numpy(dtype=float)[order]
    w_sorted = weights.to_numpy(dtype=float)[order]

    csum = np.cumsum(w_sorted)
    cutoff = 0.5 * csum[-1]
    robust_lag = float(
        lag_sorted[np.searchsorted(csum, cutoff)]
    )

    delay_min = int(
        np.clip(
            round(quality["lag_days"].quantile(0.20)),
            1,
            MAX_LAG_DAYS,
        )
    )
    delay_max = int(
        np.clip(
            round(quality["lag_days"].quantile(0.80)),
            1,
            MAX_LAG_DAYS,
        )
    )

    if delay_min > delay_max:
        delay_min, delay_max = delay_max, delay_min

    response_values = (
        quality["response_m_per_m"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    return {
        "delay_days": int(round(robust_lag)),
        "delay_min": delay_min,
        "delay_max": delay_max,
        "correlation": float(
            quality["correlation"].median()
        ),
        "response_m_per_m": (
            float(response_values.median())
            if not response_values.empty
            else np.nan
        ),
        "years": int(len(x)),
        "quality_years": int(len(quality)),
    }


# ============================================================
# RETARDOS POR TRAMO
# ============================================================

def estimate_corridor_lags(
    dataset,
):
    rows = []

    for i in range(
        len(CORRIDOR) - 1
    ):
        upstream_station = (
            CORRIDOR[i]
        )

        downstream_station = (
            CORRIDOR[i + 1]
        )

        upstream_col = (
            LEVEL_COLUMNS[
                upstream_station
            ]
        )

        downstream_col = (
            LEVEL_COLUMNS[
                downstream_station
            ]
        )

        fallback = (
            DEFAULT_SEGMENT_LAGS.get(
                (
                    upstream_station,
                    downstream_station,
                ),
                2,
            )
        )

        if (
            upstream_col not in dataset.columns
            or downstream_col not in dataset.columns
        ):
            rows.append(
                {
                    "upstream":
                        upstream_station,

                    "downstream":
                        downstream_station,

                    "lag_days":
                        fallback,

                    "correlation":
                        np.nan,

                    "overlap":
                        0,

                    "source":
                        "respaldo",
                }
            )

            continue

        result = _lag_correlation(
            dataset[
                upstream_col
            ],
            dataset[
                downstream_col
            ],
            max_lag=15,
            min_overlap=MIN_LAG_OVERLAP,
        )

        lag = result[
            "lag_days"
        ]

        if lag is None:
            lag = fallback
            source = "respaldo"
        else:
            lag = int(
                np.clip(
                    lag,
                    0,
                    15,
                )
            )

            source = "histórico"

        rows.append(
            {
                "upstream":
                    upstream_station,

                "downstream":
                    downstream_station,

                "lag_days":
                    lag,

                "correlation":
                    result[
                        "correlation"
                    ],

                "overlap":
                    result[
                        "overlap"
                    ],

                "source":
                    source,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# RETARDO DIRECTO DE CADA ESTACIÓN A SAN NICOLÁS
# ============================================================

def estimate_lag_to_sn(
    dataset,
    corridor_lags=None,
):
    rows = []

    sn_col = (
        LEVEL_COLUMNS[
            "San Nicolás"
        ]
    )

    if sn_col not in dataset.columns:
        return pd.DataFrame()

    segment_map = {}

    if (
        isinstance(
            corridor_lags,
            pd.DataFrame,
        )
        and not corridor_lags.empty
    ):
        for _, row in corridor_lags.iterrows():
            segment_map[
                (
                    row["upstream"],
                    row["downstream"],
                )
            ] = _safe_int(
                row["lag_days"],
                0,
            )

    for station in CORRIDOR[:-1]:
        col = LEVEL_COLUMNS[
            station
        ]

        direct = None

        if col in dataset.columns:
            direct = _lag_correlation(
                dataset[col],
                dataset[sn_col],
                max_lag=MAX_LAG_DAYS,
                min_overlap=MIN_LAG_OVERLAP,
            )

        direct_lag = (
            direct.get(
                "lag_days"
            )
            if isinstance(
                direct,
                dict,
            )
            else None
        )

        direct_corr = (
            direct.get(
                "correlation"
            )
            if isinstance(
                direct,
                dict,
            )
            else np.nan
        )

        direct_overlap = (
            direct.get(
                "overlap",
                0,
            )
            if isinstance(
                direct,
                dict,
            )
            else 0
        )

        # ----------------------------------------------------
        # Suma de retardos por tramo como respaldo físico.
        # ----------------------------------------------------

        index = CORRIDOR.index(
            station
        )

        segment_total = 0

        valid_segments = True

        for j in range(
            index,
            len(CORRIDOR) - 1,
        ):
            key = (
                CORRIDOR[j],
                CORRIDOR[j + 1],
            )

            if key not in segment_map:
                valid_segments = False
                break

            segment_total += (
                segment_map[key]
            )

        if (
            direct_lag is not None
            and direct_overlap
            >= MIN_LAG_OVERLAP
        ):
            lag = int(
                np.clip(
                    direct_lag,
                    0,
                    MAX_LAG_DAYS,
                )
            )

            source = (
                "correlación histórica"
            )

        elif valid_segments:
            lag = int(
                np.clip(
                    segment_total,
                    0,
                    MAX_LAG_DAYS,
                )
            )

            source = (
                "suma de tramos"
            )

        else:
            # ------------------------------------------------
            # Respaldo para cada punto a SN.
            # ------------------------------------------------

            fallback_map = {
                "Corrientes": 20,
                "Goya": 16,
                "La Paz": 12,
                "Paraná": 8,
                "Diamante": 6,
                "Rosario": 3,
                "Villa Constitución": 1,
            }

            lag = fallback_map[
                station
            ]

            source = "respaldo"

        rows.append(
            {
                "station":
                    station,

                "lag_days":
                    lag,

                "correlation":
                    direct_corr,

                "overlap":
                    direct_overlap,

                "source":
                    source,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DETECCIÓN DE EVENTOS EN SAN NICOLÁS
# ============================================================

def detect_historical_events(
    dataset,
):
    if (
        dataset is None
        or dataset.empty
        or "nivel_san_nicolas"
        not in dataset.columns
    ):
        return pd.DataFrame()

    x = dataset[
        [
            "datetime",
            "nivel_san_nicolas",
        ]
    ].copy()

    x["nivel_san_nicolas"] = _numeric(
        x["nivel_san_nicolas"]
    )

    x = x.dropna()

    if len(x) < 90:
        return pd.DataFrame()

    x = x.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    level = x[
        "nivel_san_nicolas"
    ]

    change_7 = level.diff(
        7
    )

    rolling_std = (
        change_7.rolling(
            180,
            min_periods=30,
        )
        .std()
    )

    threshold = (
        rolling_std
        * 1.25
    )

    global_threshold = (
        change_7.std()
        * 1.25
    )

    if (
        not np.isfinite(
            global_threshold
        )
        or global_threshold < 0.08
    ):
        global_threshold = 0.08

    threshold = (
        threshold
        .fillna(
            global_threshold
        )
        .clip(
            lower=0.08
        )
    )

    candidate_mask = (
        change_7
        >= threshold
    )

    candidate_indices = np.where(
        candidate_mask
        .fillna(False)
        .to_numpy()
    )[0]

    if len(candidate_indices) == 0:
        return pd.DataFrame()

    # Evitar contar todos los días de una misma creciente
    # como eventos separados.
    event_indices = []

    last_index = -999

    for idx in candidate_indices:
        if idx - last_index >= 20:
            event_indices.append(
                idx
            )
            last_index = idx

    rows = []

    for idx in event_indices:
        start_idx = max(
            0,
            idx - 7,
        )

        end_idx = min(
            len(x) - 1,
            idx + EVENT_FORWARD_DAYS,
        )

        window = x.iloc[
            start_idx:
            end_idx + 1
        ].copy()

        if len(window) < 5:
            continue

        start_level = _safe_float(
            x.iloc[
                start_idx
            ][
                "nivel_san_nicolas"
            ]
        )

        if not np.isfinite(
            start_level
        ):
            continue

        peak_index = (
            window[
                "nivel_san_nicolas"
            ]
            .idxmax()
        )

        peak_row = x.loc[
            peak_index
        ]

        peak_level = _safe_float(
            peak_row[
                "nivel_san_nicolas"
            ]
        )

        if not np.isfinite(
            peak_level
        ):
            continue

        rise = (
            peak_level
            - start_level
        )

        if rise < 0.08:
            continue

        start_date = pd.Timestamp(
            x.iloc[
                start_idx
            ][
                "datetime"
            ]
        )

        peak_date = pd.Timestamp(
            peak_row[
                "datetime"
            ]
        )

        rise_days = int(
            (
                peak_date
                - start_date
            ).days
        )

        rows.append(
            {
                "event_id":
                    len(rows) + 1,

                "start_date":
                    start_date,

                "trigger_date":
                    pd.Timestamp(
                        x.iloc[idx][
                            "datetime"
                        ]
                    ),

                "peak_date":
                    peak_date,

                "start_level_sn":
                    start_level,

                "peak_level_sn":
                    peak_level,

                "rise_sn":
                    rise,

                "rise_days":
                    rise_days,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ESTADÍSTICAS DE UN EVENTO
# ============================================================

def _event_features(
    dataset,
    event_row,
):
    start_date = pd.Timestamp(
        event_row[
            "start_date"
        ]
    )

    antecedent_start = (
        start_date
        - pd.Timedelta(
            days=15
        )
    )

    antecedent = dataset[
        (
            dataset[
                "datetime"
            ]
            >= antecedent_start
        )
        &
        (
            dataset[
                "datetime"
            ]
            <= start_date
        )
    ].copy()

    features = {
        "event_id":
            event_row[
                "event_id"
            ],

        "start_date":
            start_date,

        "peak_date":
            event_row[
                "peak_date"
            ],

        "start_level_sn":
            event_row[
                "start_level_sn"
            ],

        "peak_level_sn":
            event_row[
                "peak_level_sn"
            ],

        "rise_sn":
            event_row[
                "rise_sn"
            ],

        "rise_days":
            event_row[
                "rise_days"
            ],
    }

    # ========================================================
    # NIVELES
    # ========================================================

    for station in CORRIDOR[:-1]:
        col = LEVEL_COLUMNS[
            station
        ]

        if col not in antecedent.columns:
            continue

        values = (
            _numeric(
                antecedent[col]
            )
            .dropna()
        )

        if values.empty:
            continue

        features[
            f"level_{station}"
        ] = float(
            values.iloc[-1]
        )

        if len(values) >= 8:
            features[
                f"level_change7_{station}"
            ] = float(
                values.iloc[-1]
                - values.iloc[-8]
            )

    # ========================================================
    # CAUDALES
    # ========================================================

    for station in STATIONS:
        col = FLOW_COLUMNS[
            station
        ]

        if col not in antecedent.columns:
            continue

        q = (
            _numeric(
                antecedent[col]
            )
            .dropna()
        )

        if q.empty:
            continue

        features[
            f"flow_{station}"
        ] = float(
            q.iloc[-1]
        )

        if len(q) >= 8:
            previous = float(
                q.iloc[-8]
            )

            if previous > 0:
                features[
                    f"flow_change7_{station}"
                ] = (
                    float(
                        q.iloc[-1]
                    )
                    - previous
                ) / previous

    # ========================================================
    # LLUVIAS
    # ========================================================

    for station in STATIONS:
        col = RAIN_COLUMNS[
            station
        ]

        if col not in antecedent.columns:
            continue

        rain = (
            _numeric(
                antecedent[col]
            )
            .fillna(0.0)
        )

        features[
            f"rain15_{station}"
        ] = float(
            rain.sum()
        )

    return features


# ============================================================
# TABLA DE EVENTOS ENRIQUECIDA
# ============================================================

def build_event_table(
    dataset,
    events,
):
    if (
        events is None
        or events.empty
    ):
        return pd.DataFrame()

    rows = []

    for _, event in events.iterrows():
        rows.append(
            _event_features(
                dataset,
                event,
            )
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ESTADO ACTUAL
# ============================================================

def _current_features(
    dataset,
):
    if (
        dataset is None
        or dataset.empty
    ):
        return {}

    last_date = dataset[
        "datetime"
    ].max()

    recent = dataset[
        (
            dataset[
                "datetime"
            ]
            >= (
                last_date
                - pd.Timedelta(
                    days=15
                )
            )
        )
        &
        (
            dataset[
                "datetime"
            ]
            <= last_date
        )
    ].copy()

    result = {}

    sn = (
        _numeric(
            recent[
                "nivel_san_nicolas"
            ]
        )
        .dropna()
        if (
            "nivel_san_nicolas"
            in recent.columns
        )
        else pd.Series(
            dtype=float
        )
    )

    if not sn.empty:
        result[
            "start_level_sn"
        ] = float(
            sn.iloc[-1]
        )

    for station in CORRIDOR[:-1]:
        col = LEVEL_COLUMNS[
            station
        ]

        if col not in recent.columns:
            continue

        values = (
            _numeric(
                recent[col]
            )
            .dropna()
        )

        if values.empty:
            continue

        result[
            f"level_{station}"
        ] = float(
            values.iloc[-1]
        )

        if len(values) >= 8:
            result[
                f"level_change7_{station}"
            ] = float(
                values.iloc[-1]
                - values.iloc[-8]
            )

    for station in STATIONS:
        col = FLOW_COLUMNS[
            station
        ]

        if col not in recent.columns:
            continue

        values = (
            _numeric(
                recent[col]
            )
            .dropna()
        )

        if values.empty:
            continue

        result[
            f"flow_{station}"
        ] = float(
            values.iloc[-1]
        )

        if len(values) >= 8:
            previous = float(
                values.iloc[-8]
            )

            if previous > 0:
                result[
                    f"flow_change7_{station}"
                ] = (
                    float(
                        values.iloc[-1]
                    )
                    - previous
                ) / previous

    for station in STATIONS:
        col = RAIN_COLUMNS[
            station
        ]

        if col not in recent.columns:
            continue

        rain = (
            _numeric(
                recent[col]
            )
            .fillna(0.0)
        )

        result[
            f"rain15_{station}"
        ] = float(
            rain.sum()
        )

    return result


# ============================================================
# EVENTOS SIMILARES
# ============================================================

def find_similar_events(
    event_table,
    current_features,
    limit=SIMILAR_EVENT_LIMIT,
):
    if (
        event_table is None
        or event_table.empty
        or not current_features
    ):
        return pd.DataFrame()

    candidates = event_table.copy()

    feature_candidates = []

    # Se priorizan tendencias y condiciones del corredor.
    for col in candidates.columns:
        if (
            col.startswith(
                "level_change7_"
            )
            or col.startswith(
                "flow_change7_"
            )
            or col.startswith(
                "rain15_"
            )
            or col == "start_level_sn"
        ):
            if col in current_features:
                feature_candidates.append(
                    col
                )

    if not feature_candidates:
        return pd.DataFrame()

    distances = np.zeros(
        len(candidates),
        dtype=float,
    )

    used = np.zeros(
        len(candidates),
        dtype=float,
    )

    for col in feature_candidates:
        historical = _numeric(
            candidates[col]
        )

        current = _safe_float(
            current_features.get(
                col
            )
        )

        if not np.isfinite(
            current
        ):
            continue

        median = historical.median()

        scale = historical.std()

        if (
            not np.isfinite(scale)
            or scale <= 1e-9
        ):
            scale = historical.abs().median()

        if (
            not np.isfinite(scale)
            or scale <= 1e-9
        ):
            scale = 1.0

        valid = historical.notna()

        diff = (
            historical
            - current
        ) / scale

        distances[
            valid.to_numpy()
        ] += (
            diff[
                valid
            ].to_numpy()
            ** 2
        )

        used[
            valid.to_numpy()
        ] += 1.0

    valid_rows = (
        used >= 2
    )

    if not valid_rows.any():
        return pd.DataFrame()

    candidates = candidates.loc[
        valid_rows
    ].copy()

    distances = distances[
        valid_rows
    ]

    used = used[
        valid_rows
    ]

    candidates[
        "similarity_distance"
    ] = np.sqrt(
        distances
        / np.maximum(
            used,
            1.0,
        )
    )

    candidates[
        "similarity_weight"
    ] = 1.0 / (
        1.0
        +
        candidates[
            "similarity_distance"
        ]
    )

    candidates = (
        candidates.sort_values(
            [
                "similarity_distance",
                "start_date",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(limit)
        .reset_index(drop=True)
    )

    return candidates


# ============================================================
# PRESIÓN HIDROLÓGICA ACTUAL
# ============================================================

def calculate_hydrological_pressure(
    dataset,
    exog_future=None,
):
    if (
        dataset is None
        or dataset.empty
    ):
        return {
            "hydrological_pressure": 0.0,
            "level_signal": 0.0,
            "flow_signal": 0.0,
            "rain_signal": 0.0,
            "persistence_signal": 0.0,
        }

    last_date = dataset[
        "datetime"
    ].max()

    recent = dataset[
        dataset[
            "datetime"
        ]
        >= (
            last_date
            - pd.Timedelta(
                days=30
            )
        )
    ].copy()

    # ========================================================
    # NIVELES
    # ========================================================

    level_signals = []
    level_weights = []

    for station in CORRIDOR[:-1]:
        col = LEVEL_COLUMNS[
            station
        ]

        if col not in recent.columns:
            continue

        values = (
            _numeric(
                recent[col]
            )
            .dropna()
        )

        if len(values) < 4:
            continue

        window = min(
            8,
            len(values),
        )

        change = (
            float(
                values.iloc[-1]
            )
            -
            float(
                values.iloc[
                    -window
                ]
            )
        )

        signal = np.clip(
            change / 0.50,
            -1.5,
            2.0,
        )

        level_signals.append(
            signal
        )

        level_weights.append(
            STATION_WEIGHTS[
                station
            ]
        )

    if level_signals:
        level_signal = float(
            np.average(
                level_signals,
                weights=
                    level_weights,
            )
        )
    else:
        level_signal = 0.0

    # ========================================================
    # CAUDAL
    # ========================================================

    flow_signals = []
    flow_weights = []

    for station in STATIONS:
        col = FLOW_COLUMNS[
            station
        ]

        if col not in recent.columns:
            continue

        q = (
            _numeric(
                recent[col]
            )
            .dropna()
        )

        if len(q) < 4:
            continue

        current = float(
            q.iloc[-1]
        )

        baseline = float(
            q.tail(
                min(
                    14,
                    len(q),
                )
            ).median()
        )

        if baseline <= 0:
            continue

        signal = (
            current
            - baseline
        ) / baseline

        signal = float(
            np.clip(
                signal,
                -1.0,
                2.0,
            )
        )

        quality_col = (
            col
            + "_quality"
        )

        quality = 1.0

        if quality_col in recent.columns:
            quality_values = (
                _numeric(
                    recent[
                        quality_col
                    ]
                )
                .dropna()
            )

            if not quality_values.empty:
                quality = float(
                    np.clip(
                        quality_values.iloc[-1],
                        0.0,
                        1.0,
                    )
                )

        flow_signals.append(
            signal
        )

        flow_weights.append(
            STATION_WEIGHTS[
                station
            ]
            * (
                0.30
                +
                0.70
                * quality
            )
        )

    if flow_signals:
        flow_signal = float(
            np.average(
                flow_signals,
                weights=
                    flow_weights,
            )
        )
    else:
        flow_signal = 0.0

    # ========================================================
    # LLUVIA RECIENTE + FUTURA
    # ========================================================

    rain_signals = []
    rain_weights = []

    future = _prepare_exog(
        exog_future
    )

    for station in STATIONS:
        col = RAIN_COLUMNS[
            station
        ]

        historical_rain = 0.0

        if col in recent.columns:
            historical_rain = float(
                _numeric(
                    recent[col]
                )
                .fillna(0.0)
                .tail(7)
                .sum()
            )

        future_rain = 0.0

        if (
            not future.empty
            and col in future.columns
        ):
            future_rain = float(
                _numeric(
                    future[col]
                )
                .fillna(0.0)
                .head(15)
                .sum()
            )

        total_rain = (
            historical_rain
            +
            future_rain
        )

        rain_signal = np.clip(
            total_rain / 150.0,
            0.0,
            2.5,
        )

        rain_signals.append(
            rain_signal
        )

        rain_weights.append(
            STATION_WEIGHTS[
                station
            ]
        )

    if rain_signals:
        rain_signal = float(
            np.average(
                rain_signals,
                weights=
                    rain_weights,
            )
        )
    else:
        rain_signal = 0.0

    # ========================================================
    # PERSISTENCIA
    # ========================================================

    sn = (
        _numeric(
            recent[
                "nivel_san_nicolas"
            ]
        )
        .dropna()
        if (
            "nivel_san_nicolas"
            in recent.columns
        )
        else pd.Series(
            dtype=float
        )
    )

    if len(sn) >= 8:
        persistence_signal = float(
            np.clip(
                (
                    sn.iloc[-1]
                    - sn.iloc[-8]
                ) / 0.40,
                -1.5,
                1.5,
            )
        )
    else:
        persistence_signal = 0.0

    # ========================================================
    # ÍNDICE
    # ========================================================

    pressure = (
        0.35
        * level_signal
        +
        0.35
        * flow_signal
        +
        0.20
        * rain_signal
        +
        0.10
        * persistence_signal
    )

    pressure = float(
        np.clip(
            pressure,
            -1.5,
            2.5,
        )
    )

    return {
        "hydrological_pressure":
            pressure,

        "level_signal":
            level_signal,

        "flow_signal":
            flow_signal,

        "rain_signal":
            rain_signal,

        "persistence_signal":
            persistence_signal,
    }


# ============================================================
# TRAYECTORIA REAL DE UN EVENTO
# ============================================================

def _event_trajectory(
    dataset,
    event,
    days=MAX_SCENARIO_DAYS,
):
    start_date = pd.Timestamp(
        event[
            "start_date"
        ]
    )

    end_date = (
        start_date
        + pd.Timedelta(
            days=days
        )
    )

    window = dataset[
        (
            dataset[
                "datetime"
            ]
            >= start_date
        )
        &
        (
            dataset[
                "datetime"
            ]
            <= end_date
        )
    ][
        [
            "datetime",
            "nivel_san_nicolas",
        ]
    ].copy()

    window[
        "nivel_san_nicolas"
    ] = _numeric(
        window[
            "nivel_san_nicolas"
        ]
    )

    window = window.dropna()

    if window.empty:
        return pd.DataFrame()

    start_level = _safe_float(
        event[
            "start_level_sn"
        ]
    )

    if not np.isfinite(
        start_level
    ):
        start_level = float(
            window[
                "nivel_san_nicolas"
            ].iloc[0]
        )

    window[
        "scenario_day"
    ] = (
        window[
            "datetime"
        ]
        - start_date
    ).dt.days

    window[
        "relative_level"
    ] = (
        window[
            "nivel_san_nicolas"
        ]
        - start_level
    )

    return window[
        [
            "scenario_day",
            "relative_level",
        ]
    ]


# ============================================================
# ESCENARIOS HISTÓRICOS
# ============================================================

def build_historical_scenarios(
    dataset,
    similar_events,
    current_level,
    days=MAX_SCENARIO_DAYS,
):
    days = int(
        np.clip(
            days,
            1,
            MAX_SCENARIO_DAYS,
        )
    )

    empty = pd.DataFrame(
        {
            "scenario_day":
                np.arange(
                    1,
                    days + 1,
                ),

            "level":
                np.nan,
        }
    )

    if (
        similar_events is None
        or similar_events.empty
        or not np.isfinite(
            current_level
        )
    ):
        return (
            empty.copy(),
            empty.copy(),
            empty.copy(),
        )

    trajectories = []

    weights = []

    for _, event in (
        similar_events
        .head(SIMILAR_EVENT_LIMIT)
        .iterrows()
    ):
        trajectory = _event_trajectory(
            dataset,
            event,
            days=days,
        )

        if trajectory.empty:
            continue

        series = pd.Series(
            index=np.arange(
                1,
                days + 1,
            ),
            dtype=float,
        )

        for _, row in trajectory.iterrows():
            day = _safe_int(
                row[
                    "scenario_day"
                ],
                None,
            )

            if (
                day is None
                or day < 1
                or day > days
            ):
                continue

            series.loc[
                day
            ] = _safe_float(
                row[
                    "relative_level"
                ]
            )

        # Interpolar solamente dentro de un evento real.
        series = series.interpolate(
            limit_direction="forward",
            limit=5,
        )

        trajectories.append(
            series
        )

        weight = _safe_float(
            event.get(
                "similarity_weight"
            ),
            1.0,
        )

        weights.append(
            max(
                weight,
                0.05,
            )
        )

    if len(trajectories) < MIN_EVENTS_FOR_SCENARIO:
        return (
            empty.copy(),
            empty.copy(),
            empty.copy(),
        )

    matrix = pd.concat(
        trajectories,
        axis=1,
    )

    # ========================================================
    # Cuantiles reales por día.
    # P50 probable
    # P90 adverso
    # P98 extremo histórico comparable
    # ========================================================

    probable_rel = (
        matrix.quantile(
            0.50,
            axis=1,
        )
    )

    adverse_rel = (
        matrix.quantile(
            0.90,
            axis=1,
        )
    )

    extreme_rel = (
        matrix.quantile(
            0.98,
            axis=1,
        )
    )

    probable = pd.DataFrame(
        {
            "scenario_day":
                np.arange(
                    1,
                    days + 1,
                ),

            "level":
                current_level
                + probable_rel
                .reindex(
                    np.arange(
                        1,
                        days + 1,
                    )
                )
                .to_numpy(),
        }
    )

    adverse = pd.DataFrame(
        {
            "scenario_day":
                np.arange(
                    1,
                    days + 1,
                ),

            "level":
                current_level
                + adverse_rel
                .reindex(
                    np.arange(
                        1,
                        days + 1,
                    )
                )
                .to_numpy(),
        }
    )

    extreme = pd.DataFrame(
        {
            "scenario_day":
                np.arange(
                    1,
                    days + 1,
                ),

            "level":
                current_level
                + extreme_rel
                .reindex(
                    np.arange(
                        1,
                        days + 1,
                    )
                )
                .to_numpy(),
        }
    )

    # Garantizar orden físico de escenarios.
    for i in range(days):
        p = _safe_float(
            probable.loc[
                i,
                "level"
            ]
        )

        a = _safe_float(
            adverse.loc[
                i,
                "level"
            ]
        )

        e = _safe_float(
            extreme.loc[
                i,
                "level"
            ]
        )

        if (
            np.isfinite(p)
            and np.isfinite(a)
        ):
            adverse.loc[
                i,
                "level"
            ] = max(
                p,
                a,
            )

        a = _safe_float(
            adverse.loc[
                i,
                "level"
            ]
        )

        if (
            np.isfinite(a)
            and np.isfinite(e)
        ):
            extreme.loc[
                i,
                "level"
            ] = max(
                a,
                e,
            )

    return (
        probable,
        adverse,
        extreme,
    )


# ============================================================
# ESTIMACIÓN ACTUAL CORRIENTES -> SAN NICOLÁS
# ============================================================

def build_current_delay_estimate(
    lag_to_sn,
    similar_events,
    yearly_relation=None,
    dataset=None,
):
    """
    V11.12 usa primero la relación robusta año por año.
    Si no existe suficiente historia anual, conserva el cálculo
    directo como respaldo.
    """
    robust = summarize_corrientes_yearly(
        yearly_relation
    )

    corrientes_lag = robust.get(
        "delay_days",
        DEFAULT_CORRIENTES_LAG,
    )
    delay_min = robust.get("delay_min")
    delay_max = robust.get("delay_max")
    correlation = _safe_float(
        robust.get("correlation")
    )
    response = _safe_float(
        robust.get("response_m_per_m")
    )

    # Respaldo con la correlación directa si la anual no alcanza.
    if (
        robust.get("years", 0) == 0
        and isinstance(lag_to_sn, pd.DataFrame)
        and not lag_to_sn.empty
    ):
        match = lag_to_sn[
            lag_to_sn["station"] == "Corrientes"
        ]

        if not match.empty:
            row = match.iloc[0]
            corrientes_lag = _safe_int(
                row.get("lag_days"),
                DEFAULT_CORRIENTES_LAG,
            )
            correlation = _safe_float(
                row.get("correlation")
            )

            spread = (
                int(
                    np.clip(
                        round(
                            4
                            + 10
                            * (
                                1.0 - correlation
                            )
                        ),
                        4,
                        14,
                    )
                )
                if np.isfinite(correlation)
                else 10
            )

            delay_min = max(
                1,
                corrientes_lag - spread,
            )
            delay_max = min(
                MAX_LAG_DAYS,
                corrientes_lag + spread,
            )

    similar_count = (
        len(similar_events)
        if isinstance(
            similar_events,
            pd.DataFrame,
        )
        else 0
    )

    # Traducción de la variación reciente de Corrientes a una
    # variación esperable en San Nicolás, sin comparar alturas
    # absolutas entre escalas.
    corrientes_change_7d = np.nan
    expected_sn_change = np.nan

    if (
        isinstance(dataset, pd.DataFrame)
        and not dataset.empty
        and LEVEL_COLUMNS["Corrientes"] in dataset.columns
    ):
        c = (
            _numeric(
                dataset[
                    LEVEL_COLUMNS["Corrientes"]
                ]
            )
            .dropna()
        )

        if len(c) >= 8:
            corrientes_change_7d = float(
                c.iloc[-1] - c.iloc[-8]
            )

    if (
        np.isfinite(corrientes_change_7d)
        and np.isfinite(response)
    ):
        expected_sn_change = float(
            corrientes_change_7d * response
        )

    return {
        "delay_days": _safe_int(
            corrientes_lag,
            DEFAULT_CORRIENTES_LAG,
        ),
        "delay_min": _safe_int(delay_min, 1),
        "delay_max": _safe_int(
            delay_max,
            MAX_LAG_DAYS,
        ),
        "correlation": correlation,
        "response_m_per_m": response,
        "corrientes_change_7d": corrientes_change_7d,
        "expected_sn_change": expected_sn_change,
        "annual_years": int(
            robust.get("years", 0)
        ),
        "quality_years": int(
            robust.get("quality_years", 0)
        ),
        "similar_event_count": similar_count,
    }


# ============================================================
# RESUMEN DE EVENTOS SIMILARES
# ============================================================

def _similar_summary(
    similar_events,
):
    if (
        similar_events is None
        or similar_events.empty
    ):
        return {
            "count": 0,
            "median_rise": np.nan,
            "p90_rise": np.nan,
            "max_rise": np.nan,
            "median_rise_days": np.nan,
        }

    rise = _numeric(
        similar_events[
            "rise_sn"
        ]
    ).dropna()

    rise_days = _numeric(
        similar_events[
            "rise_days"
        ]
    ).dropna()

    return {
        "count":
            int(
                len(similar_events)
            ),

        "median_rise":
            (
                float(
                    rise.median()
                )
                if not rise.empty
                else np.nan
            ),

        "p90_rise":
            (
                float(
                    rise.quantile(
                        0.90
                    )
                )
                if not rise.empty
                else np.nan
            ),

        "max_rise":
            (
                float(
                    rise.max()
                )
                if not rise.empty
                else np.nan
            ),

        "median_rise_days":
            (
                float(
                    rise_days.median()
                )
                if not rise_days.empty
                else np.nan
            ),
    }


# ============================================================
# ESTADÍSTICAS CORRIENTES - SAN NICOLÁS
# ============================================================

def _corrientes_statistics(
    dataset,
    lag_to_sn,
):
    result = {
        "lag_days":
            DEFAULT_CORRIENTES_LAG,

        "correlation":
            np.nan,

        "overlap":
            0,
    }

    if (
        isinstance(
            lag_to_sn,
            pd.DataFrame,
        )
        and not lag_to_sn.empty
    ):
        match = lag_to_sn[
            lag_to_sn[
                "station"
            ]
            == "Corrientes"
        ]

        if not match.empty:
            row = match.iloc[0]

            result[
                "lag_days"
            ] = _safe_int(
                row.get(
                    "lag_days"
                ),
                DEFAULT_CORRIENTES_LAG,
            )

            result[
                "correlation"
            ] = _safe_float(
                row.get(
                    "correlation"
                )
            )

            result[
                "overlap"
            ] = _safe_int(
                row.get(
                    "overlap"
                ),
                0,
            )

    return result


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def analizar_corrientes_san_nicolas(
    sn_history,
    upstream_history,
    exog_history=None,
    exog_future=None,
    days=60,
    **kwargs,
):
    """
    Motor hidrológico principal.

    Compatible con app.py V11.12 y model.py V11.11.

    Devuelve:
        dataset
        corridor_lags
        lag_to_sn
        events
        event_table
        similar_events
        similar_summary
        pressure
        current_estimate
        statistics
        scenario_probable
        scenario_adverse
        scenario_extreme
    """

    days = int(
        np.clip(
            days,
            1,
            MAX_SCENARIO_DAYS,
        )
    )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = build_hydrology_dataset(
        sn_history,
        upstream_history,
        exog_history=
            exog_history,
    )

    if dataset.empty:
        return {
            "version":
                VERSION,

            "status":
                "sin_datos",

            "dataset":
                pd.DataFrame(),

            "corridor_lags":
                pd.DataFrame(),

            "lag_to_sn":
                pd.DataFrame(),

            "events":
                pd.DataFrame(),

            "event_table":
                pd.DataFrame(),

            "similar_events":
                pd.DataFrame(),

            "similar_summary":
                {},

            "pressure":
                {},

            "current_estimate":
                {},

            "corrientes_yearly":
                pd.DataFrame(),

            "corrientes_robust":
                {},

            "statistics":
                {},

            "scenario_probable":
                pd.DataFrame(),

            "scenario_adverse":
                pd.DataFrame(),

            "scenario_extreme":
                pd.DataFrame(),
        }

    # ========================================================
    # RETARDOS
    # ========================================================

    corridor_lags = (
        estimate_corridor_lags(
            dataset
        )
    )

    lag_to_sn = (
        estimate_lag_to_sn(
            dataset,
            corridor_lags=
                corridor_lags,
        )
    )

    corrientes_yearly = (
        estimate_corrientes_yearly(
            dataset,
            max_lag=
                MAX_LAG_DAYS,
        )
    )

    corrientes_robust = (
        summarize_corrientes_yearly(
            corrientes_yearly
        )
    )

    # ========================================================
    # EVENTOS
    # ========================================================

    events = detect_historical_events(
        dataset
    )

    event_table = build_event_table(
        dataset,
        events,
    )

    # ========================================================
    # ESTADO ACTUAL
    # ========================================================

    current_features = (
        _current_features(
            dataset
        )
    )

    # ========================================================
    # EVENTOS SIMILARES
    # ========================================================

    similar_events = (
        find_similar_events(
            event_table,
            current_features,
            limit=
                SIMILAR_EVENT_LIMIT,
        )
    )

    similar_summary = (
        _similar_summary(
            similar_events
        )
    )

    # ========================================================
    # PRESIÓN HIDROLÓGICA
    # ========================================================

    pressure = (
        calculate_hydrological_pressure(
            dataset,
            exog_future=
                exog_future,
        )
    )

    # ========================================================
    # NIVEL ACTUAL
    # ========================================================

    sn_values = (
        _numeric(
            dataset[
                "nivel_san_nicolas"
            ]
        )
        .dropna()
    )

    current_level = (
        float(
            sn_values.iloc[-1]
        )
        if not sn_values.empty
        else np.nan
    )

    # ========================================================
    # ESCENARIOS HISTÓRICOS
    # ========================================================

    (
        scenario_probable,
        scenario_adverse,
        scenario_extreme,
    ) = build_historical_scenarios(
        dataset,
        similar_events,
        current_level,
        days=days,
    )

    # ========================================================
    # ESTIMACIÓN ACTUAL DE PROPAGACIÓN
    # ========================================================

    current_estimate = (
        build_current_delay_estimate(
            lag_to_sn,
            similar_events,
            yearly_relation=
                corrientes_yearly,
            dataset=
                dataset,
        )
    )

    statistics = (
        _corrientes_statistics(
            dataset,
            lag_to_sn,
        )
    )

    # ========================================================
    # FECHA ESTIMADA DE PROPAGACIÓN
    # ========================================================

    if (
        not dataset.empty
        and current_estimate.get(
            "delay_days"
        )
        is not None
    ):
        base_date = (
            dataset[
                "datetime"
            ].max()
        )

        propagation_date = (
            base_date
            + pd.Timedelta(
                days=
                    current_estimate[
                        "delay_days"
                    ]
            )
        )

        current_estimate[
            "propagation_date"
        ] = propagation_date

    # ========================================================
    # RESULTADO
    # ========================================================

    return {
        "version":
            VERSION,

        "status":
            "ok",

        "dataset":
            dataset,

        "history":
            dataset,

        "features":
            current_features,

        "corridor_lags":
            corridor_lags,

        "lag_to_sn":
            lag_to_sn,

        "events":
            events,

        "event_table":
            event_table,

        "similar_events":
            similar_events,

        "similar_summary":
            similar_summary,

        "pressure":
            pressure,

        "current_estimate":
            current_estimate,

        "corrientes_yearly":
            corrientes_yearly,

        "corrientes_robust":
            corrientes_robust,

        "statistics":
            statistics,

        "scenario_probable":
            scenario_probable,

        "scenario_adverse":
            scenario_adverse,

        "scenario_extreme":
            scenario_extreme,
    }


# ============================================================
# ALIAS DE COMPATIBILIDAD
# ============================================================

def analyze_corrientes_san_nicolas(
    sn_history,
    upstream_history,
    exog_history=None,
    exog_future=None,
    days=60,
    **kwargs,
):
    return analizar_corrientes_san_nicolas(
        sn_history,
        upstream_history,
        exog_history=
            exog_history,
        exog_future=
            exog_future,
        days=
            days,
        **kwargs,
    )


# ============================================================
# DIAGNÓSTICO
# ============================================================

def diagnostic(
    sn_history,
    upstream_history,
    exog_history=None,
    exog_future=None,
):
    result = {
        "version":
            VERSION,

        "status":
            "pendiente",
    }

    try:
        hydro = (
            analizar_corrientes_san_nicolas(
                sn_history,
                upstream_history,
                exog_history=
                    exog_history,
                exog_future=
                    exog_future,
                days=60,
            )
        )

        result[
            "status"
        ] = hydro.get(
            "status",
            "ok",
        )

        result[
            "dataset_rows"
        ] = (
            len(
                hydro.get(
                    "dataset",
                    pd.DataFrame(),
                )
            )
        )

        result[
            "events"
        ] = (
            len(
                hydro.get(
                    "events",
                    pd.DataFrame(),
                )
            )
        )

        result[
            "similar_events"
        ] = (
            len(
                hydro.get(
                    "similar_events",
                    pd.DataFrame(),
                )
            )
        )

        result[
            "pressure"
        ] = hydro.get(
            "pressure",
            {}
        )

        result[
            "current_estimate"
        ] = hydro.get(
            "current_estimate",
            {}
        )

        result[
            "statistics"
        ] = hydro.get(
            "statistics",
            {}
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
