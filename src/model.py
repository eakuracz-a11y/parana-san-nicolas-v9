# ============================================================
# PARANÁ · SAN NICOLÁS
# src/model.py
# V11.11 COMPLETO
#
# MODELO HIDROLÓGICO MULTIVARIABLE
#
# Integra:
#
#   - nivel San Nicolás
#   - niveles aguas arriba
#   - caudales INA observados
#   - caudales reconstruidos
#   - calidad del caudal
#   - lluvias por estación
#   - lluvia acumulada
#   - propagación hidrológica
#   - presión hidrológica
#   - eventos históricos similares
#   - escenarios P50 / P90 / P98
#
# HORIZONTES:
#
#   1-15 días   = pronóstico operativo
#   16-30 días  = proyección hidrológica
#   31-45 días  = escenario extendido
#   46-60 días  = tendencia / escenario histórico
#
# API PRINCIPAL:
#
# train(
#     df,
#     exog_history=None,
#     upstream_history=None,
#     hydrology=None,
# )
#
# predict(
#     df,
#     models,
#     days=60,
#     exog_future=None,
#     upstream_future=None,
#     hydrology=None,
# )
#
# ============================================================


import math

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# VERSIÓN
# ============================================================

VERSION = "V11.11"


# ============================================================
# CONFIGURACIÓN
# ============================================================

MIN_OBSERVATIONS = 60

MIN_TRAINING_ROWS = 35

MAX_FORECAST_DAYS = 60

MIN_RMSE = 0.03

MAX_UNCERTAINTY = 2.00

LEVEL_MIN = -2.0

LEVEL_MAX = 12.0

RANDOM_STATE = 42


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
# COLUMNAS
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
# PESOS DEL CORREDOR
# ============================================================

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
# RETARDOS DE RESPALDO
# ============================================================

DEFAULT_PROPAGATION_LAGS = {

    "Corrientes": 20,

    "Goya": 16,

    "La Paz": 12,

    "Paraná": 8,

    "Diamante": 6,

    "Rosario": 3,

    "Villa Constitución": 1,
}


# ============================================================
# UTILIDADES
# ============================================================

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


def _numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
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


# ============================================================
# NIVEL LOCAL
# ============================================================

def preparar_nivel_local(df):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):

        raise ValueError(
            "No hay datos de nivel de San Nicolás."
        )

    x = df.copy()

    if "datetime" not in x.columns:

        raise ValueError(
            "Los datos no contienen columna datetime."
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

        raise ValueError(
            "No fue posible identificar el nivel de San Nicolás."
        )

    x[
        "datetime"
    ] = _datetime_naive(
        x[
            "datetime"
        ]
    )

    x[
        "nivel"
    ] = _numeric(
        x[
            level_col
        ]
    )

    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    x = x[
        (
            x[
                "nivel"
            ]
            >= LEVEL_MIN
        )
        &
        (
            x[
                "nivel"
            ]
            <= LEVEL_MAX
        )
    ]

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
            "nivel"
        ]
        .mean()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    if len(x) < MIN_OBSERVATIONS:

        raise ValueError(
            "No hay suficientes observaciones para entrenar el modelo."
        )

    return x


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
    ):

        return pd.DataFrame(
            columns=[
                "datetime"
            ]
        )

    x = exog_history.copy()

    if "datetime" not in x.columns:

        return pd.DataFrame(
            columns=[
                "datetime"
            ]
        )

    x[
        "datetime"
    ] = _datetime_naive(
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

    numeric_prefixes = (
        "q_",
        "rain_",
        "caudal_",
        "precip_",
        "flow_",
        "hydro_",
    )

    for col in x.columns:

        if col == "datetime":
            continue

        if (
            col.startswith(
                numeric_prefixes
            )
            or col.endswith(
                "_quality"
            )
        ):

            x[col] = _numeric(
                x[col]
            )

    return (
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


# ============================================================
# UPSTREAM
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

        return pd.DataFrame(
            columns=[
                "datetime"
            ]
        )

    x = upstream_history.copy()

    if "datetime" not in x.columns:

        return pd.DataFrame(
            columns=[
                "datetime"
            ]
        )

    x[
        "datetime"
    ] = _datetime_naive(
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

    for col in LEVEL_COLUMNS.values():

        if col in x.columns:

            x[col] = _numeric(
                x[col]
            )

    return (
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


# ============================================================
# DATASET
# ============================================================

def preparar_dataset(
    df,
    exog_history=None,
    upstream_history=None,
):

    local = preparar_nivel_local(
        df
    )

    upstream = preparar_upstream(
        upstream_history
    )

    exog = preparar_exogenas(
        exog_history
    )

    result = local.copy()

    # ========================================================
    # UPSTREAM
    # ========================================================

    if not upstream.empty:

        duplicate = [
            col
            for col in upstream.columns
            if (
                col != "datetime"
                and col in result.columns
            )
        ]

        upstream = upstream.drop(
            columns=duplicate,
            errors="ignore",
        )

        result = result.merge(
            upstream,
            on="datetime",
            how="left",
        )

    # ========================================================
    # EXÓGENAS
    # ========================================================

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
            how="left",
        )

    # ========================================================
    # INTERPOLACIÓN LIMITADA
    # ========================================================

    interpolate_cols = (
        list(
            LEVEL_COLUMNS.values()
        )
        +
        list(
            FLOW_COLUMNS.values()
        )
    )

    for col in interpolate_cols:

        if col in result.columns:

            result[col] = (
                _numeric(
                    result[col]
                )
                .interpolate(
                    limit=5,
                    limit_area="inside",
                )
            )

    return (
        result
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# RETARDOS HIDROLÓGICOS
# ============================================================

def obtener_lags_hidrologicos(
    hydrology=None,
):

    result = dict(
        DEFAULT_PROPAGATION_LAGS
    )

    if not isinstance(
        hydrology,
        dict,
    ):

        return result

    # ========================================================
    # LAG DIRECTO A SN
    # ========================================================

    lag_to_sn = hydrology.get(
        "lag_to_sn"
    )

    if (
        isinstance(
            lag_to_sn,
            pd.DataFrame,
        )
        and not lag_to_sn.empty
    ):

        for _, row in (
            lag_to_sn.iterrows()
        ):

            station = row.get(
                "station"
            )

            lag = _safe_int(
                row.get(
                    "lag_days"
                ),
                None,
            )

            if (
                station
                in result
                and lag is not None
            ):

                result[
                    station
                ] = int(
                    np.clip(
                        lag,
                        0,
                        45,
                    )
                )

    # ========================================================
    # COMPATIBILIDAD CON VERSIONES ANTERIORES
    # ========================================================

    corridor_lags = hydrology.get(
        "corridor_lags"
    )

    if (
        isinstance(
            corridor_lags,
            pd.DataFrame,
        )
        and not corridor_lags.empty
    ):

        # sólo usamos esto como respaldo si no vino lag_to_sn
        if (
            not isinstance(
                lag_to_sn,
                pd.DataFrame,
            )
            or lag_to_sn.empty
        ):

            segments = {}

            for _, row in (
                corridor_lags.iterrows()
            ):

                upstream = row.get(
                    "upstream"
                )

                downstream = row.get(
                    "downstream"
                )

                lag = _safe_int(
                    row.get(
                        "lag_days"
                    ),
                    None,
                )

                if (
                    upstream
                    and downstream
                    and lag is not None
                ):

                    segments[
                        (
                            upstream,
                            downstream,
                        )
                    ] = lag

            corridor = [
                "Corrientes",
                "Goya",
                "La Paz",
                "Paraná",
                "Diamante",
                "Rosario",
                "Villa Constitución",
                "San Nicolás",
            ]

            for i, station in enumerate(
                corridor[:-1]
            ):

                total = 0

                valid = True

                for j in range(
                    i,
                    len(corridor) - 1,
                ):

                    key = (
                        corridor[j],
                        corridor[j + 1],
                    )

                    if key not in segments:

                        valid = False
                        break

                    total += segments[key]

                if valid:

                    result[
                        station
                    ] = int(
                        np.clip(
                            total,
                            0,
                            45,
                        )
                    )

    return result


# ============================================================
# FEATURES LOCALES
# ============================================================

def agregar_features_locales(
    x,
):

    result = x.copy()

    nivel = _numeric(
        result[
            "nivel"
        ]
    )

    # ========================================================
    # LAGS
    # ========================================================

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

    # ========================================================
    # CAMBIOS
    # ========================================================

    for lag in [
        1,
        3,
        7,
        14,
    ]:

        result[
            f"nivel_diff_{lag}"
        ] = nivel.diff(
            lag
        )

    # ========================================================
    # ROLLING
    # ========================================================

    for window in [
        3,
        7,
        14,
        30,
    ]:

        roll = nivel.rolling(
            window,
            min_periods=1,
        )

        result[
            f"nivel_mean_{window}"
        ] = roll.mean()

        result[
            f"nivel_std_{window}"
        ] = roll.std()

        result[
            f"nivel_min_{window}"
        ] = roll.min()

        result[
            f"nivel_max_{window}"
        ] = roll.max()

        result[
            f"nivel_range_{window}"
        ] = (
            result[
                f"nivel_max_{window}"
            ]
            -
            result[
                f"nivel_min_{window}"
            ]
        )

    return result


# ============================================================
# FEATURES UPSTREAM
# ============================================================

def agregar_features_upstream(
    x,
    lags,
):

    result = x.copy()

    weighted_signal = pd.Series(
        0.0,
        index=result.index,
        dtype=float,
    )

    total_weight = 0.0

    for station, col in (
        LEVEL_COLUMNS.items()
    ):

        if col not in result.columns:

            continue

        level = _numeric(
            result[col]
        )

        if level.notna().sum() < 10:

            continue

        # ----------------------------------------------------
        # Cambios
        # ----------------------------------------------------

        for lag in [
            1,
            3,
            7,
            14,
        ]:

            result[
                f"{col}_diff_{lag}"
            ] = level.diff(
                lag
            )

        # ----------------------------------------------------
        # Tendencia
        # ----------------------------------------------------

        result[
            f"{col}_mean_7"
        ] = (
            level
            .rolling(
                7,
                min_periods=1,
            )
            .mean()
        )

        result[
            f"{col}_mean_14"
        ] = (
            level
            .rolling(
                14,
                min_periods=1,
            )
            .mean()
        )

        result[
            f"{col}_trend_7"
        ] = (
            level
            -
            level.shift(7)
        )

        # ----------------------------------------------------
        # Lags estándares
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
            ] = level.shift(
                lag
            )

        # ----------------------------------------------------
        # Lag de propagación aprendido
        # ----------------------------------------------------

        propagation_lag = int(
            lags.get(
                station,
                DEFAULT_PROPAGATION_LAGS.get(
                    station,
                    7,
                ),
            )
        )

        propagation_lag = int(
            np.clip(
                propagation_lag,
                0,
                45,
            )
        )

        result[
            f"{col}_propagated"
        ] = level.shift(
            propagation_lag
        )

        # ----------------------------------------------------
        # Señal normalizada
        # ----------------------------------------------------

        trend = (
            level
            -
            level.shift(7)
        )

        signal = (
            trend / 0.50
        ).clip(
            lower=-2.0,
            upper=2.0,
        )

        weight = STATION_WEIGHTS[
            station
        ]

        weighted_signal += (
            signal.fillna(0.0)
            * weight
        )

        total_weight += weight

    if total_weight > 0:

        weighted_signal /= total_weight

    result[
        "upstream_level_signal"
    ] = weighted_signal

    return result


# ============================================================
# FEATURES DE CAUDAL
# ============================================================

def agregar_features_caudal(
    x,
):

    result = x.copy()

    weighted_signal = pd.Series(
        0.0,
        index=result.index,
        dtype=float,
    )

    weighted_quality = pd.Series(
        0.0,
        index=result.index,
        dtype=float,
    )

    total_weight = 0.0

    for station, col in (
        FLOW_COLUMNS.items()
    ):

        if col not in result.columns:

            continue

        q = _numeric(
            result[col]
        )

        if q.notna().sum() < 5:

            continue

        quality_col = (
            col
            + "_quality"
        )

        if quality_col in result.columns:

            quality = (
                _numeric(
                    result[
                        quality_col
                    ]
                )
                .fillna(0.0)
                .clip(
                    lower=0.0,
                    upper=1.0,
                )
            )

        else:

            quality = pd.Series(
                1.0,
                index=result.index,
            )

        # ----------------------------------------------------
        # Cambios
        # ----------------------------------------------------

        for lag in [
            1,
            3,
            7,
            14,
        ]:

            result[
                f"{col}_diff_{lag}"
            ] = q.diff(
                lag
            )

        # ----------------------------------------------------
        # Medias
        # ----------------------------------------------------

        for window in [
            3,
            7,
            14,
            30,
        ]:

            result[
                f"{col}_mean_{window}"
            ] = (
                q
                .rolling(
                    window,
                    min_periods=1,
                )
                .mean()
            )

        result[
            f"{col}_relative_7"
        ] = (
            q.diff(7)
            /
            q.shift(7).replace(
                0,
                np.nan,
            )
        )

        result[
            f"{col}_relative_14"
        ] = (
            q.diff(14)
            /
            q.shift(14).replace(
                0,
                np.nan,
            )
        )

        # ----------------------------------------------------
        # CALIDAD COMO FEATURE
        # ----------------------------------------------------

        result[
            f"{col}_quality_feature"
        ] = quality

        # ----------------------------------------------------
        # Señal ponderada
        # ----------------------------------------------------

        baseline = (
            q
            .rolling(
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

        weight = STATION_WEIGHTS[
            station
        ]

        weighted_signal += (
            relative.fillna(0.0)
            * quality
            * weight
        )

        weighted_quality += (
            quality
            * weight
        )

        total_weight += weight

    valid_quality = (
        weighted_quality
        .replace(
            0,
            np.nan,
        )
    )

    result[
        "weighted_flow_signal"
    ] = (
        weighted_signal
        /
        valid_quality
    ).fillna(0.0)

    if total_weight > 0:

        result[
            "flow_data_quality"
        ] = (
            weighted_quality
            / total_weight
        ).clip(
            lower=0.0,
            upper=1.0,
        )

    else:

        result[
            "flow_data_quality"
        ] = 0.0

    return result


# ============================================================
# FEATURES DE LLUVIA
# ============================================================

def agregar_features_lluvia(
    x,
):

    result = x.copy()

    rain_signal = pd.Series(
        0.0,
        index=result.index,
        dtype=float,
    )

    total_weight = 0.0

    for station, col in (
        RAIN_COLUMNS.items()
    ):

        if col not in result.columns:

            continue

        rain = (
            _numeric(
                result[col]
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
                rain
                .rolling(
                    window,
                    min_periods=1,
                )
                .sum()
            )

        weight = STATION_WEIGHTS[
            station
        ]

        rain_15d = (
            rain
            .rolling(
                15,
                min_periods=1,
            )
            .sum()
        )

        rain_signal += (
            (
                rain_15d
                / 100.0
            )
            .clip(
                lower=0.0,
                upper=3.0,
            )
            * weight
        )

        total_weight += weight

    if total_weight > 0:

        rain_signal /= total_weight

    result[
        "weighted_rain_signal"
    ] = rain_signal

    return result


# ============================================================
# FEATURES TEMPORALES
# ============================================================

def agregar_features_temporales(
    x,
):

    result = x.copy()

    dt = pd.to_datetime(
        result[
            "datetime"
        ],
        errors="coerce",
    )

    result[
        "month"
    ] = dt.dt.month

    result[
        "dayofyear"
    ] = dt.dt.dayofyear

    result[
        "month_sin"
    ] = np.sin(
        2
        * np.pi
        * result[
            "month"
        ]
        / 12.0
    )

    result[
        "month_cos"
    ] = np.cos(
        2
        * np.pi
        * result[
            "month"
        ]
        / 12.0
    )

    result[
        "doy_sin"
    ] = np.sin(
        2
        * np.pi
        * result[
            "dayofyear"
        ]
        / 365.25
    )

    result[
        "doy_cos"
    ] = np.cos(
        2
        * np.pi
        * result[
            "dayofyear"
        ]
        / 365.25
    )

    return result


# ============================================================
# FEATURES HIDROLÓGICAS GLOBALES
# ============================================================

def agregar_features_hidrologicas_globales(
    x,
):

    result = x.copy()

    upstream = _numeric(
        result.get(
            "upstream_level_signal",
            0.0,
        )
    ).fillna(0.0)

    flow = _numeric(
        result.get(
            "weighted_flow_signal",
            0.0,
        )
    ).fillna(0.0)

    rain = _numeric(
        result.get(
            "weighted_rain_signal",
            0.0,
        )
    ).fillna(0.0)

    quality = _numeric(
        result.get(
            "flow_data_quality",
            0.0,
        )
    ).fillna(0.0)

    # ========================================================
    # ÍNDICE HIDROLÓGICO
    # ========================================================

    result[
        "hydrological_pressure_model"
    ] = (
        0.40
        * upstream
        +
        0.35
        * flow
        * (
            0.50
            +
            0.50
            * quality
        )
        +
        0.25
        * rain
    ).clip(
        lower=-1.5,
        upper=3.0,
    )

    # ========================================================
    # INTERACCIONES
    # ========================================================

    result[
        "flow_x_rain"
    ] = (
        flow
        * rain
    )

    result[
        "upstream_x_flow"
    ] = (
        upstream
        * flow
    )

    result[
        "upstream_x_rain"
    ] = (
        upstream
        * rain
    )

    result[
        "combined_hydro_signal"
    ] = (
        upstream
        +
        flow
        +
        rain
    ) / 3.0

    return result


# ============================================================
# CREAR TABLA DE FEATURES
# ============================================================

def crear_features(
    dataset,
    propagation_lags=None,
):

    if propagation_lags is None:

        propagation_lags = dict(
            DEFAULT_PROPAGATION_LAGS
        )

    x = dataset.copy()

    x = agregar_features_locales(
        x
    )

    x = agregar_features_upstream(
        x,
        propagation_lags,
    )

    x = agregar_features_caudal(
        x
    )

    x = agregar_features_lluvia(
        x
    )

    x = agregar_features_temporales(
        x
    )

    x = agregar_features_hidrologicas_globales(
        x
    )

    # ========================================================
    # TARGET
    # ========================================================

    x[
        "target_next_level"
    ] = x[
        "nivel"
    ].shift(
        -1
    )

    x[
        "target_delta"
    ] = (
        x[
            "target_next_level"
        ]
        -
        x[
            "nivel"
        ]
    )

    return x


# ============================================================
# SELECCIÓN DE FEATURES
# ============================================================

def seleccionar_features(
    features,
):

    excluded = {

        "datetime",
        "nivel",
        "target_next_level",
        "target_delta",
    }

    feature_names = []

    for col in features.columns:

        if col in excluded:
            continue

        # ----------------------------------------------------
        # Sólo numéricas
        # ----------------------------------------------------

        values = _numeric(
            features[col]
        )

        valid_count = int(
            values.notna().sum()
        )

        if valid_count < 20:

            continue

        if values.nunique(
            dropna=True
        ) <= 1:

            continue

        feature_names.append(
            col
        )

    return feature_names


# ============================================================
# COBERTURA POR GRUPO
# ============================================================

def _count_feature_groups(
    feature_names,
):

    upstream_count = 0
    flow_count = 0
    rain_count = 0
    hydrology_count = 0

    for feature in feature_names:

        low = feature.lower()

        if (
            "nivel_corrientes"
            in low
            or "nivel_goya"
            in low
            or "nivel_la_paz"
            in low
            or "nivel_parana"
            in low
            or "nivel_diamante"
            in low
            or "nivel_rosario"
            in low
            or "nivel_villa"
            in low
            or "upstream_"
            in low
        ):

            upstream_count += 1

        if (
            low.startswith(
                "q_"
            )
            or "caudal"
            in low
            or "flow_"
            in low
        ):

            flow_count += 1

        if (
            low.startswith(
                "rain_"
            )
            or "precip"
            in low
        ):

            rain_count += 1

        if (
            "hydro"
            in low
            or "propag"
            in low
            or "combined"
            in low
        ):

            hydrology_count += 1

    return {

        "upstream":
            upstream_count,

        "flow":
            flow_count,

        "rain":
            rain_count,

        "hydrology":
            hydrology_count,
    }


# ============================================================
# TRAIN
# ============================================================

def train(
    df,
    exog_history=None,
    upstream_history=None,
    hydrology=None,
    **kwargs,
):

    dataset = preparar_dataset(

        df,

        exog_history=
            exog_history,

        upstream_history=
            upstream_history,
    )

    propagation_lags = (
        obtener_lags_hidrologicos(
            hydrology
        )
    )

    features = crear_features(

        dataset,

        propagation_lags=
            propagation_lags,
    )

    feature_names = (
        seleccionar_features(
            features
        )
    )

    if not feature_names:

        raise ValueError(
            "No fue posible generar variables para el entrenamiento."
        )

    # ========================================================
    # TARGET
    # ========================================================

    target = _numeric(
        features[
            "target_delta"
        ]
    )

    valid_target = (
        target.notna()
    )

    training = features.loc[
        valid_target
    ].copy()

    target = target.loc[
        valid_target
    ]

    # ========================================================
    # FILL VALUES
    # ========================================================

    fill_values = {}

    for col in feature_names:

        values = _numeric(
            training[col]
        )

        median = values.median()

        if not np.isfinite(
            median
        ):

            median = 0.0

        fill_values[col] = float(
            median
        )

        training[col] = (
            values
            .fillna(
                median
            )
        )

    if len(training) < MIN_TRAINING_ROWS:

        raise ValueError(
            "No hay suficientes filas válidas para entrenar."
        )

    X = training[
        feature_names
    ]

    y = target.to_numpy(
        dtype=float
    )

    # ========================================================
    # VALIDACIÓN CRONOLÓGICA
    # ========================================================

    split = int(
        len(X)
        * 0.80
    )

    split = max(
        split,
        MIN_TRAINING_ROWS,
    )

    split = min(
        split,
        len(X) - 5,
    )

    if split <= 0:

        raise ValueError(
            "No hay datos suficientes para validación."
        )

    X_train = X.iloc[
        :split
    ]

    y_train = y[
        :split
    ]

    X_test = X.iloc[
        split:
    ]

    y_test = y[
        split:
    ]

    # ========================================================
    # MODELO DE VALIDACIÓN
    # ========================================================

    validation_model = RandomForestRegressor(

        n_estimators=700,

        max_depth=14,

        min_samples_leaf=2,

        max_features="sqrt",

        random_state=
            RANDOM_STATE,

        n_jobs=-1,
    )

    validation_model.fit(
        X_train,
        y_train,
    )

    validation_prediction = (
        validation_model.predict(
            X_test
        )
    )

    rmse = math.sqrt(
        mean_squared_error(
            y_test,
            validation_prediction,
        )
    )

    mae = mean_absolute_error(
        y_test,
        validation_prediction,
    )

    rmse = max(
        float(rmse),
        MIN_RMSE,
    )

    # ========================================================
    # MODELO FINAL
    # ========================================================

    final_model = RandomForestRegressor(

        n_estimators=1000,

        max_depth=15,

        min_samples_leaf=2,

        max_features="sqrt",

        random_state=
            RANDOM_STATE,

        n_jobs=-1,
    )

    final_model.fit(
        X,
        y,
    )

    # ========================================================
    # IMPORTANCIA
    # ========================================================

    importance = pd.DataFrame(
        {
            "feature":
                feature_names,

            "importance":
                final_model
                .feature_importances_,
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # COBERTURA
    # ========================================================

    group_counts = (
        _count_feature_groups(
            feature_names
        )
    )

    available_upstream = [
        col
        for col in LEVEL_COLUMNS.values()
        if (
            col in dataset.columns
            and _numeric(
                dataset[col]
            )
            .notna()
            .sum()
            >= 3
        )
    ]

    available_flow = [
        col
        for col in FLOW_COLUMNS.values()
        if (
            col in dataset.columns
            and _numeric(
                dataset[col]
            )
            .notna()
            .sum()
            >= 3
        )
    ]

    available_rain = [
        col
        for col in RAIN_COLUMNS.values()
        if (
            col in dataset.columns
            and _numeric(
                dataset[col]
            )
            .notna()
            .sum()
            >= 3
        )
    ]

    # ========================================================
    # HIDROLOGÍA
    # ========================================================

    hydrology_summary = {}

    if isinstance(
        hydrology,
        dict,
    ):

        hydrology_summary = {

            "current_estimate":
                hydrology.get(
                    "current_estimate",
                    {}
                ),

            "pressure":
                hydrology.get(
                    "pressure",
                    {}
                ),

            "similar_summary":
                hydrology.get(
                    "similar_summary",
                    {}
                ),
        }

    # ========================================================
    # RESULTADO
    # ========================================================

    models = {

        "version":
            VERSION,

        "model":
            final_model,

        "feature_names":
            feature_names,

        "feature_fill_values":
            fill_values,

        "rmse":
            rmse,

        "mae":
            float(mae),

        "dataset":
            dataset,

        "features_table":
            features,

        "importance":
            importance,

        "propagation_lags":
            propagation_lags,

        "hydrology":
            hydrology,

        "hydrology_summary":
            hydrology_summary,

        # ----------------------------------------------------
        # COMPATIBILIDAD APP
        # ----------------------------------------------------

        "uses_upstream":
            group_counts[
                "upstream"
            ] > 0,

        "uses_caudal":
            group_counts[
                "flow"
            ] > 0,

        "uses_rain":
            group_counts[
                "rain"
            ] > 0,

        "uses_hydrology":
            group_counts[
                "hydrology"
            ] > 0,

        "upstream_feature_count":
            group_counts[
                "upstream"
            ],

        "flow_feature_count":
            group_counts[
                "flow"
            ],

        "rain_feature_count":
            group_counts[
                "rain"
            ],

        "hydrology_feature_count":
            group_counts[
                "hydrology"
            ],

        "available_upstream_columns":
            available_upstream,

        "available_flow_columns":
            available_flow,

        "available_rain_columns":
            available_rain,

        "training_rows":
            int(
                len(training)
            ),

        "feature_count":
            int(
                len(feature_names)
            ),
    }

    metrics = {

        "version":
            VERSION,

        "rmse":
            rmse,

        "mae":
            float(mae),

        "training_rows":
            int(
                len(training)
            ),

        "features":
            int(
                len(feature_names)
            ),

        "upstream_features":
            group_counts[
                "upstream"
            ],

        "flow_features":
            group_counts[
                "flow"
            ],

        "rain_features":
            group_counts[
                "rain"
            ],

        "hydrology_features":
            group_counts[
                "hydrology"
            ],
    }

    return (
        models,
        metrics,
    )


# ============================================================
# HORIZONTE
# ============================================================

def _horizon_group(day):

    if day <= 15:

        return "1-15"

    if day <= 30:

        return "16-30"

    if day <= 45:

        return "31-45"

    return "46-60"


def _forecast_type(day):

    if day <= 15:

        return "Pronóstico"

    if day <= 30:

        return "Proyección"

    if day <= 45:

        return "Escenario extendido"

    return "Tendencia hidrológica"


# ============================================================
# PESOS POR HORIZONTE
#
# ML pierde peso con el horizonte.
# Escenario histórico gana peso.
# ============================================================

def _horizon_weights(day):

    if day <= 15:

        return {

            "ml":
                0.58,

            "hydrology":
                0.24,

            "scenario":
                0.10,

            "trend":
                0.08,
        }

    if day <= 30:

        return {

            "ml":
                0.45,

            "hydrology":
                0.27,

            "scenario":
                0.18,

            "trend":
                0.10,
        }

    if day <= 45:

        return {

            "ml":
                0.31,

            "hydrology":
                0.26,

            "scenario":
                0.31,

            "trend":
                0.12,
        }

    return {

        "ml":
            0.22,

        "hydrology":
            0.23,

        "scenario":
            0.42,

        "trend":
            0.13,
    }


# ============================================================
# INCERTIDUMBRE
# ============================================================

def _uncertainty(
    rmse,
    day,
):

    rmse = max(
        _safe_float(
            rmse,
            MIN_RMSE,
        ),
        MIN_RMSE,
    )

    if day <= 15:

        factor = 1.00

    elif day <= 30:

        factor = 1.20

    elif day <= 45:

        factor = 1.45

    else:

        factor = 1.75

    uncertainty = (
        rmse
        * math.sqrt(
            max(
                day,
                1,
            )
        )
        * factor
    )

    return float(
        np.clip(
            uncertainty,
            MIN_RMSE,
            MAX_UNCERTAINTY,
        )
    )


# ============================================================
# TENDENCIA LOCAL
# ============================================================

def _recent_local_trend(
    dataset,
):

    values = (
        _numeric(
            dataset[
                "nivel"
            ]
        )
        .dropna()
        .tail(10)
    )

    if len(values) < 3:

        return 0.0

    try:

        slope = np.polyfit(

            np.arange(
                len(values)
            ),

            values.to_numpy(
                dtype=float
            ),

            1,
        )[0]

    except Exception:

        slope = 0.0

    return float(
        np.clip(
            slope,
            -0.12,
            0.12,
        )
    )


# ============================================================
# PREPARAR FUTURO
# ============================================================

def _prepare_future(
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

        return pd.DataFrame()

    result = df.copy()

    if "datetime" in result.columns:

        result[
            "datetime"
        ] = _datetime_naive(
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
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# VALOR FUTURO
# ============================================================

def _future_value(
    future,
    index,
    column,
):

    if (
        future is None
        or future.empty
        or column
        not in future.columns
        or index
        >= len(future)
    ):

        return np.nan

    return _safe_float(
        future.iloc[
            index
        ].get(
            column
        )
    )


# ============================================================
# PROYECTAR NIVEL UPSTREAM
# ============================================================

def _project_upstream_value(
    history,
    col,
):

    if col not in history.columns:

        return np.nan

    values = (
        _numeric(
            history[col]
        )
        .dropna()
    )

    if values.empty:

        return np.nan

    current = float(
        values.iloc[-1]
    )

    recent = values.tail(
        min(
            7,
            len(values),
        )
    )

    if len(recent) >= 3:

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

    slope = float(
        np.clip(
            slope,
            -0.15,
            0.15,
        )
    )

    return current + slope


# ============================================================
# PROYECTAR CAUDAL SI NO VIENE
# ============================================================

def _project_flow_value(
    history,
    col,
):

    if col not in history.columns:

        return np.nan

    q = (
        _numeric(
            history[col]
        )
        .dropna()
    )

    if q.empty:

        return np.nan

    current = float(
        q.iloc[-1]
    )

    recent = q.tail(
        min(
            14,
            len(q),
        )
    )

    if len(recent) >= 4:

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

    max_change = max(
        current * 0.025,
        50.0,
    )

    slope = float(
        np.clip(
            slope,
            -max_change,
            max_change,
        )
    )

    return max(
        current
        + slope * 0.75,
        0.0,
    )


# ============================================================
# CREAR FILA FUTURA
# ============================================================

def crear_fila_futura(
    history,
    date,
    day,
    exog_future,
    upstream_future,
):

    row = {

        "datetime":
            pd.Timestamp(
                date
            ),

        "nivel":
            float(
                history[
                    "nivel"
                ].iloc[-1]
            ),
    }

    future_index = (
        day - 1
    )

    # ========================================================
    # LLUVIA
    # ========================================================

    for station, col in (
        RAIN_COLUMNS.items()
    ):

        rain_value = (
            _future_value(
                exog_future,
                future_index,
                col,
            )
        )

        if not np.isfinite(
            rain_value
        ):

            rain_value = 0.0

        row[col] = max(
            rain_value,
            0.0,
        )

    # ========================================================
    # CAUDAL
    # ========================================================

    for station, col in (
        FLOW_COLUMNS.items()
    ):

        q = _future_value(
            exog_future,
            future_index,
            col,
        )

        if not np.isfinite(
            q
        ):

            q = _project_flow_value(
                history,
                col,
            )

        row[col] = q

        quality_col = (
            col
            + "_quality"
        )

        quality = _future_value(
            exog_future,
            future_index,
            quality_col,
        )

        if not np.isfinite(
            quality
        ):

            quality = max(
                0.25,
                0.70
                * np.exp(
                    -day / 60.0
                ),
            )

        row[
            quality_col
        ] = float(
            np.clip(
                quality,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # LEGACY CAUDAL
    # ========================================================

    legacy_flow = _future_value(
        exog_future,
        future_index,
        "caudal_m3s",
    )

    if np.isfinite(
        legacy_flow
    ):

        row[
            "caudal_m3s"
        ] = legacy_flow

    # ========================================================
    # UPSTREAM
    # ========================================================

    for station, col in (
        LEVEL_COLUMNS.items()
    ):

        value = _future_value(
            upstream_future,
            future_index,
            col,
        )

        if not np.isfinite(
            value
        ):

            value = _project_upstream_value(
                history,
                col,
            )

        row[col] = value

    return row


# ============================================================
# ESCENARIO HIDROLÓGICO PARA UN DÍA
# ============================================================

def _hydrology_scenario_level(
    hydrology,
    day,
    scenario_name,
):

    if not isinstance(
        hydrology,
        dict,
    ):

        return np.nan

    key_map = {

        "probable":
            "scenario_probable",

        "adverse":
            "scenario_adverse",

        "extreme":
            "scenario_extreme",
    }

    key = key_map[
        scenario_name
    ]

    scenario_df = hydrology.get(
        key
    )

    if (
        not isinstance(
            scenario_df,
            pd.DataFrame,
        )
        or scenario_df.empty
    ):

        return np.nan

    if "scenario_day" in scenario_df.columns:

        match = scenario_df[
            scenario_df[
                "scenario_day"
            ]
            == day
        ]

        if not match.empty:

            if "level" in match.columns:

                return _safe_float(
                    match.iloc[0][
                        "level"
                    ]
                )

    index = day - 1

    if (
        index < len(
            scenario_df
        )
        and "level"
        in scenario_df.columns
    ):

        return _safe_float(
            scenario_df.iloc[
                index
            ][
                "level"
            ]
        )

    return np.nan


# ============================================================
# SEÑAL HIDROLÓGICA DE FUTURO
# ============================================================

def calcular_senal_hidrologica(
    history,
    future_row,
    hydrology=None,
):

    # ========================================================
    # NIVELES
    # ========================================================

    level_signals = []

    level_weights = []

    for station, col in (
        LEVEL_COLUMNS.items()
    ):

        if col not in history.columns:

            continue

        values = (
            _numeric(
                history[col]
            )
            .dropna()
            .tail(7)
        )

        future_value = _safe_float(
            future_row.get(
                col
            )
        )

        if (
            len(values) < 2
            or not np.isfinite(
                future_value
            )
        ):

            continue

        baseline = float(
            values.iloc[-1]
        )

        signal = (
            future_value
            - baseline
        )

        signal = float(
            np.clip(
                signal,
                -0.50,
                0.80,
            )
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

    for station, col in (
        FLOW_COLUMNS.items()
    ):

        if col not in history.columns:

            continue

        q_hist = (
            _numeric(
                history[col]
            )
            .dropna()
            .tail(14)
        )

        future_q = _safe_float(
            future_row.get(
                col
            )
        )

        if (
            q_hist.empty
            or not np.isfinite(
                future_q
            )
        ):

            continue

        baseline = float(
            q_hist.median()
        )

        if baseline <= 0:

            continue

        relative = (
            future_q
            - baseline
        ) / baseline

        relative = float(
            np.clip(
                relative,
                -0.50,
                1.50,
            )
        )

        quality = _safe_float(
            future_row.get(
                col
                + "_quality"
            ),
            0.5,
        )

        quality = float(
            np.clip(
                quality,
                0.0,
                1.0,
            )
        )

        flow_signals.append(
            relative
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
    # LLUVIA
    # ========================================================

    rain_values = []

    rain_weights = []

    for station, col in (
        RAIN_COLUMNS.items()
    ):

        rain = _safe_float(
            future_row.get(
                col
            ),
            0.0,
        )

        rain_values.append(
            max(
                rain,
                0.0,
            )
        )

        rain_weights.append(
            STATION_WEIGHTS[
                station
            ]
        )

    if rain_values:

        rain_signal = float(
            np.average(
                rain_values,
                weights=
                    rain_weights,
            )
        )

    else:

        rain_signal = 0.0

    rain_signal = float(
        np.clip(
            rain_signal
            / 40.0,
            0.0,
            2.0,
        )
    )

    # ========================================================
    # PRESIÓN ACTUAL DE HYDROLOGY.PY
    # ========================================================

    pressure_signal = 0.0

    if isinstance(
        hydrology,
        dict,
    ):

        pressure = hydrology.get(
            "pressure",
            {}
        )

        if isinstance(
            pressure,
            dict,
        ):

            pressure_signal = (
                _safe_float(
                    pressure.get(
                        "hydrological_pressure"
                    ),
                    0.0,
                )
            )

    # ========================================================
    # SEÑAL COMBINADA
    # ========================================================

    combined = (

        0.38
        * level_signal

        +

        0.32
        * flow_signal

        +

        0.18
        * rain_signal

        +

        0.12
        * pressure_signal
    )

    return float(
        np.clip(
            combined,
            -0.15,
            0.20,
        )
    )


# ============================================================
# LÍMITES DE CAMBIO DIARIO
# ============================================================

def _daily_change_limits(
    day,
    hydro_signal,
):

    if day <= 15:

        max_rise = 0.20
        max_drop = -0.15

    elif day <= 30:

        max_rise = 0.16
        max_drop = -0.12

    elif day <= 45:

        max_rise = 0.13
        max_drop = -0.10

    else:

        max_rise = 0.10
        max_drop = -0.08

    # ========================================================
    # Si aguas arriba está creciendo,
    # limitar las caídas bruscas.
    # ========================================================

    if hydro_signal > 0.05:

        max_drop = max(
            max_drop,
            -0.06,
        )

    if hydro_signal > 0.10:

        max_drop = max(
            max_drop,
            -0.035,
        )

    return (
        max_drop,
        max_rise,
    )


# ============================================================
# PREDICT
# ============================================================

def predict(
    df,
    models=None,
    days=15,
    exog_future=None,
    upstream_future=None,
    hydrology=None,
    **kwargs,
):

    # ========================================================
    # COMPATIBILIDAD:
    # predict(models, df, ...)
    # ========================================================

    if isinstance(
        df,
        dict,
    ) and isinstance(
        models,
        pd.DataFrame,
    ):

        df, models = (
            models,
            df,
        )

    if not isinstance(
        models,
        dict,
    ):

        raise ValueError(
            "El modelo no está entrenado."
        )

    model = models.get(
        "model"
    )

    if model is None:

        raise ValueError(
            "El modelo no está entrenado."
        )

    feature_names = models.get(
        "feature_names",
        []
    )

    fill_values = models.get(
        "feature_fill_values",
        {}
    )

    if not feature_names:

        raise ValueError(
            "El modelo no contiene variables de entrenamiento."
        )

    days = int(
        np.clip(
            days,
            1,
            MAX_FORECAST_DAYS,
        )
    )

    # ========================================================
    # BASE
    # ========================================================

    dataset = models.get(
        "dataset"
    )

    if (
        not isinstance(
            dataset,
            pd.DataFrame,
        )
        or dataset.empty
    ):

        dataset = preparar_dataset(
            df
        )

    history = dataset.copy()

    history[
        "datetime"
    ] = _datetime_naive(
        history[
            "datetime"
        ]
    )

    history = history.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # HIDROLOGÍA
    # ========================================================

    if hydrology is None:

        hydrology = models.get(
            "hydrology"
        )

    propagation_lags = (
        models.get(
            "propagation_lags",
            DEFAULT_PROPAGATION_LAGS,
        )
    )

    # ========================================================
    # FUTURO
    # ========================================================

    exog_future = _prepare_future(
        exog_future
    )

    upstream_future = _prepare_future(
        upstream_future
    )

    last_date = (
        history[
            "datetime"
        ].max()
    )

    last_level = float(
        _numeric(
            history[
                "nivel"
            ]
        )
        .dropna()
        .iloc[-1]
    )

    recent_trend = (
        _recent_local_trend(
            history
        )
    )

    rmse = _safe_float(
        models.get(
            "rmse"
        ),
        0.10,
    )

    previous_delta = 0.0

    output = []

    # ========================================================
    # LOOP RECURSIVO
    # ========================================================

    for day in range(
        1,
        days + 1,
    ):

        forecast_date = (
            last_date
            + pd.Timedelta(
                days=day
            )
        )

        future_row = (
            crear_fila_futura(

                history,

                forecast_date,

                day,

                exog_future,

                upstream_future,
            )
        )

        # ====================================================
        # INSERTAR FILA TEMPORAL PARA GENERAR FEATURES
        # ====================================================

        temporary = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        future_row
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )

        feature_table = crear_features(

            temporary,

            propagation_lags=
                propagation_lags,
        )

        feature_row = (
            feature_table.iloc[
                -1
            ]
        )

        X_values = []

        for feature in feature_names:

            value = _safe_float(
                feature_row.get(
                    feature
                )
            )

            if not np.isfinite(
                value
            ):

                value = _safe_float(
                    fill_values.get(
                        feature
                    ),
                    0.0,
                )

            X_values.append(
                value
            )

        X_pred = pd.DataFrame(
            [
                X_values
            ],
            columns=
                feature_names,
        )

        # ====================================================
        # MODELO ML
        # ====================================================

        ml_delta = float(
            model.predict(
                X_pred
            )[0]
        )

        ml_delta = float(
            np.clip(
                ml_delta,
                -0.25,
                0.25,
            )
        )

        # ====================================================
        # SEÑAL HIDROLÓGICA
        # ====================================================

        hydro_signal = (
            calcular_senal_hidrologica(

                history,

                future_row,

                hydrology=
                    hydrology,
            )
        )

        # ====================================================
        # ESCENARIO HISTÓRICO PROBABLE
        # ====================================================

        probable_level = (
            _hydrology_scenario_level(
                hydrology,
                day,
                "probable",
            )
        )

        adverse_level = (
            _hydrology_scenario_level(
                hydrology,
                day,
                "adverse",
            )
        )

        extreme_level = (
            _hydrology_scenario_level(
                hydrology,
                day,
                "extreme",
            )
        )

        current_level = float(
            history[
                "nivel"
            ].iloc[-1]
        )

        if np.isfinite(
            probable_level
        ):

            scenario_delta = (
                probable_level
                - current_level
            )

        else:

            scenario_delta = 0.0

        scenario_delta = float(
            np.clip(
                scenario_delta,
                -0.20,
                0.25,
            )
        )

        # ====================================================
        # TENDENCIA
        # ====================================================

        trend_delta = (
            recent_trend
            * np.exp(
                -day / 28.0
            )
        )

        # ====================================================
        # PESOS
        # ====================================================

        weights = _horizon_weights(
            day
        )

        combined_delta = (

            weights[
                "ml"
            ]
            * ml_delta

            +

            weights[
                "hydrology"
            ]
            * hydro_signal

            +

            weights[
                "scenario"
            ]
            * scenario_delta

            +

            weights[
                "trend"
            ]
            * trend_delta
        )

        # ====================================================
        # SUAVIZAR
        # ====================================================

        smoothed_delta = (

            0.68
            * previous_delta

            +

            0.32
            * combined_delta
        )

        min_change, max_change = (
            _daily_change_limits(
                day,
                hydro_signal,
            )
        )

        smoothed_delta = float(
            np.clip(
                smoothed_delta,
                min_change,
                max_change,
            )
        )

        prediction = (
            current_level
            + smoothed_delta
        )

        prediction = float(
            np.clip(
                prediction,
                LEVEL_MIN,
                LEVEL_MAX,
            )
        )

        # ====================================================
        # INCERTIDUMBRE
        # ====================================================

        uncertainty = (
            _uncertainty(
                rmse,
                day,
            )
        )

        lower = max(
            LEVEL_MIN,
            prediction
            - uncertainty,
        )

        upper = min(
            LEVEL_MAX,
            prediction
            + uncertainty,
        )

        # ====================================================
        # EVITAR QUE ESCENARIOS QUEDEN INVERTIDOS
        # ====================================================

        if np.isfinite(
            probable_level
        ):

            probable_level = max(
                probable_level,
                LEVEL_MIN,
            )

        if np.isfinite(
            adverse_level
        ):

            if np.isfinite(
                probable_level
            ):

                adverse_level = max(
                    adverse_level,
                    probable_level,
                )

        if np.isfinite(
            extreme_level
        ):

            if np.isfinite(
                adverse_level
            ):

                extreme_level = max(
                    extreme_level,
                    adverse_level,
                )

        # ====================================================
        # CAUDAL / LLUVIA RESUMEN
        # ====================================================

        caudal_value = _safe_float(
            future_row.get(
                "caudal_m3s"
            )
        )

        if not np.isfinite(
            caudal_value
        ):

            # buscar caudal disponible más cercano
            for station in [
                "San Nicolás",
                "Villa Constitución",
                "Rosario",
                "Diamante",
                "Paraná",
                "La Paz",
                "Goya",
                "Corrientes",
            ]:

                col = FLOW_COLUMNS[
                    station
                ]

                q = _safe_float(
                    future_row.get(
                        col
                    )
                )

                if np.isfinite(q):

                    caudal_value = q
                    break

        rain_values = []

        for col in RAIN_COLUMNS.values():

            rain = _safe_float(
                future_row.get(
                    col
                ),
                0.0,
            )

            rain_values.append(
                max(
                    rain,
                    0.0,
                )
            )

        precip_value = (
            float(
                np.mean(
                    rain_values
                )
            )
            if rain_values
            else 0.0
        )

        # ====================================================
        # SALIDA
        # ====================================================

        result_row = {

            "datetime":
                forecast_date,

            "prediction":
                prediction,

            "lower":
                lower,

            "upper":
                upper,

            "uncertainty":
                uncertainty,

            "horizon_day":
                day,

            "horizon_group":
                _horizon_group(
                    day
                ),

            "forecast_type":
                _forecast_type(
                    day
                ),

            "base_level":
                current_level,

            "ml_delta":
                ml_delta,

            "hydro_signal":
                hydro_signal,

            "scenario_delta":
                scenario_delta,

            "trend_delta":
                trend_delta,

            "daily_change":
                smoothed_delta,

            "weight_ml":
                weights[
                    "ml"
                ],

            "weight_hydrology":
                weights[
                    "hydrology"
                ],

            "weight_scenario":
                weights[
                    "scenario"
                ],

            "weight_trend":
                weights[
                    "trend"
                ],

            "scenario_probable":
                probable_level,

            "scenario_adverse":
                adverse_level,

            "scenario_extreme":
                extreme_level,

            "precip_mm":
                precip_value,

            "caudal_m3s":
                caudal_value,
        }

        # ====================================================
        # VARIABLES POR ESTACIÓN
        # ====================================================

        for station, col in (
            LEVEL_COLUMNS.items()
        ):

            result_row[col] = (
                future_row.get(
                    col
                )
            )

        for station, col in (
            FLOW_COLUMNS.items()
        ):

            result_row[col] = (
                future_row.get(
                    col
                )
            )

            result_row[
                col
                + "_quality"
            ] = (
                future_row.get(
                    col
                    + "_quality"
                )
            )

        for station, col in (
            RAIN_COLUMNS.items()
        ):

            result_row[col] = (
                future_row.get(
                    col,
                    0.0,
                )
            )

        output.append(
            result_row
        )

        # ====================================================
        # AGREGAR PREDICCIÓN A HISTORIA
        # ====================================================

        future_row[
            "nivel"
        ] = prediction

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        future_row
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )

        previous_delta = (
            smoothed_delta
        )

    return pd.DataFrame(
        output
    )


# ============================================================
# RESUMEN DE ESTACIONES
# ============================================================

def resumen_niveles_estaciones(
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

        return pd.DataFrame()

    rows = []

    for station, col in (
        LEVEL_COLUMNS.items()
    ):

        if col not in df.columns:

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

        values = (
            _numeric(
                df[col]
            )
            .dropna()
        )

        if values.empty:

            continue

        current = float(
            values.iloc[-1]
        )

        delta_1 = (
            current
            - float(
                values.iloc[-2]
            )
            if len(values) >= 2
            else np.nan
        )

        delta_3 = (
            current
            - float(
                values.iloc[-4]
            )
            if len(values) >= 4
            else np.nan
        )

        delta_7 = (
            current
            - float(
                values.iloc[-8]
            )
            if len(values) >= 8
            else np.nan
        )

        if np.isfinite(
            delta_3
        ):

            if delta_3 > 0.03:

                state = "↑ Creciendo"

            elif delta_3 < -0.03:

                state = "↓ Bajando"

            else:

                state = "→ Estable"

        else:

            state = "Sin tendencia"

        rows.append(
            {
                "Estación":
                    station,

                "Nivel":
                    current,

                "Δ 1 día":
                    delta_1,

                "Δ 3 días":
                    delta_3,

                "Δ 7 días":
                    delta_7,

                "Estado":
                    state,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BACKTEST ORIENTATIVO
# ============================================================

def backtest_horizontes(
    df,
    models,
    horizons=None,
):

    if horizons is None:

        horizons = [
            15,
            30,
            45,
            60,
        ]

    if (
        df is None
        or df.empty
    ):

        return pd.DataFrame()

    x = preparar_nivel_local(
        df
    )

    rows = []

    for horizon in horizons:

        if len(x) <= horizon:

            continue

        actual = x[
            "nivel"
        ].shift(
            -horizon
        )

        baseline = x[
            "nivel"
        ]

        valid = (
            actual.notna()
            &
            baseline.notna()
        )

        if not valid.any():

            continue

        naive_error = (
            actual[
                valid
            ]
            -
            baseline[
                valid
            ]
        )

        rmse = math.sqrt(
            np.mean(
                naive_error ** 2
            )
        )

        mae = np.mean(
            np.abs(
                naive_error
            )
        )

        rows.append(
            {
                "horizon":
                    horizon,

                "rmse_reference":
                    float(
                        rmse
                    ),

                "mae_reference":
                    float(
                        mae
                    ),

                "records":
                    int(
                        valid.sum()
                    ),

                "type":
                    "referencia_persistencia",
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# RESUMEN BACKTEST
# ============================================================

def resumen_backtest(
    backtest,
):

    if (
        backtest is None
        or backtest.empty
    ):

        return {}

    result = {}

    for _, row in (
        backtest.iterrows()
    ):

        horizon = _safe_int(
            row.get(
                "horizon"
            ),
            0,
        )

        result[
            horizon
        ] = {

            "rmse":
                _safe_float(
                    row.get(
                        "rmse_reference"
                    )
                ),

            "mae":
                _safe_float(
                    row.get(
                        "mae_reference"
                    )
                ),

            "records":
                _safe_int(
                    row.get(
                        "records"
                    )
                ),
        }

    return result


# ============================================================
# PROBABILIDAD / RANGO
# ============================================================

def prob(
    forecast,
):

    if (
        forecast is None
        or not isinstance(
            forecast,
            pd.DataFrame,
        )
        or forecast.empty
    ):

        return {}

    last = forecast.iloc[
        -1
    ]

    return {

        "prediction":
            _safe_float(
                last.get(
                    "prediction"
                )
            ),

        "lower":
            _safe_float(
                last.get(
                    "lower"
                )
            ),

        "upper":
            _safe_float(
                last.get(
                    "upper"
                )
            ),

        "scenario_probable":
            _safe_float(
                last.get(
                    "scenario_probable"
                )
            ),

        "scenario_adverse":
            _safe_float(
                last.get(
                    "scenario_adverse"
                )
            ),

        "scenario_extreme":
            _safe_float(
                last.get(
                    "scenario_extreme"
                )
            ),
    }


# ============================================================
# DIAGNÓSTICO
# ============================================================

def diagnostic(
    df,
    exog_history=None,
    upstream_history=None,
    hydrology=None,
):

    result = {

        "version":
            VERSION,

        "status":
            "pendiente",
    }

    try:

        models, metrics = train(

            df,

            exog_history=
                exog_history,

            upstream_history=
                upstream_history,

            hydrology=
                hydrology,
        )

        result[
            "status"
        ] = "ok"

        result[
            "metrics"
        ] = metrics

        result[
            "feature_count"
        ] = models.get(
            "feature_count"
        )

        result[
            "training_rows"
        ] = models.get(
            "training_rows"
        )

        result[
            "uses_upstream"
        ] = models.get(
            "uses_upstream"
        )

        result[
            "uses_caudal"
        ] = models.get(
            "uses_caudal"
        )

        result[
            "uses_rain"
        ] = models.get(
            "uses_rain"
        )

        result[
            "uses_hydrology"
        ] = models.get(
            "uses_hydrology"
        )

        result[
            "upstream_features"
        ] = models.get(
            "upstream_feature_count"
        )

        result[
            "flow_features"
        ] = models.get(
            "flow_feature_count"
        )

        result[
            "rain_features"
        ] = models.get(
            "rain_feature_count"
        )

        result[
            "hydrology_features"
        ] = models.get(
            "hydrology_feature_count"
        )

        result[
            "available_upstream"
        ] = models.get(
            "available_upstream_columns"
        )

        result[
            "available_flow"
        ] = models.get(
            "available_flow_columns"
        )

        result[
            "available_rain"
        ] = models.get(
            "available_rain_columns"
        )

        importance = models.get(
            "importance"
        )

        if (
            isinstance(
                importance,
                pd.DataFrame,
            )
            and not importance.empty
        ):

            result[
                "top_features"
            ] = (
                importance
                .head(20)
                .to_dict(
                    orient="records"
                )
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
