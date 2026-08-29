# ============================================================
# PARANÁ · SAN NICOLÁS
# src/model.py
# V11.10.1 COMPLETO
#
# MODELO HIDROLÓGICO MULTIVARIABLE
#
# ------------------------------------------------------------
# VARIABLES:
# - Nivel San Nicolás
# - Niveles aguas arriba
# - Caudales INA por estación
# - Precipitación por estación
# - Tendencias
# - Lags
# - Acumulados
# - Señales hidrológicas
#
# HORIZONTES:
# - 1-15 días
# - 16-30 días
# - 31-45 días
# - 46-60 días
#
# COMPATIBILIDAD:
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
# También soporta por compatibilidad:
#
# predict(
#     models,
#     df,
#     ...
# )
# ============================================================


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)


# ============================================================
# VERSIÓN
# ============================================================

VERSION = "V11.10.1"


# ============================================================
# CONFIGURACIÓN
# ============================================================

MIN_OBSERVATIONS = 60

MIN_TRAINING_ROWS = 35

MAX_FORECAST_DAYS = 60

MIN_RMSE = 0.03

MAX_UNCERTAINTY = 1.80

LEVEL_MIN = -2.0

LEVEL_MAX = 12.0


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
#
# Más peso a estaciones cercanas a San Nicolás.
# ============================================================

STATION_WEIGHTS = {

    "Corrientes":
        0.75,

    "Goya":
        0.82,

    "La Paz":
        0.90,

    "Paraná":
        1.00,

    "Diamante":
        1.08,

    "Rosario":
        1.18,

    "Villa Constitución":
        1.30,

    "San Nicolás":
        1.40,
}


# ============================================================
# RETARDOS DE REFERENCIA
#
# Se utilizan sólo como punto de partida.
# Si hydrology aporta retardos históricos, se reemplazan.
# ============================================================

DEFAULT_PROPAGATION_LAGS = {

    "Corrientes":
        18,

    "Goya":
        15,

    "La Paz":
        12,

    "Paraná":
        8,

    "Diamante":
        6,

    "Rosario":
        3,

    "Villa Constitución":
        1,
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


def _numeric(
    values,
):

    return pd.to_numeric(
        values,
        errors="coerce",
    )


def _safe_float(
    value,
    default=None,
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


# ============================================================
# NORMALIZAR NIVEL LOCAL
# ============================================================

def preparar_nivel_local(
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

        raise ValueError(
            "No existen datos de San Nicolás."
        )


    x = df.copy()


    if "datetime" not in x.columns:

        raise ValueError(
            "Los datos no contienen columna datetime."
        )


    x[
        "datetime"
    ] = _datetime_naive(
        x[
            "datetime"
        ]
    )


    if "nivel" in x.columns:

        source_col = "nivel"

    elif "value" in x.columns:

        source_col = "value"

    elif "nivel_san_nicolas" in x.columns:

        source_col = "nivel_san_nicolas"

    else:

        raise ValueError(
            "No existe columna de nivel de San Nicolás."
        )


    x[
        "nivel"
    ] = _numeric(
        x[
            source_col
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


    if len(
        x
    ) < MIN_OBSERVATIONS:

        raise ValueError(
            "Se necesitan al menos "
            f"{MIN_OBSERVATIONS} observaciones "
            "de San Nicolás para entrenar el modelo."
        )


    return x


# ============================================================
# NORMALIZAR EXÓGENAS
# ============================================================

def preparar_exogenas(
    df,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "datetime"
        not in df.columns
    ):

        return pd.DataFrame()


    x = df.copy()


    x[
        "datetime"
    ] = _datetime_naive(
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


    numeric_cols = [

        col

        for col in x.columns

        if col != "datetime"
    ]


    for col in numeric_cols:

        x[
            col
        ] = _numeric(
            x[
                col
            ]
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


    return x


# ============================================================
# NORMALIZAR AGUAS ARRIBA
# ============================================================

def preparar_upstream(
    df,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "datetime"
        not in df.columns
    ):

        return pd.DataFrame()


    x = df.copy()


    x[
        "datetime"
    ] = _datetime_naive(
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


    for col in LEVEL_COLUMNS.values():

        if col in x.columns:

            x[
                col
            ] = _numeric(
                x[
                    col
                ]
            )


    # --------------------------------------------------------
    # Reducir a una fila diaria.
    # --------------------------------------------------------

    numeric_cols = [

        col

        for col in LEVEL_COLUMNS.values()

        if col in x.columns
    ]


    if not numeric_cols:

        return pd.DataFrame(
            {
                "datetime":
                    x[
                        "datetime"
                    ]
            }
        ).drop_duplicates()


    x = (
        x[
            [
                "datetime",
                *numeric_cols,
            ]
        ]
        .groupby(
            "datetime",
            as_index=False,
        )
        .mean(
            numeric_only=True
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


    return x


# ============================================================
# DATASET UNIFICADO
# ============================================================

def preparar_dataset(
    df,
    exog_history=None,
    upstream_history=None,
):

    local = preparar_nivel_local(
        df
    )


    dataset = local.copy()


    exog = preparar_exogenas(
        exog_history
    )


    upstream = preparar_upstream(
        upstream_history
    )


    # ========================================================
    # MERGE AGUAS ARRIBA
    # ========================================================

    if not upstream.empty:

        dataset = dataset.merge(
            upstream,
            on="datetime",
            how="left",
        )


    # ========================================================
    # MERGE EXÓGENAS
    # ========================================================

    if not exog.empty:

        # Evitar duplicados accidentales.
        exog_cols = [

            col

            for col in exog.columns

            if (
                col == "datetime"
                or col not in dataset.columns
            )
        ]


        dataset = dataset.merge(
            exog[
                exog_cols
            ],
            on="datetime",
            how="left",
        )


    dataset = dataset.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )


    # ========================================================
    # INTERPOLACIÓN CORTA NIVELES UPSTREAM
    # ========================================================

    for col in LEVEL_COLUMNS.values():

        if col not in dataset.columns:
            continue


        values = _numeric(
            dataset[
                col
            ]
        )


        if values.notna().sum() >= 3:

            dataset[
                col
            ] = (
                values
                .interpolate(
                    limit=5,
                    limit_area="inside",
                )
            )


    # ========================================================
    # INTERPOLACIÓN CORTA DE CAUDALES
    # ========================================================

    for col in FLOW_COLUMNS.values():

        if col not in dataset.columns:
            continue


        values = _numeric(
            dataset[
                col
            ]
        )


        if values.notna().sum() >= 3:

            dataset[
                col
            ] = (
                values
                .interpolate(
                    limit=5,
                    limit_area="inside",
                )
            )


    return dataset


# ============================================================
# RETARDOS HIDROLÓGICOS
# ============================================================

def obtener_lags_hidrologicos(
    hydrology=None,
):

    lags = (
        DEFAULT_PROPAGATION_LAGS.copy()
    )


    if not isinstance(
        hydrology,
        dict,
    ):

        return lags


    corridor = hydrology.get(
        "corridor_lags"
    )


    if (
        corridor is None
        or not isinstance(
            corridor,
            pd.DataFrame,
        )
        or corridor.empty
    ):

        # Corrientes directo como mínimo.
        current_estimate = (
            hydrology.get(
                "current_estimate",
                {}
            )
        )


        if isinstance(
            current_estimate,
            dict,
        ):

            delay = _safe_int(
                current_estimate.get(
                    "delay_days"
                ),
                0,
            )


            if 1 <= delay <= 40:

                lags[
                    "Corrientes"
                ] = delay


        return lags


    # --------------------------------------------------------
    # Intentar interpretar retardos por tramo.
    # Construimos retardo acumulado hasta San Nicolás.
    # --------------------------------------------------------

    segment_lags = {}


    for _, row in corridor.iterrows():

        upstream_station = None
        downstream_station = None


        for key in [
            "upstream",
            "station_upstream",
            "from_station",
            "origen",
            "desde",
        ]:

            if key in corridor.columns:

                upstream_station = row.get(
                    key
                )

                if upstream_station is not None:
                    break


        for key in [
            "downstream",
            "station_downstream",
            "to_station",
            "destino",
            "hasta",
        ]:

            if key in corridor.columns:

                downstream_station = row.get(
                    key
                )

                if downstream_station is not None:
                    break


        lag = None


        for key in [
            "lag_days",
            "delay_days",
            "lag",
            "retardo_dias",
        ]:

            if key in corridor.columns:

                lag = _safe_int(
                    row.get(
                        key
                    ),
                    0,
                )

                if lag > 0:
                    break


        if (
            upstream_station is None
            or downstream_station is None
            or lag is None
            or lag <= 0
        ):

            continue


        segment_lags[
            (
                str(
                    upstream_station
                ),
                str(
                    downstream_station
                ),
            )
        ] = lag


    # --------------------------------------------------------
    # Si no se pudo interpretar, conservar defaults.
    # --------------------------------------------------------

    if not segment_lags:

        return lags


    # --------------------------------------------------------
    # Acumular desde cada estación hacia San Nicolás.
    # --------------------------------------------------------

    order = STATIONS


    for idx, station in enumerate(
        order[
            :-1
        ]
    ):

        total = 0

        valid = True


        for i in range(
            idx,
            len(
                order
            )
            - 1,
        ):

            pair = (
                order[
                    i
                ],
                order[
                    i + 1
                ],
            )


            segment = None


            # Coincidencia exacta.
            for key, value in segment_lags.items():

                if (
                    str(
                        key[0]
                    ).lower()
                    == str(
                        pair[0]
                    ).lower()
                    and
                    str(
                        key[1]
                    ).lower()
                    == str(
                        pair[1]
                    ).lower()
                ):

                    segment = value
                    break


            if segment is None:

                valid = False
                break


            total += segment


        if (
            valid
            and 1 <= total <= 45
        ):

            lags[
                station
            ] = total


    return lags


# ============================================================
# FEATURES NIVEL LOCAL
# ============================================================

def agregar_features_locales(
    x,
):

    result = x.copy()


    level = _numeric(
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
        ] = level.shift(
            lag
        )


    for lag in [
        1,
        3,
        7,
        14,
        30,
    ]:

        result[
            f"nivel_diff_{lag}"
        ] = level.diff(
            lag
        )


    for window in [
        3,
        7,
        14,
        30,
    ]:

        result[
            f"nivel_mean_{window}"
        ] = (
            level
            .rolling(
                window,
                min_periods=2,
            )
            .mean()
        )


        result[
            f"nivel_std_{window}"
        ] = (
            level
            .rolling(
                window,
                min_periods=2,
            )
            .std()
        )


        result[
            f"nivel_min_{window}"
        ] = (
            level
            .rolling(
                window,
                min_periods=2,
            )
            .min()
        )


        result[
            f"nivel_max_{window}"
        ] = (
            level
            .rolling(
                window,
                min_periods=2,
            )
            .max()
        )


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
# FEATURES NIVELES AGUAS ARRIBA
# ============================================================

def agregar_features_upstream(
    x,
    propagation_lags=None,
):

    result = x.copy()


    if propagation_lags is None:

        propagation_lags = (
            DEFAULT_PROPAGATION_LAGS
        )


    level_signal_parts = []


    for station, col in LEVEL_COLUMNS.items():

        if col not in result.columns:
            continue


        values = _numeric(
            result[
                col
            ]
        )


        if values.notna().sum() < 3:
            continue


        # ----------------------------------------------------
        # Valores y cambios.
        # ----------------------------------------------------

        for lag in [
            1,
            3,
            7,
            14,
        ]:

            result[
                f"{col}_diff_{lag}"
            ] = values.diff(
                lag
            )


        for window in [
            3,
            7,
            14,
        ]:

            result[
                f"{col}_mean_{window}"
            ] = (
                values
                .rolling(
                    window,
                    min_periods=2,
                )
                .mean()
            )


        result[
            f"{col}_trend_7"
        ] = (
            values
            -
            values.shift(
                7
            )
        ) / 7.0


        # ----------------------------------------------------
        # Lags generales.
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


        # ----------------------------------------------------
        # Lag hidrológico estimado hacia San Nicolás.
        # ----------------------------------------------------

        hydro_lag = _safe_int(
            propagation_lags.get(
                station
            ),
            DEFAULT_PROPAGATION_LAGS.get(
                station,
                1,
            ),
        )


        hydro_lag = max(
            1,
            min(
                hydro_lag,
                45,
            ),
        )


        result[
            f"{col}_lag_propagacion"
        ] = values.shift(
            hydro_lag
        )


        # ----------------------------------------------------
        # Señal ponderada por tendencia.
        # ----------------------------------------------------

        trend_component = (
            result[
                f"{col}_trend_7"
            ]
            *
            STATION_WEIGHTS.get(
                station,
                1.0,
            )
        )


        level_signal_parts.append(
            trend_component
        )


    if level_signal_parts:

        signal_df = pd.concat(
            level_signal_parts,
            axis=1,
        )


        result[
            "upstream_level_signal"
        ] = signal_df.mean(
            axis=1,
            skipna=True,
        )


    else:

        result[
            "upstream_level_signal"
        ] = np.nan


    return result


# ============================================================
# FEATURES CAUDAL
# ============================================================

def agregar_features_caudal(
    x,
):

    result = x.copy()


    flow_signal_parts = []


    for station, col in FLOW_COLUMNS.items():

        if col not in result.columns:
            continue


        q = _numeric(
            result[
                col
            ]
        )


        if q.notna().sum() < 3:
            continue


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
                    min_periods=2,
                )
                .mean()
            )


        result[
            f"{col}_trend_7"
        ] = (
            q
            -
            q.shift(
                7
            )
        ) / 7.0


        result[
            f"{col}_relative_7"
        ] = (
            q
            -
            q.shift(
                7
            )
        ) / q.shift(
            7
        ).replace(
            0,
            np.nan,
        )


        for lag in [
            1,
            3,
            5,
            7,
            10,
            14,
            21,
        ]:

            result[
                f"{col}_lag_{lag}"
            ] = q.shift(
                lag
            )


        normalized_flow_trend = (
            result[
                f"{col}_relative_7"
            ]
            *
            STATION_WEIGHTS.get(
                station,
                1.0,
            )
        )


        flow_signal_parts.append(
            normalized_flow_trend
        )


    if flow_signal_parts:

        signal_df = pd.concat(
            flow_signal_parts,
            axis=1,
        )


        result[
            "upstream_flow_signal"
        ] = signal_df.mean(
            axis=1,
            skipna=True,
        )


    else:

        result[
            "upstream_flow_signal"
        ] = np.nan


    return result


# ============================================================
# FEATURES LLUVIA
# ============================================================

def agregar_features_lluvia(
    x,
):

    result = x.copy()


    weighted_rain = []


    for station, col in RAIN_COLUMNS.items():

        if col not in result.columns:
            continue


        rain = _numeric(
            result[
                col
            ]
        )


        if rain.notna().sum() < 3:
            continue


        rain = rain.clip(
            lower=0
        )


        for window in [
            3,
            7,
            15,
            30,
        ]:

            feature_name = (
                f"{col}_{window}d"
            )


            # Si ya viene desde exogenous lo reemplazamos
            # con el cálculo sobre el dataset unificado.
            result[
                feature_name
            ] = (
                rain
                .fillna(
                    0
                )
                .rolling(
                    window,
                    min_periods=1,
                )
                .sum()
            )


        weighted_rain.append(
            rain.fillna(
                0
            )
            *
            STATION_WEIGHTS.get(
                station,
                1.0,
            )
        )


    if weighted_rain:

        rain_matrix = pd.concat(
            weighted_rain,
            axis=1,
        )


        result[
            "rain_signal"
        ] = rain_matrix.mean(
            axis=1,
            skipna=True,
        )


        result[
            "rain_corridor_3d"
        ] = (
            result[
                "rain_signal"
            ]
            .rolling(
                3,
                min_periods=1,
            )
            .sum()
        )


        result[
            "rain_corridor_7d"
        ] = (
            result[
                "rain_signal"
            ]
            .rolling(
                7,
                min_periods=1,
            )
            .sum()
        )


        result[
            "rain_corridor_15d"
        ] = (
            result[
                "rain_signal"
            ]
            .rolling(
                15,
                min_periods=1,
            )
            .sum()
        )


        result[
            "rain_corridor_30d"
        ] = (
            result[
                "rain_signal"
            ]
            .rolling(
                30,
                min_periods=1,
            )
            .sum()
        )


    else:

        result[
            "rain_signal"
        ] = np.nan


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
        "day_of_year"
    ] = dt.dt.dayofyear


    result[
        "sin_day_of_year"
    ] = np.sin(
        2
        * np.pi
        * result[
            "day_of_year"
        ]
        / 365.25
    )


    result[
        "cos_day_of_year"
    ] = np.cos(
        2
        * np.pi
        * result[
            "day_of_year"
        ]
        / 365.25
    )


    return result


# ============================================================
# CREAR TODAS LAS FEATURES
# ============================================================

def crear_features(
    dataset,
    propagation_lags=None,
):

    x = dataset.copy()


    x = agregar_features_locales(
        x
    )


    x = agregar_features_upstream(
        x,
        propagation_lags=
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


    # --------------------------------------------------------
    # Objetivo = variación del día siguiente.
    # --------------------------------------------------------

    x[
        "target_delta"
    ] = (
        x[
            "nivel"
        ]
        .shift(
            -1
        )
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
    features_df,
):

    excluded = {
        "datetime",
        "nivel",
        "target_delta",
    }


    selected = []


    for col in features_df.columns:

        if col in excluded:
            continue


        values = _numeric(
            features_df[
                col
            ]
        )


        valid_count = int(
            values.notna().sum()
        )


        if valid_count < 20:
            continue


        # Evitar constantes.
        if values.nunique(
            dropna=True
        ) <= 1:
            continue


        selected.append(
            col
        )


    return selected


# ============================================================
# CONTAR GRUPOS DE FEATURES
# ============================================================

def _count_feature_groups(
    feature_names,
):

    upstream_count = 0
    flow_count = 0
    rain_count = 0


    for feature in feature_names:

        name = str(
            feature
        ).lower()


        if (
            name.startswith(
                "nivel_corrientes"
            )
            or name.startswith(
                "nivel_goya"
            )
            or name.startswith(
                "nivel_la_paz"
            )
            or name.startswith(
                "nivel_parana"
            )
            or name.startswith(
                "nivel_diamante"
            )
            or name.startswith(
                "nivel_rosario"
            )
            or name.startswith(
                "nivel_villa_constitucion"
            )
            or name
            == "upstream_level_signal"
        ):

            upstream_count += 1


        if (
            name.startswith(
                "q_"
            )
            or name
            == "caudal_m3s"
            or name
            == "upstream_flow_signal"
        ):

            flow_count += 1


        if (
            name.startswith(
                "rain_"
            )
            or name
            == "precip_mm"
        ):

            rain_count += 1


    return (
        upstream_count,
        flow_count,
        rain_count,
    )


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train(
    df,
    exog_history=None,
    upstream_history=None,
    hydrology=None,
    **kwargs,
):

    """
    Entrena Random Forest para predecir variación diaria.

    Retorna:
        models
        metrics
    """


    # ========================================================
    # DATASET
    # ========================================================

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


    feature_table = crear_features(

        dataset,

        propagation_lags=
            propagation_lags,
    )


    feature_names = seleccionar_features(
        feature_table
    )


    if not feature_names:

        raise ValueError(
            "No se generaron variables suficientes "
            "para entrenar el modelo."
        )


    # ========================================================
    # MEDIANAS PARA RELLENO
    # ========================================================

    fill_values = {}


    for col in feature_names:

        values = _numeric(
            feature_table[
                col
            ]
        )


        median = _safe_float(
            values.median(),
            0.0,
        )


        if median is None:

            median = 0.0


        fill_values[
            col
        ] = median


        feature_table[
            col
        ] = values.fillna(
            median
        )


    # ========================================================
    # TARGET
    # ========================================================

    feature_table[
        "target_delta"
    ] = _numeric(
        feature_table[
            "target_delta"
        ]
    )


    training = feature_table.dropna(
        subset=[
            "target_delta"
        ]
    ).copy()


    if len(
        training
    ) < MIN_TRAINING_ROWS:

        raise ValueError(
            "No existen suficientes filas válidas "
            "para el entrenamiento multivariable. "
            f"Disponibles: {len(training)}."
        )


    X = training[
        feature_names
    ]


    y = training[
        "target_delta"
    ]


    # ========================================================
    # VALIDACIÓN TEMPORAL
    # ========================================================

    split_idx = int(
        len(
            training
        )
        * 0.80
    )


    split_idx = max(
        split_idx,
        MIN_TRAINING_ROWS,
    )


    split_idx = min(
        split_idx,
        len(
            training
        )
        - 5,
    )


    X_train = X.iloc[
        :split_idx
    ]


    y_train = y.iloc[
        :split_idx
    ]


    X_valid = X.iloc[
        split_idx:
    ]


    y_valid = y.iloc[
        split_idx:
    ]


    # ========================================================
    # MODELO VALIDACIÓN
    # ========================================================

    validation_model = RandomForestRegressor(

        n_estimators=
            700,

        max_depth=
            14,

        min_samples_leaf=
            2,

        max_features=
            "sqrt",

        random_state=
            42,

        n_jobs=
            -1,
    )


    validation_model.fit(
        X_train,
        y_train,
    )


    valid_pred = validation_model.predict(
        X_valid
    )


    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_valid,
                valid_pred,
            )
        )
    )


    mae = float(
        mean_absolute_error(
            y_valid,
            valid_pred,
        )
    )


    rmse = max(
        rmse,
        MIN_RMSE,
    )


    # ========================================================
    # MODELO FINAL
    # ========================================================

    model = RandomForestRegressor(

        n_estimators=
            1000,

        max_depth=
            14,

        min_samples_leaf=
            2,

        max_features=
            "sqrt",

        random_state=
            42,

        n_jobs=
            -1,
    )


    model.fit(
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
                model.feature_importances_,
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
    # CONTADORES
    # ========================================================

    (
        upstream_feature_count,
        flow_feature_count,
        rain_feature_count,
    ) = _count_feature_groups(
        feature_names
    )


    # ========================================================
    # COLUMNAS BASE DISPONIBLES
    # ========================================================

    available_upstream_columns = [

        col

        for col in LEVEL_COLUMNS.values()

        if (
            col in dataset.columns
            and _numeric(
                dataset[
                    col
                ]
            )
            .notna()
            .sum()
            >= 3
        )
    ]


    available_flow_columns = [

        col

        for col in FLOW_COLUMNS.values()

        if (
            col in dataset.columns
            and _numeric(
                dataset[
                    col
                ]
            )
            .notna()
            .sum()
            >= 3
        )
    ]


    available_rain_columns = [

        col

        for col in RAIN_COLUMNS.values()

        if (
            col in dataset.columns
            and _numeric(
                dataset[
                    col
                ]
            )
            .notna()
            .sum()
            >= 3
        )
    ]


    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "version":
            VERSION,

        "model":
            model,

        "feature_names":
            feature_names,

        "feature_fill_values":
            fill_values,

        "rmse":
            rmse,

        "mae":
            mae,

        "dataset":
            dataset,

        "features_table":
            feature_table,

        "importance":
            importance,

        "propagation_lags":
            propagation_lags,

        "hydrology":
            hydrology,

        # ----------------------------------------------------
        # APP V11.10
        # ----------------------------------------------------

        "uses_upstream":
            upstream_feature_count > 0,

        "uses_caudal":
            flow_feature_count > 0,

        "uses_rain":
            rain_feature_count > 0,

        "upstream_feature_count":
            upstream_feature_count,

        "flow_feature_count":
            flow_feature_count,

        "rain_feature_count":
            rain_feature_count,

        "available_upstream_columns":
            available_upstream_columns,

        "available_flow_columns":
            available_flow_columns,

        "available_rain_columns":
            available_rain_columns,

        "training_rows":
            int(
                len(
                    training
                )
            ),

        "feature_count":
            int(
                len(
                    feature_names
                )
            ),
    }


    # ========================================================
    # MÉTRICAS
    # ========================================================

    metrics = {

        "version":
            VERSION,

        "RMSE":
            rmse,

        "MAE":
            mae,

        "training_rows":
            int(
                len(
                    training
                )
            ),

        "validation_rows":
            int(
                len(
                    y_valid
                )
            ),

        "features":
            int(
                len(
                    feature_names
                )
            ),

        "upstream_features":
            upstream_feature_count,

        "flow_features":
            flow_feature_count,

        "rain_features":
            rain_feature_count,

        "uses_upstream":
            upstream_feature_count > 0,

        "uses_caudal":
            flow_feature_count > 0,

        "uses_rain":
            rain_feature_count > 0,
    }


    return (
        models,
        metrics,
    )


# ============================================================
# HORIZONTES
# ============================================================

def grupo_horizonte(
    day,
):

    if day <= 15:

        return "1-15"

    if day <= 30:

        return "16-30"

    if day <= 45:

        return "31-45"

    return "46-60"


def tipo_horizonte(
    day,
):

    if day <= 15:

        return "pronostico"

    if day <= 30:

        return "proyeccion"

    if day <= 45:

        return "escenario_probabilistico"

    return "tendencia_extendida"


# ============================================================
# PESOS POR HORIZONTE
# ============================================================

def pesos_horizonte(
    day,
):

    if day <= 15:

        return {
            "model":
                0.65,

            "hydrology":
                0.25,

            "trend":
                0.10,
        }


    if day <= 30:

        return {
            "model":
                0.52,

            "hydrology":
                0.30,

            "trend":
                0.18,
        }


    if day <= 45:

        return {
            "model":
                0.38,

            "hydrology":
                0.34,

            "trend":
                0.28,
        }


    return {
        "model":
            0.28,

        "hydrology":
            0.34,

        "trend":
            0.38,
    }


# ============================================================
# INCERTIDUMBRE
# ============================================================

def incertidumbre_horizonte(
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

        factor = 1.10

    elif day <= 45:

        factor = 1.25

    else:

        factor = 1.40


    uncertainty = (
        rmse
        * np.sqrt(
            day
        )
        * factor
    )


    return float(
        min(
            uncertainty,
            MAX_UNCERTAINTY,
        )
    )


# ============================================================
# TENDENCIA LOCAL
# ============================================================

def tendencia_local_reciente(
    history,
):

    if (
        history is None
        or history.empty
        or "nivel"
        not in history.columns
    ):

        return 0.0


    values = (
        _numeric(
            history[
                "nivel"
            ]
        )
        .dropna()
        .tail(
            10
        )
    )


    if len(
        values
    ) < 3:

        return 0.0


    try:

        slope = float(
            np.polyfit(
                np.arange(
                    len(
                        values
                    ),
                    dtype=float,
                ),
                values.to_numpy(
                    dtype=float
                ),
                1,
            )[0]
        )


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
# FUTURE EXOG
# ============================================================

def preparar_future(
    df,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "datetime"
        not in df.columns
    ):

        return pd.DataFrame()


    x = df.copy()


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


    for col in x.columns:

        if col == "datetime":
            continue


        x[
            col
        ] = _numeric(
            x[
                col
            ]
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
# OBTENER VALOR FUTURO
# ============================================================

def _future_value(
    future_df,
    target_date,
    col,
):

    if (
        future_df is None
        or future_df.empty
        or col not in future_df.columns
    ):

        return np.nan


    rows = future_df[
        future_df[
            "datetime"
        ]
        == target_date
    ]


    if rows.empty:

        return np.nan


    return _safe_float(
        rows.iloc[
            0
        ][
            col
        ],
        np.nan,
    )


# ============================================================
# PROYECTAR CAUDAL FALTANTE
# ============================================================

def _project_flow_one_day(
    history,
    col,
):

    if (
        col not in history.columns
    ):

        return np.nan


    values = (
        _numeric(
            history[
                col
            ]
        )
        .dropna()
        .tail(
            14
        )
    )


    if values.empty:

        return np.nan


    current = float(
        values.iloc[-1]
    )


    if len(
        values
    ) >= 4:

        slope = float(
            np.polyfit(
                np.arange(
                    len(
                        values
                    ),
                    dtype=float,
                ),
                values.to_numpy(
                    dtype=float
                ),
                1,
            )[0]
        )

    else:

        slope = 0.0


    max_change = max(
        abs(
            current
        )
        * 0.025,
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
        0.0,
        current
        + slope
        * 0.75,
    )


# ============================================================
# PROYECTAR NIVEL UPSTREAM FALTANTE
# ============================================================

def _project_upstream_one_day(
    history,
    col,
):

    if col not in history.columns:

        return np.nan


    values = (
        _numeric(
            history[
                col
            ]
        )
        .dropna()
        .tail(
            7
        )
    )


    if values.empty:

        return np.nan


    current = float(
        values.iloc[-1]
    )


    if len(
        values
    ) >= 3:

        slope = float(
            np.polyfit(
                np.arange(
                    len(
                        values
                    ),
                    dtype=float,
                ),
                values.to_numpy(
                    dtype=float
                ),
                1,
            )[0]
        )

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
# CREAR FILA FUTURA
# ============================================================

def crear_fila_futura(
    history,
    target_date,
    current_level,
    exog_future=None,
    upstream_future=None,
    horizon_day=1,
):

    row = {
        "datetime":
            target_date,

        "nivel":
            current_level,
    }


    # ========================================================
    # LLUVIA
    # ========================================================

    for col in RAIN_COLUMNS.values():

        value = _future_value(
            exog_future,
            target_date,
            col,
        )


        if not np.isfinite(
            value
        ):

            value = 0.0


        # ----------------------------------------------------
        # Después del horizonte meteorológico corto, cero no
        # debe actuar como señal fuerte de sequía.
        #
        # Se mantiene neutro.
        # ----------------------------------------------------

        if horizon_day > 16:

            value = 0.0


        row[
            col
        ] = value


    # Compatibilidad
    row[
        "precip_mm"
    ] = _future_value(
        exog_future,
        target_date,
        "precip_mm",
    )


    if not np.isfinite(
        row[
            "precip_mm"
        ]
    ):

        row[
            "precip_mm"
        ] = 0.0


    # ========================================================
    # CAUDALES
    # ========================================================

    for col in FLOW_COLUMNS.values():

        value = _future_value(
            exog_future,
            target_date,
            col,
        )


        if not np.isfinite(
            value
        ):

            value = _project_flow_one_day(
                history,
                col,
            )


        row[
            col
        ] = value


    row[
        "caudal_m3s"
    ] = _future_value(
        exog_future,
        target_date,
        "caudal_m3s",
    )


    if not np.isfinite(
        row[
            "caudal_m3s"
        ]
    ):

        # Primer caudal válido.
        for col in FLOW_COLUMNS.values():

            value = row.get(
                col
            )

            if (
                value is not None
                and np.isfinite(
                    value
                )
            ):

                row[
                    "caudal_m3s"
                ] = value

                break


    # ========================================================
    # NIVELES AGUAS ARRIBA
    # ========================================================

    for col in LEVEL_COLUMNS.values():

        value = _future_value(
            upstream_future,
            target_date,
            col,
        )


        if not np.isfinite(
            value
        ):

            value = _project_upstream_one_day(
                history,
                col,
            )


        row[
            col
        ] = value


    return row


# ============================================================
# SEÑAL HIDROLÓGICA
# ============================================================

def calcular_senal_hidrologica(
    history,
    future_row=None,
):

    if (
        history is None
        or history.empty
    ):

        return {
            "combined":
                0.0,

            "level":
                0.0,

            "flow":
                0.0,

            "rain":
                0.0,
        }


    level_parts = []
    flow_parts = []
    rain_parts = []


    # ========================================================
    # NIVELES
    # ========================================================

    for station, col in LEVEL_COLUMNS.items():

        if col not in history.columns:
            continue


        values = (
            _numeric(
                history[
                    col
                ]
            )
            .dropna()
            .tail(
                8
            )
        )


        if len(
            values
        ) < 3:

            continue


        slope = (
            float(
                values.iloc[-1]
            )
            -
            float(
                values.iloc[0]
            )
        ) / max(
            len(
                values
            )
            - 1,
            1,
        )


        level_parts.append(
            slope
            *
            STATION_WEIGHTS.get(
                station,
                1.0,
            )
        )


    # ========================================================
    # CAUDALES
    # ========================================================

    for station, col in FLOW_COLUMNS.items():

        if col not in history.columns:
            continue


        values = (
            _numeric(
                history[
                    col
                ]
            )
            .dropna()
            .tail(
                8
            )
        )


        if len(
            values
        ) < 3:

            continue


        first = float(
            values.iloc[
                0
            ]
        )


        last = float(
            values.iloc[
                -1
            ]
        )


        if abs(
            first
        ) < 1e-9:

            continue


        relative = (
            last
            - first
        ) / abs(
            first
        )


        flow_parts.append(
            relative
            *
            STATION_WEIGHTS.get(
                station,
                1.0,
            )
        )


    # ========================================================
    # LLUVIA FUTURA
    # ========================================================

    if future_row is not None:

        for station, col in RAIN_COLUMNS.items():

            value = _safe_float(
                future_row.get(
                    col
                ),
                0.0,
            )


            if value is None:
                value = 0.0


            rain_parts.append(
                value
                *
                STATION_WEIGHTS.get(
                    station,
                    1.0,
                )
            )


    level_signal = (
        float(
            np.nanmean(
                level_parts
            )
        )
        if level_parts
        else 0.0
    )


    flow_signal = (
        float(
            np.nanmean(
                flow_parts
            )
        )
        if flow_parts
        else 0.0
    )


    rain_signal = (
        float(
            np.nanmean(
                rain_parts
            )
        )
        if rain_parts
        else 0.0
    )


    # --------------------------------------------------------
    # Convertimos a variación diaria aproximada.
    # --------------------------------------------------------

    combined = (

        0.70
        * level_signal

        +

        0.20
        * flow_signal

        +

        0.01
        * rain_signal
    )


    combined = float(
        np.clip(
            combined,
            -0.12,
            0.12,
        )
    )


    return {

        "combined":
            combined,

        "level":
            level_signal,

        "flow":
            flow_signal,

        "rain":
            rain_signal,
    }


# ============================================================
# LÍMITE DE CAMBIO DIARIO
# ============================================================

def limite_cambio_diario(
    horizon_day,
    hydro_signal=0.0,
):

    if horizon_day <= 15:

        max_abs = 0.20

    elif horizon_day <= 30:

        max_abs = 0.16

    elif horizon_day <= 45:

        max_abs = 0.13

    else:

        max_abs = 0.10


    max_rise = max_abs

    max_drop = -max_abs


    # --------------------------------------------------------
    # Si aguas arriba está creciendo claramente,
    # restringimos las bajadas abruptas.
    # --------------------------------------------------------

    if hydro_signal > 0.03:

        max_drop = max(
            max_drop,
            -0.08,
        )


    if hydro_signal > 0.06:

        max_drop = max(
            max_drop,
            -0.04,
        )


    return (
        max_drop,
        max_rise,
    )


# ============================================================
# PREDICCIÓN
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

    """
    Pronóstico recursivo hasta 60 días.

    Compatible con:
        predict(df, models, ...)

    y:
        predict(models, df, ...)
    """


    # ========================================================
    # COMPATIBILIDAD ARGUMENTOS INVERTIDOS
    # ========================================================

    if (
        isinstance(
            df,
            dict,
        )
        and
        isinstance(
            models,
            pd.DataFrame,
        )
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


    days = max(
        1,
        min(
            int(
                days
            ),
            MAX_FORECAST_DAYS,
        ),
    )


    feature_names = models.get(
        "feature_names",
        []
    )


    fill_values = models.get(
        "feature_fill_values",
        {}
    )


    rmse = models.get(
        "rmse",
        MIN_RMSE,
    )


    propagation_lags = models.get(
        "propagation_lags",
        DEFAULT_PROPAGATION_LAGS,
    )


    if hydrology is None:

        hydrology = models.get(
            "hydrology",
            {}
        )


    # ========================================================
    # DATASET BASE
    # ========================================================

    base_dataset = models.get(
        "dataset"
    )


    if (
        base_dataset is None
        or not isinstance(
            base_dataset,
            pd.DataFrame,
        )
        or base_dataset.empty
    ):

        base_dataset = preparar_dataset(
            df
        )


    history = base_dataset.copy()


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


    exog_future = preparar_future(
        exog_future
    )


    upstream_future = preparar_future(
        upstream_future
    )


    # ========================================================
    # ESTADO INICIAL
    # ========================================================

    current_level = _safe_float(
        history[
            "nivel"
        ].dropna().iloc[
            -1
        ]
    )


    if current_level is None:

        raise ValueError(
            "No existe un nivel inicial válido."
        )


    last_date = pd.to_datetime(
        history[
            "datetime"
        ].max()
    ).normalize()


    local_trend = tendencia_local_reciente(
        history
    )


    previous_delta = local_trend


    forecast_rows = []


    # ========================================================
    # CICLO 60 DÍAS
    # ========================================================

    for horizon_day in range(
        1,
        days + 1,
    ):

        target_date = (
            last_date
            + pd.Timedelta(
                days=horizon_day
            )
        )


        # ====================================================
        # FILA FUTURA
        # ====================================================

        future_row = crear_fila_futura(

            history,

            target_date,

            current_level,

            exog_future=
                exog_future,

            upstream_future=
                upstream_future,

            horizon_day=
                horizon_day,
        )


        # ====================================================
        # AGREGAR TEMPORALMENTE
        # ====================================================

        temp_history = pd.concat(

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


        # ====================================================
        # FEATURES DE LA FILA FUTURA
        # ====================================================

        temp_features = crear_features(

            temp_history,

            propagation_lags=
                propagation_lags,
        )


        row_features = (
            temp_features.iloc[
                -1
            ]
        )


        X_row = {}


        for feature in feature_names:

            value = _safe_float(
                row_features.get(
                    feature
                ),
                None,
            )


            if value is None:

                value = fill_values.get(
                    feature,
                    0.0,
                )


            X_row[
                feature
            ] = value


        X_row = pd.DataFrame(
            [
                X_row
            ],
            columns=
                feature_names,
        )


        # ====================================================
        # MODELO ML
        # ====================================================

        ml_delta = float(
            model.predict(
                X_row
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

        hydro = calcular_senal_hidrologica(

            history,

            future_row=
                future_row,
        )


        hydro_delta = hydro[
            "combined"
        ]


        # ====================================================
        # TENDENCIA LOCAL AMORTIGUADA
        # ====================================================

        trend_damping = np.exp(
            -horizon_day
            / 25.0
        )


        trend_delta = (
            local_trend
            * trend_damping
        )


        # ====================================================
        # PESOS
        # ====================================================

        weights = pesos_horizonte(
            horizon_day
        )


        combined_delta = (

            weights[
                "model"
            ]
            * ml_delta

            +

            weights[
                "hydrology"
            ]
            * hydro_delta

            +

            weights[
                "trend"
            ]
            * trend_delta
        )


        # ====================================================
        # SUAVIZADO
        # ====================================================

        combined_delta = (

            0.68
            * previous_delta

            +

            0.32
            * combined_delta
        )


        # ====================================================
        # LÍMITES
        # ====================================================

        min_change, max_change = (
            limite_cambio_diario(

                horizon_day,

                hydro_signal=
                    hydro_delta,
            )
        )


        combined_delta = float(
            np.clip(

                combined_delta,

                min_change,

                max_change,
            )
        )


        # ====================================================
        # NIVEL NUEVO
        # ====================================================

        prediction = float(
            np.clip(

                current_level
                + combined_delta,

                LEVEL_MIN,

                LEVEL_MAX,
            )
        )


        # ====================================================
        # INCERTIDUMBRE
        # ====================================================

        uncertainty = incertidumbre_horizonte(
            rmse,
            horizon_day,
        )


        lower = prediction - uncertainty

        upper = prediction + uncertainty


        # ====================================================
        # SALIDA
        # ====================================================

        output_row = {

            "datetime":
                target_date,

            "prediction":
                prediction,

            "lower":
                lower,

            "upper":
                upper,

            "horizon_day":
                horizon_day,

            "horizon_group":
                grupo_horizonte(
                    horizon_day
                ),

            "forecast_type":
                tipo_horizonte(
                    horizon_day
                ),

            "nivel_base":
                current_level,

            "variacion_dia":
                combined_delta,

            "delta_prediction":
                combined_delta,

            "variacion_modelo":
                ml_delta,

            "variacion_hidrologica":
                hydro_delta,

            "variacion_tendencia":
                trend_delta,

            "upstream_level_signal":
                hydro[
                    "level"
                ],

            "upstream_flow_signal":
                hydro[
                    "flow"
                ],

            "rain_signal":
                hydro[
                    "rain"
                ],

            "model_weight":
                weights[
                    "model"
                ],

            "hydrology_weight":
                weights[
                    "hydrology"
                ],

            "trend_weight":
                weights[
                    "trend"
                ],

            "precip_mm":
                future_row.get(
                    "precip_mm",
                    np.nan,
                ),

            "caudal_m3s":
                future_row.get(
                    "caudal_m3s",
                    np.nan,
                ),
        }


        # ----------------------------------------------------
        # Guardar lluvia por estación.
        # ----------------------------------------------------

        for col in RAIN_COLUMNS.values():

            output_row[
                col
            ] = future_row.get(
                col,
                np.nan,
            )


        # ----------------------------------------------------
        # Guardar caudal por estación.
        # ----------------------------------------------------

        for col in FLOW_COLUMNS.values():

            output_row[
                col
            ] = future_row.get(
                col,
                np.nan,
            )


        # ----------------------------------------------------
        # Guardar nivel upstream.
        # ----------------------------------------------------

        for col in LEVEL_COLUMNS.values():

            output_row[
                col
            ] = future_row.get(
                col,
                np.nan,
            )


        forecast_rows.append(
            output_row
        )


        # ====================================================
        # ACTUALIZAR HISTORIA RECURSIVA
        # ====================================================

        append_row = future_row.copy()


        append_row[
            "nivel"
        ] = prediction


        history = pd.concat(

            [
                history,

                pd.DataFrame(
                    [
                        append_row
                    ]
                ),
            ],

            ignore_index=True,

            sort=False,
        )


        current_level = prediction

        previous_delta = (
            combined_delta
        )


    forecast = pd.DataFrame(
        forecast_rows
    )


    forecast[
        "datetime"
    ] = _datetime_naive(
        forecast[
            "datetime"
        ]
    )


    return forecast


# ============================================================
# RESUMEN DE NIVELES
# ============================================================

def resumen_niveles_estaciones(
    upstream_history,
):

    rows = []


    upstream = preparar_upstream(
        upstream_history
    )


    if upstream.empty:

        return pd.DataFrame()


    for station, col in LEVEL_COLUMNS.items():

        if col not in upstream.columns:
            continue


        data = (
            upstream[
                [
                    "datetime",
                    col,
                ]
            ]
            .dropna()
            .sort_values(
                "datetime"
            )
        )


        if data.empty:
            continue


        current = float(
            data.iloc[
                -1
            ][
                col
            ]
        )


        delta_1 = None
        delta_3 = None
        delta_7 = None


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


        if (
            delta_3 is not None
            and delta_3 > 0.03
        ):

            state = "↑ Creciendo"

        elif (
            delta_3 is not None
            and delta_3 < -0.03
        ):

            state = "↓ Bajando"

        else:

            state = "→ Estable"


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
#
# Nota:
# esta rutina utiliza el modelo entrenado actual.
# No debe considerarse todavía un rolling backtest
# completamente libre de leakage.
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


    local = preparar_nivel_local(
        df
    )


    rows = []


    for horizon in horizons:

        if len(
            local
        ) <= horizon + 30:

            continue


        cutoff_index = (
            len(
                local
            )
            - horizon
            - 1
        )


        cutoff = local.iloc[
            :cutoff_index + 1
        ]


        actual_row = local.iloc[
            cutoff_index
            + horizon
        ]


        try:

            pred = predict(

                cutoff,

                models,

                days=
                    horizon,
            )


            if pred.empty:
                continue


            predicted = float(
                pred.iloc[
                    -1
                ][
                    "prediction"
                ]
            )


            actual = float(
                actual_row[
                    "nivel"
                ]
            )


            error = (
                predicted
                - actual
            )


            rows.append(
                {
                    "horizon":
                        horizon,

                    "prediction":
                        predicted,

                    "actual":
                        actual,

                    "error":
                        error,

                    "abs_error":
                        abs(
                            error
                        ),
                }
            )


        except Exception:

            continue


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
        or not isinstance(
            backtest,
            pd.DataFrame,
        )
        or backtest.empty
    ):

        return pd.DataFrame()


    rows = []


    for horizon, group in backtest.groupby(
        "horizon"
    ):

        errors = _numeric(
            group[
                "error"
            ]
        ).dropna()


        if errors.empty:
            continue


        rows.append(
            {
                "horizon":
                    int(
                        horizon
                    ),

                "MAE":
                    float(
                        np.mean(
                            np.abs(
                                errors
                            )
                        )
                    ),

                "RMSE":
                    float(
                        np.sqrt(
                            np.mean(
                                errors
                                ** 2
                            )
                        )
                    ),

                "n":
                    int(
                        len(
                            errors
                        )
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# PROBABILIDAD AUXILIAR
# ============================================================

def prob(
    value,
    center=0.0,
    scale=1.0,
):

    try:

        value = float(
            value
        )

        center = float(
            center
        )

        scale = max(
            abs(
                float(
                    scale
                )
            ),
            1e-6,
        )


        z = (
            value
            - center
        ) / scale


        return float(
            1.0
            /
            (
                1.0
                +
                np.exp(
                    -z
                )
            )
        )


    except Exception:

        return np.nan


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


        feature_names = seleccionar_features(
            features
        )


        (
            upstream_count,
            flow_count,
            rain_count,
        ) = _count_feature_groups(
            feature_names
        )


        result.update(
            {
                "status":
                    "ok",

                "dataset_rows":
                    int(
                        len(
                            dataset
                        )
                    ),

                "feature_count":
                    int(
                        len(
                            feature_names
                        )
                    ),

                "upstream_feature_count":
                    upstream_count,

                "flow_feature_count":
                    flow_count,

                "rain_feature_count":
                    rain_count,

                "uses_upstream":
                    upstream_count > 0,

                "uses_caudal":
                    flow_count > 0,

                "uses_rain":
                    rain_count > 0,

                "propagation_lags":
                    propagation_lags,

                "upstream_columns_present":
                    [
                        col
                        for col
                        in LEVEL_COLUMNS.values()
                        if (
                            col
                            in dataset.columns
                            and _numeric(
                                dataset[
                                    col
                                ]
                            )
                            .notna()
                            .any()
                        )
                    ],

                "flow_columns_present":
                    [
                        col
                        for col
                        in FLOW_COLUMNS.values()
                        if (
                            col
                            in dataset.columns
                            and _numeric(
                                dataset[
                                    col
                                ]
                            )
                            .notna()
                            .any()
                        )
                    ],

                "rain_columns_present":
                    [
                        col
                        for col
                        in RAIN_COLUMNS.values()
                        if (
                            col
                            in dataset.columns
                            and _numeric(
                                dataset[
                                    col
                                ]
                            )
                            .notna()
                            .any()
                        )
                    ],
            }
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
