# ============================================================
# PARANÁ · SAN NICOLÁS
# src/model.py
# V11.9.7 COMPLETO
#
# OBJETIVOS
# ------------------------------------------------------------
# - Corregir compatibilidad app.py <-> model.py
# - Pronóstico recursivo hasta 60 días
# - Mantener horizonte 15 / 30 / 45 / 60
# - Usar nivel San Nicolás
# - Usar lluvia
# - Usar caudal
# - Usar estaciones aguas arriba
# - Aprender variación diaria de nivel
# - Evitar saltos diarios irreales
# - Compatible con llamadas:
#
#       predict(df, models, ...)
#
#   y también accidentalmente:
#
#       predict(models, df, ...)
#
# ============================================================


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ============================================================
# VERSIÓN
# ============================================================

VERSION = "V11.9.7"


# ============================================================
# CONFIGURACIÓN
# ============================================================

MIN_OBSERVATIONS = 60

MAX_FORECAST_DAYS = 60

RANDOM_STATE = 42


# ------------------------------------------------------------
# Límites de seguridad.
#
# NO son la escala de los gráficos.
# Los gráficos deben usar escala automática desde app.py.
# ------------------------------------------------------------

LEVEL_MIN = -2.0
LEVEL_MAX = 12.0


LOCAL_LAGS = [
    1,
    2,
    3,
    5,
    7,
    10,
    14,
]


Q_LAGS = [
    1,
    3,
    5,
    7,
    10,
    14,
]


UPSTREAM_LAGS = [
    1,
    2,
    3,
    5,
    7,
    10,
    14,
]


# ============================================================
# UTILIDADES
# ============================================================

def _normalizar_datetime(
    serie,
):

    return (
        pd.to_datetime(
            serie,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
        .dt
        .normalize()
    )


def _to_numeric(
    serie,
):

    return pd.to_numeric(
        serie,
        errors="coerce",
    )


def _safe_last(
    serie,
    default=np.nan,
):

    try:

        s = pd.to_numeric(
            serie,
            errors="coerce",
        ).dropna()

        if s.empty:
            return default

        return float(
            s.iloc[-1]
        )

    except Exception:

        return default


def _safe_slope(
    serie,
    window=7,
):

    try:

        s = (
            pd.to_numeric(
                serie,
                errors="coerce",
            )
            .dropna()
            .tail(
                window
            )
        )

        if len(s) < 3:
            return 0.0

        x = np.arange(
            len(s),
            dtype=float,
        )

        return float(
            np.polyfit(
                x,
                s.to_numpy(
                    dtype=float
                ),
                1,
            )[0]
        )

    except Exception:

        return 0.0


def _upstream_cols(
    df,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
    ):

        return []

    return [
        c
        for c in df.columns
        if (
            c.startswith(
                "nivel_"
            )
            and c != "nivel"
            and "_lag" not in c
            and "_diff" not in c
            and "_trend" not in c
            and "_mean" not in c
            and "_actual" not in c
            and "_next" not in c
        )
    ]


# ============================================================
# PREPARAR NIVEL SAN NICOLÁS
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
            "No hay datos de San Nicolás."
        )

    x = df.copy()

    if "datetime" not in x.columns:

        raise ValueError(
            "Falta la columna datetime."
        )

    if "nivel" in x.columns:

        x[
            "nivel"
        ] = _to_numeric(
            x[
                "nivel"
            ]
        )

    elif "value" in x.columns:

        x[
            "nivel"
        ] = _to_numeric(
            x[
                "value"
            ]
        )

    else:

        raise ValueError(
            "Falta la columna nivel/value."
        )

    x[
        "datetime"
    ] = _normalizar_datetime(
        x[
            "datetime"
        ]
    )

    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    if x.empty:

        raise ValueError(
            "No quedan niveles válidos de San Nicolás."
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

    # --------------------------------------------------------
    # Sólo quitar valores claramente imposibles.
    # No limitar a 0-7.
    # --------------------------------------------------------

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
    ].copy()

    return x


# ============================================================
# PREPARAR EXÓGENAS
# ============================================================

def _preparar_exog_history(
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

    exog = exog_history.copy()

    exog[
        "datetime"
    ] = _normalizar_datetime(
        exog[
            "datetime"
        ]
    )

    columns = [
        "datetime"
    ]

    if (
        "precip_mm"
        in exog.columns
    ):

        exog[
            "precip_mm"
        ] = (
            _to_numeric(
                exog[
                    "precip_mm"
                ]
            )
            .fillna(
                0.0
            )
            .clip(
                lower=0.0
            )
        )

        columns.append(
            "precip_mm"
        )

    if (
        "caudal_m3s"
        in exog.columns
    ):

        exog[
            "caudal_m3s"
        ] = _to_numeric(
            exog[
                "caudal_m3s"
            ]
        )

        columns.append(
            "caudal_m3s"
        )

    exog = exog[
        columns
    ].copy()

    agg = {}

    if (
        "precip_mm"
        in exog.columns
    ):

        agg[
            "precip_mm"
        ] = "sum"

    if (
        "caudal_m3s"
        in exog.columns
    ):

        agg[
            "caudal_m3s"
        ] = "mean"

    if not agg:

        return pd.DataFrame()

    return (
        exog
        .dropna(
            subset=[
                "datetime"
            ]
        )
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
# PREPARAR AGUAS ARRIBA
# ============================================================

def _preparar_upstream_history(
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
    ):

        return pd.DataFrame()

    up = upstream_history.copy()

    up[
        "datetime"
    ] = _normalizar_datetime(
        up[
            "datetime"
        ]
    )

    station_cols = [
        c
        for c in up.columns
        if c.startswith(
            "nivel_"
        )
    ]

    if not station_cols:

        return pd.DataFrame()

    for col in station_cols:

        up[
            col
        ] = _to_numeric(
            up[
                col
            ]
        )

    agg = {
        col:
            "mean"
        for col
        in station_cols
    }

    return (
        up[
            [
                "datetime"
            ]
            + station_cols
        ]
        .dropna(
            subset=[
                "datetime"
            ]
        )
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
# PREPARAR DATASET
# ============================================================

def preparar_dataset(
    df,
    exog_history=None,
    upstream_history=None,
):

    x = preparar_nivel_local(
        df
    )

    # ========================================================
    # EXÓGENAS
    # ========================================================

    exog = _preparar_exog_history(
        exog_history
    )

    if not exog.empty:

        x = x.merge(
            exog,
            on="datetime",
            how="left",
        )

    # ========================================================
    # AGUAS ARRIBA
    # ========================================================

    upstream = (
        _preparar_upstream_history(
            upstream_history
        )
    )

    if not upstream.empty:

        x = x.merge(
            upstream,
            on="datetime",
            how="left",
        )

    # ========================================================
    # VARIABLES BASE
    # ========================================================

    if (
        "precip_mm"
        not in x.columns
    ):

        x[
            "precip_mm"
        ] = 0.0

    if (
        "caudal_m3s"
        not in x.columns
    ):

        x[
            "caudal_m3s"
        ] = np.nan


    x[
        "precip_mm"
    ] = (
        _to_numeric(
            x[
                "precip_mm"
            ]
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
    )


    x[
        "caudal_m3s"
    ] = _to_numeric(
        x[
            "caudal_m3s"
        ]
    )


    # ========================================================
    # INTERPOLAR CAUDAL
    # ========================================================

    if (
        x[
            "caudal_m3s"
        ]
        .notna()
        .sum()
        >= 7
    ):

        x[
            "caudal_m3s"
        ] = (
            x[
                "caudal_m3s"
            ]
            .interpolate(
                limit=7,
                limit_direction=
                    "both",
            )
        )


    # ========================================================
    # INTERPOLAR ESTACIONES
    # ========================================================

    for col in _upstream_cols(
        x
    ):

        x[
            col
        ] = _to_numeric(
            x[
                col
            ]
        )

        if (
            x[
                col
            ]
            .notna()
            .sum()
            >= 5
        ):

            x[
                col
            ] = (
                x[
                    col
                ]
                .interpolate(
                    limit=5,
                    limit_direction=
                        "both",
                )
            )


    if len(
        x
    ) < MIN_OBSERVATIONS:

        raise ValueError(
            f"Se requieren al menos "
            f"{MIN_OBSERVATIONS} días de datos. "
            f"Disponibles: {len(x)}."
        )

    return (
        x
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# FEATURES
# ============================================================

def crear_features(
    df,
):

    x = df.copy()

    # ========================================================
    # SAN NICOLÁS
    # ========================================================

    x[
        "sn_actual"
    ] = x[
        "nivel"
    ]


    for lag in LOCAL_LAGS:

        x[
            f"sn_lag_{lag}"
        ] = (
            x[
                "nivel"
            ]
            .shift(
                lag
            )
        )


    x[
        "sn_diff1"
    ] = (
        x[
            "nivel"
        ]
        - x[
            "nivel"
        ].shift(
            1
        )
    )


    x[
        "sn_diff3"
    ] = (
        x[
            "nivel"
        ]
        - x[
            "nivel"
        ].shift(
            3
        )
    )


    x[
        "sn_diff7"
    ] = (
        x[
            "nivel"
        ]
        - x[
            "nivel"
        ].shift(
            7
        )
    )


    x[
        "sn_trend3"
    ] = (
        x[
            "sn_diff3"
        ]
        / 3.0
    )


    x[
        "sn_trend7"
    ] = (
        x[
            "sn_diff7"
        ]
        / 7.0
    )


    x[
        "sn_media3"
    ] = (
        x[
            "nivel"
        ]
        .rolling(
            3
        )
        .mean()
    )


    x[
        "sn_media7"
    ] = (
        x[
            "nivel"
        ]
        .rolling(
            7
        )
        .mean()
    )


    x[
        "sn_media14"
    ] = (
        x[
            "nivel"
        ]
        .rolling(
            14
        )
        .mean()
    )


    # ========================================================
    # LLUVIA
    # ========================================================

    x[
        "rain_actual"
    ] = x[
        "precip_mm"
    ]


    x[
        "rain_next"
    ] = (
        x[
            "precip_mm"
        ]
        .shift(
            -1
        )
    )


    x[
        "rain_3d"
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
        "rain_7d"
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
        "rain_14d"
    ] = (
        x[
            "precip_mm"
        ]
        .rolling(
            14
        )
        .sum()
    )


    # ========================================================
    # CAUDAL
    # ========================================================

    if (
        "caudal_m3s"
        in x.columns
        and x[
            "caudal_m3s"
        ]
        .notna()
        .sum()
        >= 10
    ):

        x[
            "q_actual"
        ] = x[
            "caudal_m3s"
        ]


        x[
            "q_next"
        ] = (
            x[
                "caudal_m3s"
            ]
            .shift(
                -1
            )
        )


        for lag in Q_LAGS:

            x[
                f"q_lag_{lag}"
            ] = (
                x[
                    "caudal_m3s"
                ]
                .shift(
                    lag
                )
            )


        x[
            "q_diff1"
        ] = (
            x[
                "caudal_m3s"
            ]
            - x[
                "caudal_m3s"
            ].shift(
                1
            )
        )


        x[
            "q_diff3"
        ] = (
            x[
                "caudal_m3s"
            ]
            - x[
                "caudal_m3s"
            ].shift(
                3
            )
        )


        x[
            "q_trend3"
        ] = (
            x[
                "q_diff3"
            ]
            / 3.0
        )


        x[
            "q_trend7"
        ] = (
            x[
                "caudal_m3s"
            ]
            - x[
                "caudal_m3s"
            ].shift(
                7
            )
        ) / 7.0


        x[
            "q_media7"
        ] = (
            x[
                "caudal_m3s"
            ]
            .rolling(
                7
            )
            .mean()
        )


    # ========================================================
    # AGUAS ARRIBA
    # ========================================================

    for col in _upstream_cols(
        x
    ):

        x[
            f"{col}_actual"
        ] = x[
            col
        ]


        x[
            f"{col}_next"
        ] = (
            x[
                col
            ]
            .shift(
                -1
            )
        )


        x[
            f"{col}_diff1"
        ] = (
            x[
                col
            ]
            - x[
                col
            ].shift(
                1
            )
        )


        x[
            f"{col}_trend3"
        ] = (
            x[
                col
            ]
            - x[
                col
            ].shift(
                3
            )
        ) / 3.0


        x[
            f"{col}_trend7"
        ] = (
            x[
                col
            ]
            - x[
                col
            ].shift(
                7
            )
        ) / 7.0


        x[
            f"{col}_mean7"
        ] = (
            x[
                col
            ]
            .rolling(
                7
            )
            .mean()
        )


        for lag in UPSTREAM_LAGS:

            x[
                f"{col}_lag{lag}"
            ] = (
                x[
                    col
                ]
                .shift(
                    lag
                )
            )


    # ========================================================
    # TARGET
    # ========================================================

    x[
        "target_nivel"
    ] = (
        x[
            "nivel"
        ]
        .shift(
            -1
        )
    )


    x[
        "target_delta"
    ] = (
        x[
            "target_nivel"
        ]
        - x[
            "nivel"
        ]
    )


    return x


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train(
    df,
    exog_history=None,
    upstream_history=None,
):

    dataset = preparar_dataset(
        df,
        exog_history=
            exog_history,
        upstream_history=
            upstream_history,
    )


    features = crear_features(
        dataset
    )


    excluded = {
        "datetime",
        "nivel",
        "precip_mm",
        "caudal_m3s",
        "target_nivel",
        "target_delta",
    }


    feature_cols = [
        c
        for c
        in features.columns
        if c
        not in excluded
    ]


    # ========================================================
    # FILTRAR FEATURES CON POCA INFORMACIÓN
    # ========================================================

    usable = []

    for col in feature_cols:

        count = (
            features[
                col
            ]
            .notna()
            .sum()
        )

        if count >= 20:

            usable.append(
                col
            )


    feature_cols = usable


    if not feature_cols:

        raise ValueError(
            "No fue posible construir variables suficientes para entrenar."
        )


    # ========================================================
    # EVITAR QUE UNA ESTACIÓN CON MUCHOS NaN ELIMINE TODO
    #
    # En vez de dropna sobre TODAS las features,
    # rellenamos las variables faltantes con mediana histórica.
    # ========================================================

    work = features[
        features[
            "target_delta"
        ].notna()
    ].copy()


    for col in feature_cols:

        values = pd.to_numeric(
            work[
                col
            ],
            errors="coerce",
        )

        median = values.median()

        if pd.isna(
            median
        ):

            median = 0.0

        work[
            col
        ] = values.fillna(
            float(
                median
            )
        )


    work[
        "target_delta"
    ] = pd.to_numeric(
        work[
            "target_delta"
        ],
        errors="coerce",
    )


    work = work.dropna(
        subset=[
            "target_delta"
        ]
    )


    if len(
        work
    ) < 30:

        raise ValueError(
            "No quedan suficientes registros después de "
            "construir las variables: "
            f"{len(work)}."
        )


    # ========================================================
    # VALIDACIÓN CRONOLÓGICA
    # ========================================================

    split = int(
        len(
            work
        )
        * 0.80
    )


    split = max(
        25,
        split,
    )


    split = min(
        split,
        len(
            work
        )
        - 1,
    )


    X_train = work[
        feature_cols
    ].iloc[
        :split
    ]


    X_test = work[
        feature_cols
    ].iloc[
        split:
    ]


    y_train = work[
        "target_delta"
    ].iloc[
        :split
    ]


    y_test_delta = work[
        "target_delta"
    ].iloc[
        split:
    ]


    nivel_base_test = work[
        "nivel"
    ].iloc[
        split:
    ]


    # ========================================================
    # MODELO VALIDACIÓN
    # ========================================================

    validation_model = (
        RandomForestRegressor(
            n_estimators=600,
            max_depth=12,
            min_samples_leaf=2,
            max_features=
                "sqrt",
            random_state=
                RANDOM_STATE,
            n_jobs=-1,
        )
    )


    validation_model.fit(
        X_train,
        y_train,
    )


    pred_delta = (
        validation_model.predict(
            X_test
        )
    )


    pred_nivel = (
        nivel_base_test.to_numpy(
            dtype=float
        )
        + pred_delta
    )


    y_test_nivel = (
        nivel_base_test.to_numpy(
            dtype=float
        )
        + y_test_delta.to_numpy(
            dtype=float
        )
    )


    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test_nivel,
                pred_nivel,
            )
        )
    )


    rmse = max(
        rmse,
        0.03,
    )


    # ========================================================
    # MODELO FINAL
    # ========================================================

    final_model = (
        RandomForestRegressor(
            n_estimators=900,
            max_depth=12,
            min_samples_leaf=2,
            max_features=
                "sqrt",
            random_state=
                RANDOM_STATE,
            n_jobs=-1,
        )
    )


    final_model.fit(
        work[
            feature_cols
        ],
        work[
            "target_delta"
        ],
    )


    # ========================================================
    # IMPORTANCIA
    # ========================================================

    importance = pd.DataFrame(
        {
            "feature":
                feature_cols,

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


    feature_fill_values = {}

    for col in feature_cols:

        median = pd.to_numeric(
            work[
                col
            ],
            errors="coerce",
        ).median()

        if pd.isna(
            median
        ):
            median = 0.0

        feature_fill_values[
            col
        ] = float(
            median
        )


    models = {

        "version":
            VERSION,

        "model":
            final_model,

        "feature_cols":
            feature_cols,

        "feature_fill_values":
            feature_fill_values,

        "rmse":
            rmse,

        "dataset":
            dataset,

        "observations":
            len(
                dataset
            ),

        "training_rows":
            len(
                work
            ),

        "importance":
            importance,

        "target":
            "daily_level_change",

        "max_forecast_days":
            MAX_FORECAST_DAYS,

        "uses_rain":
            any(
                c.startswith(
                    "rain_"
                )
                for c
                in feature_cols
            ),

        "uses_caudal":
            any(
                c.startswith(
                    "q_"
                )
                for c
                in feature_cols
            ),

        "uses_upstream":
            any(
                c.startswith(
                    "nivel_"
                )
                for c
                in feature_cols
            ),
    }


    metrics = {

        "RMSE":
            rmse,

        "observations":
            len(
                dataset
            ),

        "training_rows":
            len(
                work
            ),

        "test_rows":
            len(
                X_test
            ),

        "features":
            len(
                feature_cols
            ),

        "max_forecast_days":
            MAX_FORECAST_DAYS,
    }


    return (
        models,
        metrics,
    )


# ============================================================
# COMPATIBILIDAD
# RESUMEN DE NIVELES DE ESTACIONES
# ============================================================

def resumen_niveles_estaciones(
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


    rows = []


    labels = {

        "nivel_corrientes":
            "Corrientes",

        "nivel_goya":
            "Goya",

        "nivel_la_paz":
            "La Paz",

        "nivel_parana":
            "Paraná",

        "nivel_diamante":
            "Diamante",

        "nivel_rosario":
            "Rosario",

        "nivel_villa_constitucion":
            "Villa Constitución",
    }


    for col in _upstream_cols(
        upstream_history
    ):

        valid = pd.to_numeric(
            upstream_history[
                col
            ],
            errors="coerce",
        ).dropna()


        if valid.empty:

            continue


        current = float(
            valid.iloc[-1]
        )


        previous = (
            float(
                valid.iloc[-2]
            )
            if len(
                valid
            ) >= 2
            else np.nan
        )


        change = (
            current
            - previous
            if np.isfinite(
                previous
            )
            else np.nan
        )


        rows.append(
            {
                "Estación":
                    labels.get(
                        col,
                        col,
                    ),

                "Nivel":
                    current,

                "Anterior":
                    previous,

                "Variación":
                    change,
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# PROYECTAR CAUDAL UN DÍA
# ============================================================

def _proyectar_caudal_un_dia(
    history,
):

    if (
        "caudal_m3s"
        not in history.columns
    ):

        return np.nan


    valid = pd.to_numeric(
        history[
            "caudal_m3s"
        ],
        errors="coerce",
    ).dropna()


    if valid.empty:

        return np.nan


    actual = float(
        valid.iloc[-1]
    )


    slope = _safe_slope(
        valid,
        window=7,
    )


    limit = max(
        abs(
            actual
        )
        * 0.025,
        50.0,
    )


    slope = float(
        np.clip(
            slope,
            -limit,
            limit,
        )
    )


    return max(
        0.0,
        actual
        + slope,
    )


# ============================================================
# PROYECTAR NIVEL UPSTREAM UN DÍA
# ============================================================

def _proyectar_upstream_un_dia(
    history,
    col,
):

    valid = pd.to_numeric(
        history[
            col
        ],
        errors="coerce",
    ).dropna()


    if valid.empty:

        return np.nan


    actual = float(
        valid.iloc[-1]
    )


    pendiente = _safe_slope(
        valid,
        window=7,
    )


    limite = max(
        abs(
            actual
        )
        * 0.025,
        0.03,
    )


    pendiente = float(
        np.clip(
            pendiente,
            -limite,
            limite,
        )
    )


    return float(
        actual
        + pendiente
    )


# ============================================================
# FUTURO EXÓGENO
# ============================================================

def _preparar_future(
    exog_future,
):

    if (
        exog_future is None
        or not isinstance(
            exog_future,
            pd.DataFrame,
        )
        or exog_future.empty
        or "datetime"
        not in exog_future.columns
    ):

        return pd.DataFrame()


    future = exog_future.copy()


    future[
        "datetime"
    ] = _normalizar_datetime(
        future[
            "datetime"
        ]
    )


    if (
        "precip_mm"
        in future.columns
    ):

        future[
            "precip_mm"
        ] = (
            _to_numeric(
                future[
                    "precip_mm"
                ]
            )
            .fillna(
                0.0
            )
            .clip(
                lower=0.0
            )
        )


    if (
        "caudal_m3s"
        in future.columns
    ):

        future[
            "caudal_m3s"
        ] = _to_numeric(
            future[
                "caudal_m3s"
            ]
        )


    return (
        future
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
# FUTURO UPSTREAM
# ============================================================

def _preparar_upstream_future(
    upstream_future,
):

    if (
        upstream_future is None
        or not isinstance(
            upstream_future,
            pd.DataFrame,
        )
        or upstream_future.empty
        or "datetime"
        not in upstream_future.columns
    ):

        return pd.DataFrame()


    future = upstream_future.copy()


    future[
        "datetime"
    ] = _normalizar_datetime(
        future[
            "datetime"
        ]
    )


    for col in _upstream_cols(
        future
    ):

        future[
            col
        ] = _to_numeric(
            future[
                col
            ]
        )


    return (
        future
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
# CONSTRUIR FEATURES PARA PRONÓSTICO
# ============================================================

def _crear_features_pronostico(
    history,
    feature_cols,
    target_date,
    precip_next,
    caudal_next,
    upstream_next,
    fill_values=None,
):

    temp = history.copy()


    row = {

        "datetime":
            target_date,

        "nivel":
            float(
                temp[
                    "nivel"
                ].iloc[-1]
            ),

        "precip_mm":
            precip_next,

        "caudal_m3s":
            caudal_next,
    }


    for col in _upstream_cols(
        temp
    ):

        row[
            col
        ] = upstream_next.get(
            col,
            _safe_last(
                temp[
                    col
                ]
            ),
        )


    trial = pd.concat(
        [
            temp,
            pd.DataFrame(
                [
                    row
                ]
            ),
        ],
        ignore_index=True,
    )


    feat = crear_features(
        trial
    )


    # --------------------------------------------------------
    # La penúltima fila representa el estado actual y
    # contiene rain_next / q_next / nivel_estacion_next
    # correspondientes al target_date.
    # --------------------------------------------------------

    source = feat.iloc[
        -2
    ]


    values = {}


    fill_values = (
        fill_values
        if isinstance(
            fill_values,
            dict,
        )
        else {}
    )


    for col in feature_cols:

        value = source.get(
            col,
            np.nan,
        )


        value = pd.to_numeric(
            value,
            errors="coerce",
        )


        if pd.isna(
            value
        ):

            value = fill_values.get(
                col,
                0.0,
            )


        values[
            col
        ] = float(
            value
        )


    return pd.DataFrame(
        [
            values
        ],
        columns=
            feature_cols,
    )


# ============================================================
# PRONÓSTICO RECURSIVO HASTA 60 DÍAS
# ============================================================

def predict(
    df,
    models=None,
    days=15,
    exog_future=None,
    upstream_future=None,
):

    # ========================================================
    # COMPATIBILIDAD CON APP ANTIGUA
    #
    # Si llega:
    #
    # predict(models, df, ...)
    #
    # intercambiar automáticamente.
    # ========================================================

    if (
        isinstance(
            df,
            dict,
        )
        and isinstance(
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
            "El modelo no está entrenado o models no es válido."
        )


    required = [
        "dataset",
        "model",
        "feature_cols",
        "rmse",
    ]


    for key in required:

        if key not in models:

            raise ValueError(
                f"Falta models['{key}']. "
                "Debe ejecutarse train() antes de predict()."
            )


    try:

        days = int(
            days
        )

    except Exception:

        days = 15


    days = max(
        1,
        min(
            days,
            MAX_FORECAST_DAYS,
        ),
    )


    history = (
        models[
            "dataset"
        ]
        .copy()
    )


    history[
        "datetime"
    ] = _normalizar_datetime(
        history[
            "datetime"
        ]
    )


    model = models[
        "model"
    ]


    feature_cols = models[
        "feature_cols"
    ]


    fill_values = models.get(
        "feature_fill_values",
        {},
    )


    rmse = float(
        models[
            "rmse"
        ]
    )


    future = _preparar_future(
        exog_future
    )


    up_future = (
        _preparar_upstream_future(
            upstream_future
        )
    )


    # ========================================================
    # ÚLTIMA MEDICIÓN REAL COMO BASE
    # ========================================================

    local = preparar_nivel_local(
        df
    )


    latest_local_date = (
        local[
            "datetime"
        ].max()
    )


    last_date = history[
        "datetime"
    ].max()


    # --------------------------------------------------------
    # Agregar observaciones reales que falten en dataset.
    # --------------------------------------------------------

    if (
        pd.notna(
            latest_local_date
        )
        and latest_local_date
        > last_date
    ):

        missing = local[
            local[
                "datetime"
            ]
            > last_date
        ].copy()


        for _, obs in missing.iterrows():

            row = {

                "datetime":
                    obs[
                        "datetime"
                    ],

                "nivel":
                    float(
                        obs[
                            "nivel"
                        ]
                    ),

                "precip_mm":
                    0.0,

                "caudal_m3s":
                    _safe_last(
                        history[
                            "caudal_m3s"
                        ]
                    ),
            }


            for col in _upstream_cols(
                history
            ):

                row[
                    col
                ] = _safe_last(
                    history[
                        col
                    ]
                )


            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        [
                            row
                        ]
                    ),
                ],
                ignore_index=True,
            )


    last_date = history[
        "datetime"
    ].max()


    output = []


    # ========================================================
    # PRONÓSTICO DÍA POR DÍA
    # ========================================================

    for h in range(
        1,
        days
        + 1,
    ):

        target_date = (
            last_date
            + pd.Timedelta(
                days=h
            )
        )


        # ====================================================
        # LLUVIA
        # ====================================================

        precip_next = 0.0


        if not future.empty:

            match = future[
                future[
                    "datetime"
                ]
                == target_date
            ]


            if (
                not match.empty
                and "precip_mm"
                in match.columns
            ):

                value = pd.to_numeric(
                    match[
                        "precip_mm"
                    ].iloc[0],
                    errors="coerce",
                )


                if pd.notna(
                    value
                ):

                    precip_next = max(
                        float(
                            value
                        ),
                        0.0,
                    )


        # ====================================================
        # CAUDAL
        # ====================================================

        caudal_next = np.nan


        if not future.empty:

            match = future[
                future[
                    "datetime"
                ]
                == target_date
            ]


            if (
                not match.empty
                and "caudal_m3s"
                in match.columns
            ):

                value = pd.to_numeric(
                    match[
                        "caudal_m3s"
                    ].iloc[0],
                    errors="coerce",
                )


                if pd.notna(
                    value
                ):

                    caudal_next = float(
                        value
                    )


        if pd.isna(
            caudal_next
        ):

            caudal_next = (
                _proyectar_caudal_un_dia(
                    history
                )
            )


        # ====================================================
        # NIVELES AGUAS ARRIBA
        # ====================================================

        upstream_next = {}


        for col in _upstream_cols(
            history
        ):

            value = np.nan


            if (
                not up_future.empty
                and col
                in up_future.columns
            ):

                match = up_future[
                    up_future[
                        "datetime"
                    ]
                    == target_date
                ]


                if not match.empty:

                    value = pd.to_numeric(
                        match[
                            col
                        ].iloc[0],
                        errors="coerce",
                    )


            if pd.isna(
                value
            ):

                value = (
                    _proyectar_upstream_un_dia(
                        history,
                        col,
                    )
                )


            upstream_next[
                col
            ] = value


        # ====================================================
        # VARIABLES PARA RANDOM FOREST
        # ====================================================

        X = _crear_features_pronostico(

            history=
                history,

            feature_cols=
                feature_cols,

            target_date=
                target_date,

            precip_next=
                precip_next,

            caudal_next=
                caudal_next,

            upstream_next=
                upstream_next,

            fill_values=
                fill_values,
        )


        # ====================================================
        # PREDECIR VARIACIÓN DIARIA
        # ====================================================

        delta_model = float(
            model.predict(
                X
            )[0]
        )


        nivel_base = float(
            history[
                "nivel"
            ].iloc[-1]
        )


        # ====================================================
        # CONTROL DE PERSISTENCIA
        #
        # A horizontes largos no dejamos que RandomForest
        # repita indefinidamente un mismo cambio diario.
        # ====================================================

        recent_slope = _safe_slope(
            history[
                "nivel"
            ],
            window=7,
        )


        if h <= 15:

            model_weight = 1.00
            trend_weight = 0.00

        elif h <= 30:

            model_weight = 0.85
            trend_weight = 0.15

        elif h <= 45:

            model_weight = 0.65
            trend_weight = 0.35

        else:

            model_weight = 0.50
            trend_weight = 0.50


        trend_component = float(
            np.clip(
                recent_slope,
                -0.10,
                0.10,
            )
        )


        delta_pred = (
            model_weight
            * delta_model
            + trend_weight
            * trend_component
        )


        # ====================================================
        # CONTROL DE SALTO DIARIO
        # ====================================================

        if h <= 15:

            max_change = 0.20

        elif h <= 30:

            max_change = 0.16

        elif h <= 45:

            max_change = 0.13

        else:

            max_change = 0.10


        delta_pred = float(
            np.clip(
                delta_pred,
                -max_change,
                max_change,
            )
        )


        # ====================================================
        # NIVEL
        # ====================================================

        prediction = (
            nivel_base
            + delta_pred
        )


        prediction = float(
            np.clip(
                prediction,
                LEVEL_MIN,
                LEVEL_MAX,
            )
        )


        # ====================================================
        # INTERVALO DE INCERTIDUMBRE
        # ====================================================

        horizon_factor = (
            np.sqrt(
                max(
                    h,
                    1,
                )
            )
        )


        sigma = (
            rmse
            * horizon_factor
        )


        # En horizontes largos no expandir sin límite.

        sigma = min(
            sigma,
            1.50,
        )


        lower = float(
            np.clip(
                prediction
                - (
                    1.96
                    * sigma
                ),
                LEVEL_MIN,
                LEVEL_MAX,
            )
        )


        upper = float(
            np.clip(
                prediction
                + (
                    1.96
                    * sigma
                ),
                LEVEL_MIN,
                LEVEL_MAX,
            )
        )


        # ====================================================
        # CLASIFICAR HORIZONTE
        # ====================================================

        if h <= 15:

            horizon_group = (
                "1-15"
            )

        elif h <= 30:

            horizon_group = (
                "16-30"
            )

        elif h <= 45:

            horizon_group = (
                "31-45"
            )

        else:

            horizon_group = (
                "46-60"
            )


        # ====================================================
        # GUARDAR RESULTADO
        # ====================================================

        item = {

            "datetime":
                target_date,

            "prediction":
                prediction,

            "lower":
                lower,

            "upper":
                upper,

            "horizon_day":
                h,

            "horizon_group":
                horizon_group,

            "nivel_base":
                nivel_base,

            "variacion_dia":
                delta_pred,

            "variacion_modelo":
                delta_model,

            "precip_mm":
                precip_next,

            "caudal_m3s":
                caudal_next,
        }


        for (
            col,
            value,
        ) in upstream_next.items():

            item[
                col
            ] = value


        output.append(
            item
        )


        # ====================================================
        # NUEVA FILA PARA EL SIGUIENTE DÍA
        # ====================================================

        new_row = {

            "datetime":
                target_date,

            "nivel":
                prediction,

            "precip_mm":
                precip_next,

            "caudal_m3s":
                caudal_next,
        }


        for (
            col,
            value,
        ) in upstream_next.items():

            new_row[
                col
            ] = value


        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        new_row
                    ]
                ),
            ],
            ignore_index=True,
        )


    return pd.DataFrame(
        output
    )


# ============================================================
# PROBABILIDAD DE SUPERAR UMBRAL
# ============================================================

def prob(
    prediction,
    threshold,
    rmse,
):

    sigma = max(
        float(
            rmse
        ),
        0.05,
    )


    z = (
        float(
            prediction
        )
        - float(
            threshold
        )
    ) / sigma


    z = np.clip(
        z,
        -50,
        50,
    )


    return float(
        1.0
        / (
            1.0
            + np.exp(
                -z
            )
        )
    )
