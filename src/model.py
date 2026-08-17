import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ============================================================
# CONFIGURACIÓN
# ============================================================

MIN_OBSERVATIONS = 60

MAX_FORECAST_DAYS = 30

RANDOM_STATE = 42

LEVEL_MIN = 0.0
LEVEL_MAX = 7.0


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
        .tz_localize(
            None
        )
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


def _upstream_cols(
    df,
):

    return [
        c
        for c
        in df.columns
        if (
            c.startswith(
                "nivel_"
            )
            and c != "nivel"
            and "_lag"
            not in c
            and "_diff"
            not in c
            and "_trend"
            not in c
            and "_mean"
            not in c
            and "_actual"
            not in c
            and "_next"
            not in c
        )
    ]


def _safe_last(
    serie,
    default=np.nan,
):

    s = pd.to_numeric(
        serie,
        errors="coerce",
    ).dropna()

    if s.empty:

        return default

    return float(
        s.iloc[-1]
    )


def _safe_slope(
    serie,
    window=7,
):

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

    if len(
        s
    ) < 3:

        return 0.0

    x = np.arange(
        len(
            s
        ),
        dtype=float,
    )

    try:

        pendiente = float(
            np.polyfit(
                x,
                s.to_numpy(
                    dtype=float
                ),
                1,
            )[0]
        )

    except Exception:

        pendiente = 0.0

    return pendiente


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

    x[
        "nivel"
    ] = x[
        "nivel"
    ].clip(
        lower=LEVEL_MIN,
        upper=LEVEL_MAX,
    )

    return x


# ============================================================
# PREPARAR DATASET COMPLETO
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
    # LLUVIA + CAUDAL
    # ========================================================

    if (
        isinstance(
            exog_history,
            pd.DataFrame,
        )
        and not exog_history.empty
        and "datetime"
        in exog_history.columns
    ):

        exog = (
            exog_history
            .copy()
        )

        exog[
            "datetime"
        ] = _normalizar_datetime(
            exog[
                "datetime"
            ]
        )

        columnas = [
            c
            for c
            in [
                "datetime",
                "precip_mm",
                "caudal_m3s",
            ]
            if c
            in exog.columns
        ]

        exog = exog[
            columnas
        ].copy()

        x = x.merge(
            exog,
            on="datetime",
            how="left",
        )

    # ========================================================
    # NIVELES AGUAS ARRIBA
    # ========================================================

    if (
        isinstance(
            upstream_history,
            pd.DataFrame,
        )
        and not upstream_history.empty
        and "datetime"
        in upstream_history.columns
    ):

        upstream = (
            upstream_history
            .copy()
        )

        upstream[
            "datetime"
        ] = _normalizar_datetime(
            upstream[
                "datetime"
            ]
        )

        x = x.merge(
            upstream,
            on="datetime",
            how="left",
        )

    # ========================================================
    # COLUMNAS BASE
    # ========================================================

    if "precip_mm" not in x.columns:

        x[
            "precip_mm"
        ] = 0.0

    if "caudal_m3s" not in x.columns:

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
                limit=5,
                limit_direction="both",
            )
        )

    # ========================================================
    # INTERPOLAR NIVELES AGUAS ARRIBA
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

        x[
            col
        ] = (
            x[
                col
            ]
            .interpolate(
                limit=3,
                limit_direction="both",
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
# CREAR FEATURES
# ============================================================

def crear_features(
    df,
):

    x = df.copy()

    # ========================================================
    # NIVEL SAN NICOLÁS
    # ========================================================

    x[
        "sn_actual"
    ] = x[
        "nivel"
    ]

    for lag in LOCAL_LAGS:

        x[
            f"sn_lag_{lag}"
        ] = x[
            "nivel"
        ].shift(
            lag
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
        "sn_trend3"
    ] = (
        x[
            "nivel"
        ]
        - x[
            "nivel"
        ].shift(
            3
        )
    ) / 3.0

    x[
        "sn_trend7"
    ] = (
        x[
            "nivel"
        ]
        - x[
            "nivel"
        ].shift(
            7
        )
    ) / 7.0

    x[
        "sn_media3"
    ] = (
        x[
            "nivel"
        ]
        .shift(
            1
        )
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
        .shift(
            1
        )
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
        .shift(
            1
        )
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

    # Lluvia correspondiente al día siguiente.
    # Durante entrenamiento es histórica.
    # Durante pronóstico se reemplaza por la prevista.

    x[
        "rain_next"
    ] = x[
        "precip_mm"
    ].shift(
        -1
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

        # Caudal correspondiente al día siguiente.

        x[
            "q_next"
        ] = x[
            "caudal_m3s"
        ].shift(
            -1
        )

        for lag in Q_LAGS:

            x[
                f"q_lag_{lag}"
            ] = x[
                "caudal_m3s"
            ].shift(
                lag
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
                "caudal_m3s"
            ]
            - x[
                "caudal_m3s"
            ].shift(
                3
            )
        ) / 3.0

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
    # ESTACIONES AGUAS ARRIBA
    # ========================================================

    for col in _upstream_cols(
        x
    ):

        x[
            f"{col}_actual"
        ] = x[
            col
        ]

        # Valor del día siguiente.
        # Históricamente se conoce y sirve para entrenar.
        # En el futuro será reemplazado por la proyección.

        x[
            f"{col}_next"
        ] = x[
            col
        ].shift(
            -1
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
            ] = x[
                col
            ].shift(
                lag
            )

    # ========================================================
    # OBJETIVO DEL MODELO
    # ========================================================

    # Nivel real del día siguiente.

    x[
        "target_nivel"
    ] = x[
        "nivel"
    ].shift(
        -1
    )

    # Cambio real de nivel de un día al siguiente.

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
        exog_history=exog_history,
        upstream_history=upstream_history,
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
    # ELIMINAR FEATURES CON MUY POCOS DATOS
    # ========================================================

    feature_cols = [
        c
        for c
        in feature_cols
        if features[
            c
        ]
        .notna()
        .sum()
        >= 20
    ]

    work = (
        features
        .dropna(
            subset=[
                "target_delta",
            ]
            + feature_cols
        )
        .copy()
    )

    if len(
        work
    ) < 30:

        raise ValueError(
            "No quedan suficientes registros "
            "después de construir las variables: "
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
    # MODELO DE VARIACIÓN DIARIA
    # ========================================================

    model = RandomForestRegressor(
        n_estimators=700,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    pred_delta = model.predict(
        X_test
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
        0.05,
    )

    # ========================================================
    # MODELO FINAL
    # ========================================================

    final_model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
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
    # IMPORTANCIA DE VARIABLES
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

    models = {

        "model":
            final_model,

        "feature_cols":
            feature_cols,

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
    }

    return (
        models,
        metrics,
    )


# ============================================================
# PROYECTAR UN NIVEL AGUAS ARRIBA UN DÍA
# ============================================================

def _proyectar_upstream_un_dia(
    history,
    col,
):

    valid = (
        pd.to_numeric(
            history[
                col
            ],
            errors="coerce",
        )
        .dropna()
    )

    if valid.empty:

        return np.nan

    actual = float(
        valid.iloc[-1]
    )

    pendiente = _safe_slope(
        valid,
        window=7,
    )

    # Limitar saltos diarios irreales.

    limite = max(
        abs(
            actual
        )
        * 0.03,
        0.03,
    )

    pendiente = float(
        np.clip(
            pendiente,
            -limite,
            limite,
        )
    )

    valor = (
        actual
        + pendiente
    )

    return float(
        max(
            0.0,
            valor,
        )
    )


# ============================================================
# CREAR FEATURES PARA UN DÍA FUTURO
# ============================================================

def _crear_features_pronostico(
    history,
    feature_cols,
    target_date,
    precip_next,
    caudal_next,
    upstream_next,
):

    temp = history.copy()

    # Se agrega una fila futura únicamente para construir
    # rain_next, q_next y nivel_estacion_next correctamente.

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

    # La fila anterior contiene:
    #
    # nivel actual de San Nicolás
    # + lluvia del día siguiente
    # + caudal del día siguiente
    # + niveles aguas arriba del día siguiente

    source = feat.iloc[
        -2
    ]

    values = {}

    for col in feature_cols:

        value = source.get(
            col,
            np.nan,
        )

        if pd.isna(
            value
        ):

            value = 0.0

        values[
            col
        ] = float(
            value
        )

    return pd.DataFrame(
        [
            values
        ]
    )


# ============================================================
# PREPARAR FUTURO LLUVIA + CAUDAL
# ============================================================

def _preparar_future(
    exog_future,
):

    if (
        not isinstance(
            exog_future,
            pd.DataFrame,
        )
        or exog_future.empty
        or "datetime"
        not in exog_future.columns
    ):

        return pd.DataFrame()

    future = (
        exog_future
        .copy()
    )

    future[
        "datetime"
    ] = _normalizar_datetime(
        future[
            "datetime"
        ]
    )

    if "precip_mm" in future.columns:

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

    if "caudal_m3s" in future.columns:

        future[
            "caudal_m3s"
        ] = _to_numeric(
            future[
                "caudal_m3s"
            ]
        )

    future = (
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

    return future


# ============================================================
# PREPARAR FUTURO AGUAS ARRIBA
# ============================================================

def _preparar_upstream_future(
    upstream_future,
):

    if (
        not isinstance(
            upstream_future,
            pd.DataFrame,
        )
        or upstream_future.empty
        or "datetime"
        not in upstream_future.columns
    ):

        return pd.DataFrame()

    future = (
        upstream_future
        .copy()
    )

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

    future = (
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

    return future


# ============================================================
# PRONÓSTICO RECURSIVO 1 A 30 DÍAS
# ============================================================

def predict(
    df,
    models,
    days=15,
    exog_future=None,
    upstream_future=None,
):

    days = max(
        1,
        min(
            int(
                days
            ),
            MAX_FORECAST_DAYS,
        ),
    )

    if not isinstance(
        models,
        dict,
    ):

        raise ValueError(
            "models no es válido."
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
                f"Falta models['{key}']."
            )

    history = (
        models[
            "dataset"
        ]
        .copy()
    )

    model = models[
        "model"
    ]

    feature_cols = models[
        "feature_cols"
    ]

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

    last_date = history[
        "datetime"
    ].max()

    # ========================================================
    # ASEGURAR QUE LA ÚLTIMA MEDICIÓN REAL SEA LA BASE
    # ========================================================

    try:

        local = preparar_nivel_local(
            df
        )

        latest_local_date = local[
            "datetime"
        ].max()

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
            ][
                [
                    "datetime",
                    "nivel",
                ]
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

    except Exception:

        pass

    output = []

    # ========================================================
    # PRONÓSTICO DÍA POR DÍA
    # ========================================================

    for h in range(
        1,
        days + 1,
    ):

        target_date = (
            last_date
            + pd.Timedelta(
                days=h
            )
        )

        # ====================================================
        # LLUVIA DEL DÍA
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
                    ].iloc[
                        0
                    ],
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
        # CAUDAL DEL DÍA
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
                    ].iloc[
                        0
                    ],
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

            caudal_next = _safe_last(
                history[
                    "caudal_m3s"
                ],
                default=np.nan,
            )

        # ====================================================
        # NIVELES AGUAS ARRIBA DEL DÍA
        # ====================================================

        upstream_next = {}

        for col in _upstream_cols(
            history
        ):

            value = np.nan

            # Si tenemos una serie futura calculada
            # específicamente para aguas arriba, usarla.

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
                        ].iloc[
                            0
                        ],
                        errors="coerce",
                    )

            # Si todavía no existe serie futura oficial,
            # proyectar el comportamiento reciente.

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
        # CREAR VARIABLES DEL DÍA
        # ====================================================

        X = _crear_features_pronostico(
            history=history,
            feature_cols=feature_cols,
            target_date=target_date,
            precip_next=precip_next,
            caudal_next=caudal_next,
            upstream_next=upstream_next,
        )

        # ====================================================
        # PREDECIR VARIACIÓN DEL NIVEL
        # ====================================================

        delta_pred = float(
            model.predict(
                X
            )[0]
        )

        nivel_base = float(
            history[
                "nivel"
            ].iloc[
                -1
            ]
        )

        # ====================================================
        # CONTROL DE SALTO DIARIO
        # ====================================================

        max_daily_level_change = min(
            0.35,
            0.18
            + (
                0.005
                * h
            ),
        )

        delta_pred = float(
            np.clip(
                delta_pred,
                -max_daily_level_change,
                max_daily_level_change,
            )
        )

        # ====================================================
        # NIVEL RESULTANTE
        # ====================================================

        prediction = float(
            np.clip(
                nivel_base
                + delta_pred,
                LEVEL_MIN,
                LEVEL_MAX,
            )
        )

        # ====================================================
        # INTERVALO EXPERIMENTAL
        # ====================================================

        sigma = (
            rmse
            * np.sqrt(
                max(
                    h,
                    1,
                )
            )
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
        # GUARDAR RESULTADO DEL DÍA
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

            "nivel_base":
                nivel_base,

            "variacion_dia":
                delta_pred,

            "precip_mm":
                precip_next,

            "caudal_m3s":
                caudal_next,
        }

        for col, value in upstream_next.items():

            item[
                col
            ] = value

        output.append(
            item
        )

        # ====================================================
        # MUY IMPORTANTE
        #
        # EL NIVEL CALCULADO HOY PASA A SER LA BASE
        # PARA CALCULAR EL DÍA SIGUIENTE.
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

        for col, value in upstream_next.items():

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
