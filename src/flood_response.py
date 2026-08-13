import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RESPONSE_HORIZON_DAYS = 14

MIN_TRAINING_ROWS = 60

MAX_LEVEL = 7.0


# ============================================================
# UNIFICAR HISTÓRICOS
# ============================================================

def merge_historical_data(
    level_history,
    rain_history,
    flow_history,
):

    if (
        level_history is None
        or level_history.empty
    ):

        raise ValueError(
            "No existe histórico de nivel "
            "de San Nicolás."
        )

    level = level_history.copy()

    level[
        "datetime"
    ] = pd.to_datetime(
        level[
            "datetime"
        ],
        errors="coerce",
    ).dt.normalize()

    level[
        "nivel"
    ] = pd.to_numeric(
        level[
            "nivel"
        ],
        errors="coerce",
    )

    level = level.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    # ========================================================
    # LLUVIA
    # ========================================================

    if (
        rain_history is not None
        and not rain_history.empty
    ):

        rain = rain_history.copy()

        rain[
            "datetime"
        ] = pd.to_datetime(
            rain[
                "datetime"
            ],
            errors="coerce",
        ).dt.normalize()

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
                lower=0
            )
        )

        level = level.merge(
            rain[
                [
                    "datetime",
                    "precip_mm",
                ]
            ],
            on="datetime",
            how="left",
        )

    else:

        level[
            "precip_mm"
        ] = 0.0

    # ========================================================
    # CAUDAL
    # ========================================================

    if (
        flow_history is not None
        and not flow_history.empty
    ):

        flow = flow_history.copy()

        flow[
            "datetime"
        ] = pd.to_datetime(
            flow[
                "datetime"
            ],
            errors="coerce",
        ).dt.normalize()

        flow[
            "caudal_m3s"
        ] = pd.to_numeric(
            flow[
                "caudal_m3s"
            ],
            errors="coerce",
        )

        level = level.merge(
            flow[
                [
                    "datetime",
                    "caudal_m3s",
                ]
            ],
            on="datetime",
            how="left",
        )

    else:

        level[
            "caudal_m3s"
        ] = np.nan

    # ========================================================
    # LIMPIEZA
    # ========================================================

    level[
        "precip_mm"
    ] = (
        level[
            "precip_mm"
        ]
        .fillna(
            0.0
        )
    )

    # Interpolación corta del caudal.
    # No queremos inventar períodos enormes.
    level[
        "caudal_m3s"
    ] = (
        level[
            "caudal_m3s"
        ]
        .interpolate(
            limit=7,
            limit_direction="both",
        )
    )

    level = (
        level
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

    return level


# ============================================================
# FEATURES
# ============================================================

def create_response_features(
    df,
):

    work = df.copy()

    # ========================================================
    # LLUVIA
    # ========================================================

    work[
        "rain_1d"
    ] = work[
        "precip_mm"
    ]

    work[
        "rain_3d"
    ] = (
        work[
            "precip_mm"
        ]
        .rolling(
            3,
            min_periods=1,
        )
        .sum()
    )

    work[
        "rain_7d"
    ] = (
        work[
            "precip_mm"
        ]
        .rolling(
            7,
            min_periods=1,
        )
        .sum()
    )

    # ========================================================
    # CAUDAL
    # ========================================================

    q = work[
        "caudal_m3s"
    ]

    work[
        "flow_k"
    ] = (
        q
        / 1000.0
    )

    work[
        "flow_rise_3d_k"
    ] = (
        (
            q
            - q.shift(
                3
            )
        )
        .clip(
            lower=0
        )
        / 1000.0
    )

    rolling_flow_reference = (
        q
        .rolling(
            30,
            min_periods=5,
        )
        .median()
        .replace(
            0,
            np.nan,
        )
    )

    work[
        "flow_ratio"
    ] = (
        q
        / rolling_flow_reference
    )

    # ========================================================
    # DINÁMICA DEL NIVEL
    # ========================================================

    work[
        "local_rise_3d"
    ] = (
        work[
            "nivel"
        ]
        - work[
            "nivel"
        ].shift(
            3
        )
    ).clip(
        lower=0
    )

    feature_cols = [
        "rain_1d",
        "rain_3d",
        "rain_7d",
        "flow_k",
        "flow_rise_3d_k",
        "flow_ratio",
        "local_rise_3d",
    ]

    for col in feature_cols:

        work[
            col
        ] = (
            pd.to_numeric(
                work[
                    col
                ],
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )

    return (
        work,
        feature_cols,
    )


# ============================================================
# TARGET:
# MÁXIMA CRECIDA POSTERIOR
# ============================================================

def create_growth_target(
    df,
):

    work = df.copy()

    levels = (
        work[
            "nivel"
        ]
        .to_numpy(
            dtype=float
        )
    )

    growth = np.full(
        len(
            work
        ),
        np.nan,
    )

    lag = np.full(
        len(
            work
        ),
        np.nan,
    )

    for i in range(
        len(
            work
        )
    ):

        start = (
            i
            + 1
        )

        end = min(
            len(
                work
            ),
            i
            + RESPONSE_HORIZON_DAYS
            + 1,
        )

        if start >= end:

            continue

        future = levels[
            start:
            end
        ]

        if len(
            future
        ) == 0:

            continue

        relative_index = int(
            np.argmax(
                future
            )
        )

        future_max = float(
            future[
                relative_index
            ]
        )

        delta = (
            future_max
            - float(
                levels[
                    i
                ]
            )
        )

        growth[
            i
        ] = max(
            delta,
            0.0,
        )

        lag[
            i
        ] = (
            relative_index
            + 1
        )

    work[
        "target_growth"
    ] = growth

    work[
        "target_lag"
    ] = lag

    return work


# ============================================================
# ENTRENAMIENTO
# ============================================================

def fit_flood_response(
    level_history,
    rain_history,
    flow_history,
):

    merged = merge_historical_data(
        level_history=level_history,
        rain_history=rain_history,
        flow_history=flow_history,
    )

    featured, feature_cols = (
        create_response_features(
            merged
        )
    )

    work = create_growth_target(
        featured
    )

    work = (
        work
        .dropna(
            subset=(
                feature_cols
                + [
                    "target_growth",
                    "target_lag",
                ]
            )
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        work
    ) < MIN_TRAINING_ROWS:

        raise ValueError(
            "No existen suficientes fechas históricas "
            "coincidentes para calibrar la creciente. "
            f"Registros disponibles: {len(work)}."
        )

    split = int(
        len(
            work
        )
        * 0.80
    )

    split = max(
        30,
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

    y_train = work[
        "target_growth"
    ].iloc[
        :split
    ]

    X_test = work[
        feature_cols
    ].iloc[
        split:
    ]

    y_test = work[
        "target_growth"
    ].iloc[
        split:
    ]

    validation_model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LinearRegression(
                    positive=True,
                ),
            ),
        ]
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

    validation_prediction = np.clip(
        validation_prediction,
        0.0,
        None,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                validation_prediction,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_test,
            validation_prediction,
        )
    )

    final_model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LinearRegression(
                    positive=True,
                ),
            ),
        ]
    )

    final_model.fit(
        work[
            feature_cols
        ],
        work[
            "target_growth"
        ],
    )

    # ========================================================
    # RETARDO HISTÓRICO
    # ========================================================

    growing_events = work[
        work[
            "target_growth"
        ] >= 0.05
    ]

    if not growing_events.empty:

        response_lag = int(
            round(
                growing_events[
                    "target_lag"
                ].median()
            )
        )

    else:

        response_lag = 3

    response_lag = int(
        np.clip(
            response_lag,
            1,
            RESPONSE_HORIZON_DAYS,
        )
    )

    return {
        "model":
            final_model,

        "feature_cols":
            feature_cols,

        "rmse":
            max(
                rmse,
                0.03,
            ),

        "mae":
            mae,

        "response_lag":
            response_lag,

        "training_rows":
            int(
                len(
                    work
                )
            ),

        "historical_max_growth":
            float(
                work[
                    "target_growth"
                ].max()
            ),

        "historical_p95_growth":
            float(
                work[
                    "target_growth"
                ].quantile(
                    0.95
                )
            ),

        "historical_median_growth":
            float(
                work[
                    "target_growth"
                ].median()
            ),
    }


# ============================================================
# PREDECIR CRECIMIENTO PARA UN DÍA FUTURO
# ============================================================

def predict_growth(
    response_model,
    current_level,
    rain_1d,
    rain_3d,
    rain_7d,
    flow_current,
    flow_change_3d,
    historical_flow_reference,
):

    flow_reference = max(
        float(
            historical_flow_reference
        ),
        1.0,
    )

    flow_ratio = (
        float(
            flow_current
        )
        / flow_reference
    )

    row = {
        "rain_1d":
            max(
                float(
                    rain_1d
                ),
                0.0,
            ),

        "rain_3d":
            max(
                float(
                    rain_3d
                ),
                0.0,
            ),

        "rain_7d":
            max(
                float(
                    rain_7d
                ),
                0.0,
            ),

        "flow_k":
            max(
                float(
                    flow_current
                )
                / 1000.0,
                0.0,
            ),

        "flow_rise_3d_k":
            max(
                float(
                    flow_change_3d
                )
                / 1000.0,
                0.0,
            ),

        "flow_ratio":
            max(
                flow_ratio,
                0.0,
            ),

        "local_rise_3d":
            0.0,
    }

    X = pd.DataFrame(
        [
            {
                col:
                    row.get(
                        col,
                        0.0,
                    )
                for col
                in response_model[
                    "feature_cols"
                ]
            }
        ]
    )

    growth = float(
        response_model[
            "model"
        ].predict(
            X
        )[0]
    )

    # Escenario específicamente de CRECIDA:
    # jamás reducimos el nivel base.
    growth = max(
        growth,
        0.0,
    )

    max_allowed_growth = max(
        MAX_LEVEL
        - float(
            current_level
        ),
        0.0,
    )

    growth = min(
        growth,
        max_allowed_growth,
    )

    return growth
