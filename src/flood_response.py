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

RESPONSE_HORIZON = 14

MIN_TRAINING_ROWS = 35

MAX_PUBLIC_LEVEL = 7.0


# ============================================================
# UTILIDADES
# ============================================================

def _numeric(
    series,
):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _safe_float(
    value,
    default=0.0,
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

    return float(
        default
    )


# ============================================================
# PREPARAR DATASET
# ============================================================

def prepare_flood_dataset(
    dataset,
):

    if (
        dataset is None
        or not isinstance(
            dataset,
            pd.DataFrame,
        )
        or dataset.empty
    ):

        raise ValueError(
            "No existe dataset histórico para "
            "entrenar el modelo de creciente."
        )

    df = dataset.copy()

    if (
        "datetime"
        not in df.columns
        or "nivel"
        not in df.columns
    ):

        raise ValueError(
            "El dataset debe contener datetime y nivel."
        )

    df[
        "datetime"
    ] = pd.to_datetime(
        df[
            "datetime"
        ],
        errors="coerce",
    )

    df[
        "nivel"
    ] = _numeric(
        df[
            "nivel"
        ]
    )

    if (
        "precip_mm"
        not in df.columns
    ):

        df[
            "precip_mm"
        ] = 0.0

    df[
        "precip_mm"
    ] = (
        _numeric(
            df[
                "precip_mm"
            ]
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0
        )
    )

    if (
        "caudal_m3s"
        not in df.columns
    ):

        df[
            "caudal_m3s"
        ] = np.nan

    df[
        "caudal_m3s"
    ] = _numeric(
        df[
            "caudal_m3s"
        ]
    )

    upstream_cols = [
        c
        for c in df.columns
        if (
            c.startswith(
                "nivel_"
            )
            and c != "nivel"
        )
    ]

    for col in upstream_cols:

        df[
            col
        ] = _numeric(
            df[
                col
            ]
        )

    df = (
        df
        .dropna(
            subset=[
                "datetime",
                "nivel",
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

    return (
        df,
        upstream_cols,
    )


# ============================================================
# CARACTERÍSTICAS HIDROLÓGICAS
# ============================================================

def build_flood_features(
    dataset,
):

    df, upstream_cols = (
        prepare_flood_dataset(
            dataset
        )
    )

    out = df.copy()

    # ========================================================
    # LLUVIA
    # ========================================================

    out[
        "stress_rain_1d"
    ] = out[
        "precip_mm"
    ]

    out[
        "stress_rain_3d"
    ] = (
        out[
            "precip_mm"
        ]
        .rolling(
            3,
            min_periods=1,
        )
        .sum()
    )

    out[
        "stress_rain_7d"
    ] = (
        out[
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

    q = out[
        "caudal_m3s"
    ]

    if q.notna().sum() >= 5:

        q = q.interpolate(
            limit_direction="both"
        )

        out[
            "stress_flow_k"
        ] = (
            q
            / 1000.0
        )

        out[
            "stress_flow_rise3_k"
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

        q_reference = (
            q.rolling(
                30,
                min_periods=5,
            )
            .median()
        )

        q_reference = (
            q_reference
            .replace(
                0,
                np.nan,
            )
        )

        out[
            "stress_flow_ratio"
        ] = (
            q
            / q_reference
        )

    else:

        out[
            "stress_flow_k"
        ] = 0.0

        out[
            "stress_flow_rise3_k"
        ] = 0.0

        out[
            "stress_flow_ratio"
        ] = 1.0

    # ========================================================
    # NIVELES AGUAS ARRIBA
    # ========================================================

    upstream_ratio_columns = []

    upstream_rise_columns = []

    for col in upstream_cols:

        values = out[
            col
        ]

        if values.notna().sum() < 5:

            continue

        values = values.interpolate(
            limit_direction="both"
        )

        reference = (
            values
            .rolling(
                30,
                min_periods=5,
            )
            .median()
        )

        reference = reference.replace(
            0,
            np.nan,
        )

        ratio_col = (
            f"__ratio_{col}"
        )

        rise_col = (
            f"__rise_{col}"
        )

        out[
            ratio_col
        ] = (
            values
            / reference
        )

        out[
            rise_col
        ] = (
            values
            - values.shift(
                3
            )
        ).clip(
            lower=0
        )

        upstream_ratio_columns.append(
            ratio_col
        )

        upstream_rise_columns.append(
            rise_col
        )

    if upstream_ratio_columns:

        out[
            "stress_upstream_ratio"
        ] = (
            out[
                upstream_ratio_columns
            ]
            .mean(
                axis=1
            )
        )

    else:

        out[
            "stress_upstream_ratio"
        ] = 1.0

    if upstream_rise_columns:

        out[
            "stress_upstream_rise3"
        ] = (
            out[
                upstream_rise_columns
            ]
            .mean(
                axis=1
            )
        )

    else:

        out[
            "stress_upstream_rise3"
        ] = 0.0

    # ========================================================
    # DINÁMICA LOCAL
    # ========================================================

    out[
        "stress_local_rise3"
    ] = (
        out[
            "nivel"
        ]
        - out[
            "nivel"
        ].shift(
            3
        )
    ).clip(
        lower=0
    )

    # ========================================================
    # LIMPIEZA
    # ========================================================

    feature_cols = [
        "stress_rain_1d",
        "stress_rain_3d",
        "stress_rain_7d",
        "stress_flow_k",
        "stress_flow_rise3_k",
        "stress_flow_ratio",
        "stress_upstream_ratio",
        "stress_upstream_rise3",
        "stress_local_rise3",
    ]

    for col in feature_cols:

        out[
            col
        ] = (
            pd.to_numeric(
                out[
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
        out,
        feature_cols,
        upstream_cols,
    )


# ============================================================
# TARGET:
# CRECIMIENTO MÁXIMO EN LOS SIGUIENTES 14 DÍAS
# ============================================================

def add_response_target(
    features,
):

    df = features.copy()

    levels = (
        df[
            "nivel"
        ]
        .to_numpy(
            dtype=float
        )
    )

    targets = np.full(
        len(
            df
        ),
        np.nan,
        dtype=float,
    )

    lags = np.full(
        len(
            df
        ),
        np.nan,
        dtype=float,
    )

    for i in range(
        len(
            df
        )
    ):

        start = (
            i
            + 1
        )

        end = min(
            len(
                df
            ),
            i
            + RESPONSE_HORIZON
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

        growth = (
            future_max
            - levels[
                i
            ]
        )

        targets[
            i
        ] = max(
            growth,
            0.0,
        )

        lags[
            i
        ] = (
            relative_index
            + 1
        )

    df[
        "target_growth"
    ] = targets

    df[
        "target_lag"
    ] = lags

    return df


# ============================================================
# ENTRENAR MODELO DE RESPUESTA A CRECIENTES
# ============================================================

def fit_flood_response(
    dataset,
):

    features, feature_cols, upstream_cols = (
        build_flood_features(
            dataset
        )
    )

    work = add_response_target(
        features
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
            "No existen suficientes registros "
            "para calibrar la respuesta a crecientes. "
            f"Disponibles: {len(work)}."
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
        20,
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

    # ========================================================
    # MODELO
    #
    # positive=True fuerza que una mayor condición de estrés
    # no reduzca artificialmente el crecimiento estimado.
    # ========================================================

    validation_model = Pipeline(
        steps=[
            (
                "scale",
                StandardScaler(),
            ),
            (
                "regression",
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

    test_prediction = (
        validation_model.predict(
            X_test
        )
    )

    test_prediction = np.clip(
        test_prediction,
        0.0,
        None,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                test_prediction,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_test,
            test_prediction,
        )
    )

    # ========================================================
    # MODELO FINAL
    # ========================================================

    model = Pipeline(
        steps=[
            (
                "scale",
                StandardScaler(),
            ),
            (
                "regression",
                LinearRegression(
                    positive=True,
                ),
            ),
        ]
    )

    model.fit(
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

    positive_events = work[
        work[
            "target_growth"
        ]
        > 0.05
    ]

    if not positive_events.empty:

        response_lag = int(
            round(
                positive_events[
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
            RESPONSE_HORIZON,
        )
    )

    # ========================================================
    # RESPUESTAS HISTÓRICAS
    # ========================================================

    historical_max_growth = float(
        work[
            "target_growth"
        ].max()
    )

    historical_p95_growth = float(
        work[
            "target_growth"
        ].quantile(
            0.95
        )
    )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "upstream_cols": upstream_cols,
        "training_rows": len(
            work
        ),
        "rmse": max(
            rmse,
            0.03,
        ),
        "mae": mae,
        "response_lag": response_lag,
        "historical_max_growth": (
            historical_max_growth
        ),
        "historical_p95_growth": (
            historical_p95_growth
        ),
        "training_frame": work,
    }


# ============================================================
# ESTADÍSTICAS NECESARIAS PARA UN ESCENARIO FUTURO
# ============================================================

def historical_reference(
    dataset,
):

    df, upstream_cols = (
        prepare_flood_dataset(
            dataset
        )
    )

    result = {
        "rain_max_day": float(
            df[
                "precip_mm"
            ].max()
        ),
        "rain_max_3d": float(
            df[
                "precip_mm"
            ]
            .rolling(
                3,
                min_periods=1,
            )
            .sum()
            .max()
        ),
        "rain_max_7d": float(
            df[
                "precip_mm"
            ]
            .rolling(
                7,
                min_periods=1,
            )
            .sum()
            .max()
        ),
        "flow_current": np.nan,
        "flow_max": np.nan,
        "flow_median": np.nan,
        "upstream": {},
    }

    flow = (
        df[
            "caudal_m3s"
        ]
        .dropna()
    )

    if not flow.empty:

        result[
            "flow_current"
        ] = float(
            flow.iloc[-1]
        )

        result[
            "flow_max"
        ] = float(
            flow.max()
        )

        result[
            "flow_median"
        ] = float(
            flow.median()
        )

    for col in upstream_cols:

        values = (
            df[
                col
            ]
            .dropna()
        )

        if values.empty:

            continue

        result[
            "upstream"
        ][
            col
        ] = {
            "current": float(
                values.iloc[-1]
            ),
            "maximum": float(
                values.max()
            ),
            "median": float(
                values.median()
            ),
        }

    return result


# ============================================================
# CREAR VECTOR DE ESTRÉS
# ============================================================

def build_stress_vector(
    response_model,
    dataset,
    rain_day,
    rain_3d,
    rain_7d,
    flow_peak,
    upstream_targets=None,
):

    reference = historical_reference(
        dataset
    )

    current_flow = reference.get(
        "flow_current"
    )

    flow_median = reference.get(
        "flow_median"
    )

    if not np.isfinite(
        current_flow
    ):

        current_flow = flow_peak

    if (
        not np.isfinite(
            flow_median
        )
        or flow_median == 0
    ):

        flow_median = max(
            current_flow,
            1.0,
        )

    flow_rise3 = max(
        (
            flow_peak
            - current_flow
        )
        / 1000.0,
        0.0,
    )

    flow_ratio = (
        flow_peak
        / flow_median
    )

    upstream_targets = (
        upstream_targets
        or {}
    )

    ratio_values = []

    rise_values = []

    for col, info in (
        reference[
            "upstream"
        ].items()
    ):

        current = info[
            "current"
        ]

        median = info[
            "median"
        ]

        target = upstream_targets.get(
            col,
            info[
                "maximum"
            ],
        )

        if (
            median is not None
            and median != 0
        ):

            ratio_values.append(
                target
                / median
            )

        rise_values.append(
            max(
                target
                - current,
                0.0,
            )
        )

    upstream_ratio = (
        float(
            np.mean(
                ratio_values
            )
        )
        if ratio_values
        else 1.0
    )

    upstream_rise = (
        float(
            np.mean(
                rise_values
            )
        )
        if rise_values
        else 0.0
    )

    values = {
        "stress_rain_1d": max(
            _safe_float(
                rain_day
            ),
            0.0,
        ),
        "stress_rain_3d": max(
            _safe_float(
                rain_3d
            ),
            0.0,
        ),
        "stress_rain_7d": max(
            _safe_float(
                rain_7d
            ),
            0.0,
        ),
        "stress_flow_k": max(
            _safe_float(
                flow_peak
            )
            / 1000.0,
            0.0,
        ),
        "stress_flow_rise3_k": (
            flow_rise3
        ),
        "stress_flow_ratio": max(
            _safe_float(
                flow_ratio,
                1.0,
            ),
            0.0,
        ),
        "stress_upstream_ratio": max(
            upstream_ratio,
            0.0,
        ),
        "stress_upstream_rise3": max(
            upstream_rise,
            0.0,
        ),
        "stress_local_rise3": 0.0,
    }

    return pd.DataFrame(
        [
            {
                col: values.get(
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


# ============================================================
# PREDECIR CRECIMIENTO BAJO ESTRÉS
# ============================================================

def predict_stress_growth(
    response_model,
    dataset,
    rain_day,
    rain_3d,
    rain_7d,
    flow_peak,
    upstream_targets=None,
):

    X = build_stress_vector(
        response_model=response_model,
        dataset=dataset,
        rain_day=rain_day,
        rain_3d=rain_3d,
        rain_7d=rain_7d,
        flow_peak=flow_peak,
        upstream_targets=upstream_targets,
    )

    growth = float(
        response_model[
            "model"
        ].predict(
            X
        )[0]
    )

    growth = max(
        growth,
        0.0,
    )

    current_level = float(
        pd.to_numeric(
            dataset[
                "nivel"
            ],
            errors="coerce",
        )
        .dropna()
        .iloc[-1]
    )

    growth = min(
        growth,
        max(
            MAX_PUBLIC_LEVEL
            - current_level,
            0.0,
        ),
    )

    return {
        "growth": growth,
        "peak_level": (
            current_level
            + growth
        ),
        "current_level": current_level,
        "response_lag": response_model[
            "response_lag"
        ],
        "rmse": response_model[
            "rmse"
        ],
        "mae": response_model[
            "mae"
        ],
        "training_rows": response_model[
            "training_rows"
        ],
        "historical_max_growth": (
            response_model[
                "historical_max_growth"
            ]
        ),
        "historical_p95_growth": (
            response_model[
                "historical_p95_growth"
            ]
        ),
    }
