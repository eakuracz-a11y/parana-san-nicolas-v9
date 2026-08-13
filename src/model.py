import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ============================================================
# CONFIGURACIÓN
# ============================================================

MIN_OBSERVATIONS = 30

DEFAULT_FORECAST_DAYS = 15

RANDOM_STATE = 42


# ============================================================
# PREPARAR DATOS
# ============================================================

def preparar_serie(
    df,
    exog_history=None,
):

    if df is None or not isinstance(
        df,
        pd.DataFrame,
    ):

        raise ValueError(
            "No se recibieron datos válidos."
        )

    if df.empty:

        raise ValueError(
            "No hay observaciones disponibles."
        )

    work = df.copy()

    if "datetime" not in work.columns:

        raise ValueError(
            "No se encontró datetime."
        )

    if "nivel" in work.columns:

        work["nivel"] = pd.to_numeric(
            work["nivel"],
            errors="coerce",
        )

    elif "value" in work.columns:

        work["nivel"] = pd.to_numeric(
            work["value"],
            errors="coerce",
        )

    else:

        raise ValueError(
            "No se encontró nivel/value."
        )

    work["datetime"] = pd.to_datetime(
        work["datetime"],
        errors="coerce",
        utc=True,
    )

    work["datetime"] = (
        work["datetime"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    work = work.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    work = (
        work.groupby(
            "datetime",
            as_index=False,
        )["nivel"]
        .mean()
    )

    # ========================================================
    # VARIABLES EXTERNAS
    # ========================================================

    if (
        exog_history is not None
        and isinstance(
            exog_history,
            pd.DataFrame,
        )
        and not exog_history.empty
    ):

        exog = exog_history.copy()

        exog["datetime"] = pd.to_datetime(
            exog["datetime"],
            errors="coerce",
        )

        exog["datetime"] = (
            exog["datetime"]
            .dt.normalize()
        )

        work = work.merge(
            exog,
            on="datetime",
            how="left",
        )

    if "precip_mm" not in work.columns:

        work[
            "precip_mm"
        ] = 0.0

    work["precip_mm"] = pd.to_numeric(
        work["precip_mm"],
        errors="coerce",
    ).fillna(0.0)

    if "caudal_m3s" not in work.columns:

        work[
            "caudal_m3s"
        ] = np.nan

    work["caudal_m3s"] = pd.to_numeric(
        work["caudal_m3s"],
        errors="coerce",
    )

    if work[
        "caudal_m3s"
    ].notna().sum() >= 5:

        work[
            "caudal_m3s"
        ] = (
            work["caudal_m3s"]
            .interpolate(
                limit_direction="both"
            )
        )

    if len(work) < MIN_OBSERVATIONS:

        raise ValueError(
            f"Se requieren al menos "
            f"{MIN_OBSERVATIONS} observaciones. "
            f"Disponibles: {len(work)}."
        )

    return work


# ============================================================
# FEATURES
# ============================================================

def crear_features(df):

    x = df.copy()

    # ========================================================
    # NIVEL
    # ========================================================

    for lag in [
        1,
        2,
        3,
        7,
        14,
    ]:

        x[
            f"nivel_lag_{lag}"
        ] = x[
            "nivel"
        ].shift(lag)

    x["nivel_diff_1"] = (
        x["nivel"].diff()
    )

    x["nivel_media_3"] = (
        x["nivel"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    x["nivel_media_7"] = (
        x["nivel"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    x["nivel_trend_3"] = (
        x["nivel"].shift(1)
        - x["nivel"].shift(4)
    ) / 3.0

    # ========================================================
    # LLUVIA
    # ========================================================

    x[
        "precip_today"
    ] = x[
        "precip_mm"
    ]

    x[
        "precip_lag_1"
    ] = x[
        "precip_mm"
    ].shift(1)

    x[
        "precip_acum_3"
    ] = (
        x["precip_mm"]
        .rolling(3)
        .sum()
    )

    x[
        "precip_acum_7"
    ] = (
        x["precip_mm"]
        .rolling(7)
        .sum()
    )

    # ========================================================
    # CAUDAL
    # ========================================================

    if (
        "caudal_m3s" in x.columns
        and x[
            "caudal_m3s"
        ].notna().sum() >= 5
    ):

        x[
            "caudal_actual"
        ] = x[
            "caudal_m3s"
        ]

        x[
            "caudal_lag_1"
        ] = x[
            "caudal_m3s"
        ].shift(1)

        x[
            "caudal_diff_1"
        ] = x[
            "caudal_m3s"
        ].diff()

        x[
            "caudal_media_3"
        ] = (
            x["caudal_m3s"]
            .rolling(3)
            .mean()
        )

        x[
            "caudal_trend_3"
        ] = (
            x["caudal_m3s"]
            - x[
                "caudal_m3s"
            ].shift(3)
        ) / 3.0

    return x


# ============================================================
# TRAIN
# ============================================================

def train(
    df,
    exog_history=None,
):

    serie = preparar_serie(
        df,
        exog_history,
    )

    features = crear_features(
        serie
    )

    feature_cols = [
        c
        for c in features.columns
        if c not in [
            "datetime",
            "nivel",
            "precip_mm",
            "caudal_m3s",
        ]
    ]

    # Eliminar features completamente vacías
    feature_cols = [
        c
        for c in feature_cols
        if features[
            c
        ].notna().any()
    ]

    work = features.dropna(
        subset=[
            "nivel",
        ]
        + feature_cols
    )

    if len(work) < 15:

        raise ValueError(
            "Después de generar las variables "
            f"quedan solamente {len(work)} registros."
        )

    split = int(
        len(work) * 0.80
    )

    split = max(
        10,
        split,
    )

    split = min(
        split,
        len(work) - 1,
    )

    X_train = work[
        feature_cols
    ].iloc[:split]

    y_train = work[
        "nivel"
    ].iloc[:split]

    X_test = work[
        feature_cols
    ].iloc[split:]

    y_test = work[
        "nivel"
    ].iloc[split:]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    if len(X_test):

        p = model.predict(
            X_test
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    p,
                )
            )
        )

    else:

        rmse = 0.15

    rmse = max(
        rmse,
        0.05,
    )

    # ========================================================
    # MODELO FINAL
    # ========================================================

    final_model = RandomForestRegressor(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    final_model.fit(
        work[
            feature_cols
        ],
        work[
            "nivel"
        ],
    )

    models = {
        "model": final_model,
        "feature_cols": feature_cols,
        "rmse": rmse,
        "observations": len(
            serie
        ),
        "training_rows": len(
            work
        ),
        "history": serie,
        "uses_rain": True,
        "uses_caudal": any(
            c.startswith(
                "caudal_"
            )
            for c in feature_cols
        ),
    }

    metrics = {
        "RMSE": rmse,
        "observations": len(
            serie
        ),
        "training_rows": len(
            work
        ),
        "test_rows": len(
            X_test
        ),
    }

    return (
        models,
        metrics,
    )


# ============================================================
# CREAR FEATURES FUTURAS
# ============================================================

def crear_fila_futura(
    history,
    feature_cols,
):

    temp = crear_features(
        history
    )

    latest = temp.iloc[-1]

    row = {}

    for col in feature_cols:

        value = latest.get(
            col,
            0.0,
        )

        if pd.isna(
            value
        ):
            value = 0.0

        row[
            col
        ] = float(
            value
        )

    return pd.DataFrame(
        [
            row
        ]
    )


# ============================================================
# PREDICT
# ============================================================

def predict(
    df,
    models,
    days=15,
    exog_future=None,
):

    days = max(
        1,
        min(
            int(days),
            15,
        ),
    )

    history = models[
        "history"
    ].copy()

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

    if (
        exog_future is not None
        and isinstance(
            exog_future,
            pd.DataFrame,
        )
    ):

        future = exog_future.copy()

        future["datetime"] = (
            pd.to_datetime(
                future["datetime"],
                errors="coerce",
            )
            .dt.normalize()
        )

    else:

        future = pd.DataFrame()

    last_date = history[
        "datetime"
    ].max()

    output = []

    for h in range(
        1,
        days + 1,
    ):

        date_future = (
            last_date
            + pd.Timedelta(
                days=h
            )
        )

        rain = 0.0
        flow = np.nan

        if not future.empty:

            match = future[
                future[
                    "datetime"
                ]
                == date_future.normalize()
            ]

            if not match.empty:

                if (
                    "precip_mm"
                    in match.columns
                ):

                    rain = float(
                        match[
                            "precip_mm"
                        ].iloc[0]
                    )

                if (
                    "caudal_m3s"
                    in match.columns
                ):

                    flow = pd.to_numeric(
                        match[
                            "caudal_m3s"
                        ].iloc[0],
                        errors="coerce",
                    )

        new_exog = pd.DataFrame(
            {
                "datetime": [
                    date_future
                ],
                "nivel": [
                    history[
                        "nivel"
                    ].iloc[-1]
                ],
                "precip_mm": [
                    rain
                ],
                "caudal_m3s": [
                    flow
                ],
            }
        )

        trial = pd.concat(
            [
                history,
                new_exog,
            ],
            ignore_index=True,
        )

        X = crear_fila_futura(
            trial,
            feature_cols,
        )

        prediction = float(
            model.predict(
                X
            )[0]
        )

        sigma = (
            rmse
            * np.sqrt(
                h
            )
        )

        lower = max(
            0.0,
            prediction
            - 1.96
            * sigma,
        )

        upper = (
            prediction
            + 1.96
            * sigma
        )

        output.append(
            {
                "datetime": date_future,
                "prediction": prediction,
                "lower": lower,
                "upper": upper,
                "horizon_day": h,
                "precip_mm": rain,
                "caudal_m3s": flow,
            }
        )

        new_row = pd.DataFrame(
            {
                "datetime": [
                    date_future
                ],
                "nivel": [
                    prediction
                ],
                "precip_mm": [
                    rain
                ],
                "caudal_m3s": [
                    flow
                ],
            }
        )

        history = pd.concat(
            [
                history,
                new_row,
            ],
            ignore_index=True,
        )

    return pd.DataFrame(
        output
    )


# ============================================================
# PROBABILIDAD
# ============================================================

def prob(
    prediction,
    threshold,
    rmse,
):

    sigma = max(
        float(rmse),
        0.05,
    )

    z = (
        float(prediction)
        - float(threshold)
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
            + np.exp(-z)
        )
    )
