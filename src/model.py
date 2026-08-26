import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_squared_error
)


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/model.py
# BASE V11
# ============================================================

MIN_OBSERVATIONS = 45

FORECAST_DAYS = 15

RANDOM_STATE = 42


# ============================================================
# NIVEL LOCAL
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

    if (
        "datetime"
        not in x.columns
    ):

        raise ValueError(
            "Falta datetime."
        )

    if "nivel" in x.columns:

        x[
            "nivel"
        ] = pd.to_numeric(
            x[
                "nivel"
            ],
            errors="coerce",
        )

    elif "value" in x.columns:

        x[
            "nivel"
        ] = pd.to_numeric(
            x[
                "value"
            ],
            errors="coerce",
        )

    else:

        raise ValueError(
            "Falta nivel/value."
        )

    x[
        "datetime"
    ] = pd.to_datetime(
        x[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    x[
        "datetime"
    ] = (
        x[
            "datetime"
        ]
        .dt.tz_localize(
            None
        )
        .dt.normalize()
    )

    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    x = (
        x.groupby(
            "datetime",
            as_index=False,
        )[
            "nivel"
        ]
        .mean()
    )

    return x


# ============================================================
# DATASET
# ============================================================

def preparar_dataset(
    df,
    exog_history=None,
    upstream_history=None,
):

    x = preparar_nivel_local(
        df
    )

    if (
        isinstance(
            exog_history,
            pd.DataFrame,
        )
        and not exog_history.empty
    ):

        exog = (
            exog_history.copy()
        )

        exog[
            "datetime"
        ] = pd.to_datetime(
            exog[
                "datetime"
            ],
            errors="coerce",
        ).dt.normalize()

        x = x.merge(
            exog,
            on="datetime",
            how="left",
        )

    if (
        isinstance(
            upstream_history,
            pd.DataFrame,
        )
        and not upstream_history.empty
    ):

        upstream = (
            upstream_history.copy()
        )

        upstream[
            "datetime"
        ] = pd.to_datetime(
            upstream[
                "datetime"
            ],
            errors="coerce",
        ).dt.normalize()

        x = x.merge(
            upstream,
            on="datetime",
            how="left",
        )

    if (
        "precip_mm"
        not in x.columns
    ):

        x[
            "precip_mm"
        ] = 0.0

    x[
        "precip_mm"
    ] = pd.to_numeric(
        x[
            "precip_mm"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    if (
        "caudal_m3s"
        not in x.columns
    ):

        x[
            "caudal_m3s"
        ] = np.nan

    x[
        "caudal_m3s"
    ] = pd.to_numeric(
        x[
            "caudal_m3s"
        ],
        errors="coerce",
    )

    upstream_cols = [
        c
        for c in x.columns
        if c.startswith(
            "nivel_"
        )
        and c != "nivel"
    ]

    for col in upstream_cols:

        x[
            col
        ] = pd.to_numeric(
            x[
                col
            ],
            errors="coerce",
        )

        x[
            col
        ] = (
            x[
                col
            ]
            .interpolate(
                limit=3,
                limit_direction=
                    "both",
            )
        )

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
                limit_direction=
                    "both"
            )
        )

    if len(
        x
    ) < MIN_OBSERVATIONS:

        raise ValueError(
            f"Se requieren al menos "
            f"{MIN_OBSERVATIONS} días. "
            f"Disponibles: {len(x)}."
        )

    return x


# ============================================================
# FEATURES
# ============================================================

def crear_features(
    df,
):

    x = df.copy()

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
            f"sn_lag_{lag}"
        ] = x[
            "nivel"
        ].shift(
            lag
        )

    x[
        "sn_diff1"
    ] = x[
        "nivel"
    ].diff()

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

    # --------------------------------------------------------
    # LLUVIA
    # --------------------------------------------------------

    x[
        "rain_0"
    ] = x[
        "precip_mm"
    ]

    x[
        "rain_1"
    ] = x[
        "precip_mm"
    ].shift(
        1
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

    # --------------------------------------------------------
    # CAUDAL
    # --------------------------------------------------------

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

        for lag in [
            1,
            3,
            5,
            7,
        ]:

            x[
                f"q_lag_{lag}"
            ] = x[
                "caudal_m3s"
            ].shift(
                lag
            )

        x[
            "q_diff1"
        ] = x[
            "caudal_m3s"
        ].diff()

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

    # --------------------------------------------------------
    # AGUAS ARRIBA
    # --------------------------------------------------------

    upstream_cols = [
        c
        for c in x.columns
        if c.startswith(
            "nivel_"
        )
        and c != "nivel"
        and "_lag" not in c
        and "_diff" not in c
        and "_trend" not in c
    ]

    for col in upstream_cols:

        x[
            f"{col}_actual"
        ] = x[
            col
        ]

        x[
            f"{col}_diff1"
        ] = x[
            col
        ].diff()

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
                f"{col}_lag{lag}"
            ] = x[
                col
            ].shift(
                lag
            )

    return x


# ============================================================
# TRAIN
# ============================================================

def train(
    df,
    exog_history=None,
    upstream_history=None,
):

    dataset = preparar_dataset(
        df,
        exog_history,
        upstream_history,
    )

    features = crear_features(
        dataset
    )

    excluded = [
        "datetime",
        "nivel",
        "precip_mm",
        "caudal_m3s",
    ]

    feature_cols = [
        c
        for c in features.columns
        if c not in excluded
    ]

    feature_cols = [
        c
        for c in feature_cols
        if features[
            c
        ].notna().sum()
        >= 15
    ]

    work = features.dropna(
        subset=[
            "nivel"
        ]
        + feature_cols
    )

    if len(
        work
    ) < 25:

        raise ValueError(
            "No quedan suficientes registros "
            "para entrenar el modelo: "
            f"{len(work)}."
        )

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

    X_test = work[
        feature_cols
    ].iloc[
        split:
    ]

    y_train = work[
        "nivel"
    ].iloc[
        :split
    ]

    y_test = work[
        "nivel"
    ].iloc[
        split:
    ]

    model = (
        RandomForestRegressor(
            n_estimators=700,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=
                RANDOM_STATE,
            n_jobs=-1,
        )
    )

    model.fit(
        X_train,
        y_train,
    )

    pred = model.predict(
        X_test
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                pred,
            )
        )
    )

    rmse = max(
        rmse,
        0.05,
    )

    final_model = (
        RandomForestRegressor(
            n_estimators=900,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
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
            "nivel"
        ],
    )

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

        "importance":
            importance,

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
            int(
                days
            ),
            15,
        ),
    )

    history = models[
        "dataset"
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
        isinstance(
            exog_future,
            pd.DataFrame,
        )
        and not exog_future.empty
    ):

        future = (
            exog_future.copy()
        )

        future[
            "datetime"
        ] = pd.to_datetime(
            future[
                "datetime"
            ],
            errors="coerce",
        ).dt.normalize()

    else:

        future = (
            pd.DataFrame()
        )

    last_date = history[
        "datetime"
    ].max()

    output = []

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

        precip = 0.0

        caudal = np.nan

        if not future.empty:

            match = future[
                future[
                    "datetime"
                ]
                == target_date
                .normalize()
            ]

            if not match.empty:

                if (
                    "precip_mm"
                    in match.columns
                ):

                    precip = float(
                        pd.to_numeric(
                            match[
                                "precip_mm"
                            ],
                            errors="coerce",
                        )
                        .fillna(
                            0
                        )
                        .iloc[
                            0
                        ]
                    )

                if (
                    "caudal_m3s"
                    in match.columns
                ):

                    value = (
                        pd.to_numeric(
                            match[
                                "caudal_m3s"
                            ],
                            errors="coerce",
                        )
                    )

                    if (
                        value
                        .notna()
                        .any()
                    ):

                        caudal = float(
                            value
                            .dropna()
                            .iloc[
                                0
                            ]
                        )

        row = {
            "datetime":
                target_date,

            "nivel":
                float(
                    history[
                        "nivel"
                    ].iloc[
                        -1
                    ]
                ),

            "precip_mm":
                precip,

            "caudal_m3s":
                caudal,
        }

        upstream_cols = [
            c
            for c in history.columns
            if c.startswith(
                "nivel_"
            )
            and c != "nivel"
        ]

        for col in upstream_cols:

            valid = (
                history[
                    col
                ]
                .dropna()
            )

            row[
                col
            ] = (
                float(
                    valid.iloc[
                        -1
                    ]
                )
                if len(
                    valid
                )
                else np.nan
            )

        trial = pd.concat(
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

        features = (
            crear_features(
                trial
            )
        )

        latest = (
            features
            .iloc[
                -1
            ]
        )

        values = {}

        for col in (
            feature_cols
        ):

            value = latest.get(
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

        X = pd.DataFrame(
            [
                values
            ]
        )

        prediction = float(
            model.predict(
                X
            )[0]
        )

        prediction = float(
            np.clip(
                prediction,
                0.0,
                7.0,
            )
        )

        lower = max(
            0.0,
            prediction
            - 1.96
            * rmse,
        )

        upper = min(
            7.0,
            prediction
            + 1.96
            * rmse,
        )

        output.append(
            {
                "datetime":
                    target_date,

                "prediction":
                    prediction,

                "lower":
                    lower,

                "upper":
                    upper,
            }
        )

        row[
            "nivel"
        ] = prediction

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

    return pd.DataFrame(
        output
    )
