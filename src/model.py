import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/model.py
# V11.2 - PROPAGACIÓN AGUAS ARRIBA
# ============================================================

MIN_OBSERVATIONS = 45
FORECAST_DAYS = 15
RANDOM_STATE = 42

Y_MIN = 0.0
Y_MAX = 7.0


# ============================================================
# RETARDOS HIDROLÓGICOS APROXIMADOS
# días entre estación y San Nicolás
# ============================================================

UPSTREAM_LAGS = {
    "nivel_corrientes": [5, 7, 10, 12, 14],
    "nivel_goya": [4, 5, 7, 9, 12],
    "nivel_la_paz": [3, 4, 5, 7, 9],
    "nivel_parana": [2, 3, 4, 5, 7],
    "nivel_diamante": [2, 3, 4, 5],
    "nivel_rosario": [1, 2, 3, 4],
    "nivel_villa_constitucion": [1, 2, 3],
}


# ============================================================
# NIVEL LOCAL
# ============================================================

def preparar_nivel_local(df):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
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

        x["nivel"] = pd.to_numeric(
            x["nivel"],
            errors="coerce",
        )

    elif "value" in x.columns:

        x["nivel"] = pd.to_numeric(
            x["value"],
            errors="coerce",
        )

    else:
        raise ValueError(
            "Falta la columna nivel/value."
        )

    x["datetime"] = pd.to_datetime(
        x["datetime"],
        errors="coerce",
        utc=True,
    )

    x["datetime"] = (
        x["datetime"]
        .dt.tz_localize(None)
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
        )["nivel"]
        .mean()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return x


# ============================================================
# UNIR DATASET
# ============================================================

def preparar_dataset(
    df,
    exog_history=None,
    upstream_history=None,
):

    x = preparar_nivel_local(df)

    # --------------------------------------------------------
    # LLUVIA + CAUDAL
    # --------------------------------------------------------

    if (
        isinstance(exog_history, pd.DataFrame)
        and not exog_history.empty
        and "datetime" in exog_history.columns
    ):

        exog = exog_history.copy()

        exog["datetime"] = pd.to_datetime(
            exog["datetime"],
            errors="coerce",
        ).dt.normalize()

        x = x.merge(
            exog,
            on="datetime",
            how="left",
        )

    # --------------------------------------------------------
    # ESTACIONES AGUAS ARRIBA
    # --------------------------------------------------------

    if (
        isinstance(upstream_history, pd.DataFrame)
        and not upstream_history.empty
        and "datetime" in upstream_history.columns
    ):

        upstream = upstream_history.copy()

        upstream["datetime"] = pd.to_datetime(
            upstream["datetime"],
            errors="coerce",
        ).dt.normalize()

        x = x.merge(
            upstream,
            on="datetime",
            how="left",
        )

    # --------------------------------------------------------
    # LLUVIA
    # --------------------------------------------------------

    if "precip_mm" not in x.columns:
        x["precip_mm"] = 0.0

    x["precip_mm"] = pd.to_numeric(
        x["precip_mm"],
        errors="coerce",
    ).fillna(0.0)

    x["precip_mm"] = x["precip_mm"].clip(
        lower=0.0
    )

    # --------------------------------------------------------
    # CAUDAL
    # --------------------------------------------------------

    if "caudal_m3s" not in x.columns:
        x["caudal_m3s"] = np.nan

    x["caudal_m3s"] = pd.to_numeric(
        x["caudal_m3s"],
        errors="coerce",
    )

    if x["caudal_m3s"].notna().sum() >= 7:

        x["caudal_m3s"] = (
            x["caudal_m3s"]
            .interpolate(
                limit_direction="both"
            )
        )

    # --------------------------------------------------------
    # AGUAS ARRIBA
    # --------------------------------------------------------

    upstream_cols = [
        c
        for c in x.columns
        if c.startswith("nivel_")
        and c != "nivel"
    ]

    for col in upstream_cols:

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

        if x[col].notna().sum() >= 3:

            x[col] = (
                x[col]
                .interpolate(
                    limit=3,
                    limit_direction="both",
                )
            )

    x = (
        x.sort_values("datetime")
        .reset_index(drop=True)
    )

    if len(x) < MIN_OBSERVATIONS:

        raise ValueError(
            f"Se requieren al menos "
            f"{MIN_OBSERVATIONS} días. "
            f"Disponibles: {len(x)}."
        )

    return x


# ============================================================
# CORRELACIÓN RETARDADA
# ============================================================

def calcular_mejor_lag(
    df,
    upstream_col,
    max_lag=20,
):

    if (
        upstream_col not in df.columns
        or "nivel" not in df.columns
    ):
        return None, None

    best_lag = None
    best_corr = None

    for lag in range(
        1,
        max_lag + 1,
    ):

        test = pd.DataFrame(
            {
                "sn":
                    pd.to_numeric(
                        df["nivel"],
                        errors="coerce",
                    ),

                "up":
                    pd.to_numeric(
                        df[upstream_col],
                        errors="coerce",
                    ).shift(lag),
            }
        ).dropna()

        if len(test) < 20:
            continue

        corr = test[
            "sn"
        ].corr(
            test["up"]
        )

        if pd.isna(corr):
            continue

        if (
            best_corr is None
            or abs(corr)
            > abs(best_corr)
        ):

            best_corr = float(corr)
            best_lag = int(lag)

    return (
        best_lag,
        best_corr,
    )


# ============================================================
# FEATURES
# ============================================================

def crear_features(df):

    x = df.copy()

    # ========================================================
    # SAN NICOLÁS
    # ========================================================

    for lag in [
        1,
        2,
        3,
        5,
        7,
        10,
        14,
    ]:

        x[f"sn_lag_{lag}"] = (
            x["nivel"].shift(lag)
        )

    x["sn_diff1"] = (
        x["nivel"].diff()
    )

    x["sn_diff3"] = (
        x["nivel"]
        - x["nivel"].shift(3)
    )

    x["sn_diff7"] = (
        x["nivel"]
        - x["nivel"].shift(7)
    )

    x["sn_trend3"] = (
        x["sn_diff3"] / 3.0
    )

    x["sn_trend7"] = (
        x["sn_diff7"] / 7.0
    )

    x["sn_media3"] = (
        x["nivel"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    x["sn_media7"] = (
        x["nivel"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    x["sn_media14"] = (
        x["nivel"]
        .shift(1)
        .rolling(14)
        .mean()
    )

    # ========================================================
    # PRECIPITACIÓN
    # ========================================================

    if "precip_mm" in x.columns:

        rain = pd.to_numeric(
            x["precip_mm"],
            errors="coerce",
        ).fillna(0.0)

        x["rain_0"] = rain

        for lag in [
            1,
            2,
            3,
            5,
            7,
        ]:

            x[f"rain_lag_{lag}"] = (
                rain.shift(lag)
            )

        x["rain_3d"] = (
            rain.rolling(3).sum()
        )

        x["rain_7d"] = (
            rain.rolling(7).sum()
        )

        x["rain_14d"] = (
            rain.rolling(14).sum()
        )

    # ========================================================
    # CAUDAL
    # ========================================================

    if (
        "caudal_m3s" in x.columns
        and x["caudal_m3s"].notna().sum()
        >= 10
    ):

        q = pd.to_numeric(
            x["caudal_m3s"],
            errors="coerce",
        )

        x["q_actual"] = q

        for lag in [
            1,
            2,
            3,
            5,
            7,
            10,
        ]:

            x[f"q_lag_{lag}"] = (
                q.shift(lag)
            )

        x["q_diff1"] = (
            q.diff()
        )

        x["q_diff3"] = (
            q
            - q.shift(3)
        )

        x["q_trend3"] = (
            x["q_diff3"] / 3.0
        )

        x["q_trend7"] = (
            (
                q
                - q.shift(7)
            )
            / 7.0
        )

    # ========================================================
    # AGUAS ARRIBA
    # ========================================================

    upstream_cols = [
        c
        for c in x.columns
        if c.startswith("nivel_")
        and c != "nivel"
        and "_lag" not in c
        and "_diff" not in c
        and "_trend" not in c
        and "_media" not in c
        and "_actual" not in c
    ]

    for col in upstream_cols:

        serie = pd.to_numeric(
            x[col],
            errors="coerce",
        )

        x[f"{col}_actual"] = serie

        x[f"{col}_diff1"] = (
            serie.diff()
        )

        x[f"{col}_diff3"] = (
            serie
            - serie.shift(3)
        )

        x[f"{col}_trend3"] = (
            x[f"{col}_diff3"] / 3.0
        )

        x[f"{col}_trend7"] = (
            (
                serie
                - serie.shift(7)
            )
            / 7.0
        )

        # ----------------------------------------------------
        # RETARDOS ESPECÍFICOS POR ESTACIÓN
        # ----------------------------------------------------

        lags = UPSTREAM_LAGS.get(
            col,
            [
                1,
                2,
                3,
                5,
                7,
            ],
        )

        for lag in lags:

            x[
                f"{col}_lag{lag}"
            ] = serie.shift(
                lag
            )

    return x


# ============================================================
# SELECCIONAR FEATURES
# ============================================================

def seleccionar_features(features):

    excluded = {
        "datetime",
        "nivel",
        "precip_mm",
        "caudal_m3s",
    }

    feature_cols = []

    for col in features.columns:

        if col in excluded:
            continue

        numeric = pd.to_numeric(
            features[col],
            errors="coerce",
        )

        if numeric.notna().sum() < 15:
            continue

        if numeric.nunique(
            dropna=True
        ) <= 1:
            continue

        features[col] = numeric

        feature_cols.append(
            col
        )

    return (
        features,
        feature_cols,
    )


# ============================================================
# RESUMEN DE RELACIÓN AGUAS ARRIBA
# ============================================================

def resumen_niveles_estaciones(
    df,
    upstream_history=None,
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

    base = preparar_nivel_local(
        df
    )

    up = upstream_history.copy()

    up["datetime"] = pd.to_datetime(
        up["datetime"],
        errors="coerce",
    ).dt.normalize()

    merged = base.merge(
        up,
        on="datetime",
        how="inner",
    )

    rows = []

    upstream_cols = [
        c
        for c in merged.columns
        if c.startswith("nivel_")
    ]

    for col in upstream_cols:

        lag, corr = calcular_mejor_lag(
            merged,
            col,
            max_lag=20,
        )

        values = pd.to_numeric(
            merged[col],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        actual = float(
            values.iloc[-1]
        )

        anterior = None
        variacion = None

        if len(values) >= 2:

            anterior = float(
                values.iloc[-2]
            )

            variacion = (
                actual
                - anterior
            )

        rows.append(
            {
                "estacion":
                    col.replace(
                        "nivel_",
                        "",
                    )
                    .replace(
                        "_",
                        " ",
                    )
                    .title(),

                "nivel_actual":
                    actual,

                "nivel_anterior":
                    anterior,

                "variacion":
                    variacion,

                "mejor_lag_dias":
                    lag,

                "correlacion":
                    corr,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TRAIN
# ============================================================

def train(
    df,
    exog_history=None,
    upstream_history=None,
):

    dataset = preparar_dataset(
        df=df,
        exog_history=exog_history,
        upstream_history=upstream_history,
    )

    features = crear_features(
        dataset
    )

    (
        features,
        feature_cols,
    ) = seleccionar_features(
        features
    )

    if not feature_cols:

        raise ValueError(
            "No se pudieron construir "
            "variables válidas."
        )

    work = features.dropna(
        subset=[
            "nivel"
        ]
        + feature_cols
    ).copy()

    if len(work) < 25:

        raise ValueError(
            "No quedan suficientes registros "
            "después de construir los retardos. "
            f"Disponibles: {len(work)}."
        )

    # ========================================================
    # VALIDACIÓN TEMPORAL
    # ========================================================

    split = int(
        len(work) * 0.80
    )

    split = max(
        20,
        split,
    )

    split = min(
        split,
        len(work) - 1,
    )

    X_train = (
        work[
            feature_cols
        ]
        .iloc[:split]
    )

    X_test = (
        work[
            feature_cols
        ]
        .iloc[split:]
    )

    y_train = (
        work[
            "nivel"
        ]
        .iloc[:split]
    )

    y_test = (
        work[
            "nivel"
        ]
        .iloc[split:]
    )

    if len(X_test) == 0:

        raise ValueError(
            "No quedaron datos para validar."
        )

    validation_model = (
        RandomForestRegressor(
            n_estimators=650,
            max_depth=14,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=
                RANDOM_STATE,
            n_jobs=-1,
        )
    )

    validation_model.fit(
        X_train,
        y_train,
    )

    pred_test = (
        validation_model.predict(
            X_test
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                pred_test,
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

    final_model = (
        RandomForestRegressor(
            n_estimators=900,
            max_depth=14,
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

    relation_summary = (
        resumen_niveles_estaciones(
            df,
            upstream_history,
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

        "relation_summary":
            relation_summary,

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
            len(dataset),

        "training_rows":
            len(work),

        "test_rows":
            len(X_test),

        "features":
            len(feature_cols),
    }

    return (
        models,
        metrics,
    )


# ============================================================
# FUTURO EXÓGENO
# ============================================================

def preparar_exog_future(
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

    if "precip_mm" not in future.columns:
        future["precip_mm"] = 0.0

    future["precip_mm"] = pd.to_numeric(
        future["precip_mm"],
        errors="coerce",
    ).fillna(0.0)

    if "caudal_m3s" not in future.columns:
        future["caudal_m3s"] = np.nan

    future["caudal_m3s"] = pd.to_numeric(
        future["caudal_m3s"],
        errors="coerce",
    )

    return future


# ============================================================
# X FUTURA
# ============================================================

def construir_X_futura(
    history,
    feature_cols,
):

    temp = crear_features(
        history
    )

    latest = temp.iloc[-1]

    values = {}

    for col in feature_cols:

        value = latest.get(
            col,
            np.nan,
        )

        value = pd.to_numeric(
            pd.Series(
                [value]
            ),
            errors="coerce",
        ).iloc[0]

        if pd.isna(value):
            value = 0.0

        values[col] = float(
            value
        )

    return pd.DataFrame(
        [values],
        columns=feature_cols,
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

    if (
        models is None
        or "model"
        not in models
    ):

        raise ValueError(
            "El modelo no está entrenado."
        )

    days = max(
        1,
        min(
            int(days),
            FORECAST_DAYS,
        ),
    )

    model = models[
        "model"
    ]

    feature_cols = models[
        "feature_cols"
    ]

    rmse = float(
        models.get(
            "rmse",
            0.10,
        )
    )

    history = (
        models[
            "dataset"
        ].copy()
    )

    history[
        "datetime"
    ] = pd.to_datetime(
        history[
            "datetime"
        ],
        errors="coerce",
    ).dt.normalize()

    history = (
        history
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    future = preparar_exog_future(
        exog_future
    )

    last_date = (
        history[
            "datetime"
        ].iloc[-1]
    )

    upstream_cols = [
        c
        for c in history.columns
        if c.startswith("nivel_")
        and c != "nivel"
    ]

    output = []

    # ========================================================
    # PRONÓSTICO RECURSIVO
    # ========================================================

    for step in range(
        1,
        days + 1,
    ):

        target_date = (
            last_date
            + pd.Timedelta(
                days=step
            )
        )

        precip = 0.0
        caudal = np.nan

        # ----------------------------------------------------
        # LLUVIA Y CAUDAL FUTURO
        # ----------------------------------------------------

        if not future.empty:

            match = future[
                future[
                    "datetime"
                ]
                == target_date
            ]

            if not match.empty:

                if (
                    "precip_mm"
                    in match.columns
                ):

                    rain_value = (
                        pd.to_numeric(
                            match[
                                "precip_mm"
                            ],
                            errors="coerce",
                        )
                    )

                    if rain_value.notna().any():

                        precip = float(
                            rain_value
                            .dropna()
                            .iloc[0]
                        )

                if (
                    "caudal_m3s"
                    in match.columns
                ):

                    q_value = (
                        pd.to_numeric(
                            match[
                                "caudal_m3s"
                            ],
                            errors="coerce",
                        )
                    )

                    if q_value.notna().any():

                        caudal = float(
                            q_value
                            .dropna()
                            .iloc[0]
                        )

        # ----------------------------------------------------
        # ÚLTIMO CAUDAL SI NO HAY FUTURO
        # ----------------------------------------------------

        if (
            pd.isna(caudal)
            and "caudal_m3s"
            in history.columns
        ):

            q_valid = (
                history[
                    "caudal_m3s"
                ]
                .dropna()
            )

            if not q_valid.empty:

                caudal = float(
                    q_valid.iloc[-1]
                )

        row = {
            "datetime":
                target_date,

            "nivel":
                float(
                    history[
                        "nivel"
                    ].iloc[-1]
                ),

            "precip_mm":
                precip,

            "caudal_m3s":
                caudal,
        }

        # ----------------------------------------------------
        # AGUAS ARRIBA
        #
        # Por ahora se mantiene el último valor observado.
        # Los retardos históricos continúan propagándose
        # durante los primeros días del forecast.
        # ----------------------------------------------------

        for col in upstream_cols:

            valid = (
                history[
                    col
                ]
                .dropna()
            )

            if not valid.empty:

                row[col] = float(
                    valid.iloc[-1]
                )

            else:

                row[col] = np.nan

        trial = pd.concat(
            [
                history,
                pd.DataFrame(
                    [row]
                ),
            ],
            ignore_index=True,
        )

        X_future = (
            construir_X_futura(
                trial,
                feature_cols,
            )
        )

        prediction = float(
            model.predict(
                X_future
            )[0]
        )

        # ----------------------------------------------------
        # EVITAR SALTOS IRREALES
        # ----------------------------------------------------

        previous = float(
            history[
                "nivel"
            ].iloc[-1]
        )

        max_daily_step = 0.18

        prediction = float(
            np.clip(
                prediction,
                previous
                - max_daily_step,
                previous
                + max_daily_step,
            )
        )

        prediction = float(
            np.clip(
                prediction,
                Y_MIN,
                Y_MAX,
            )
        )

        # ----------------------------------------------------
        # INCERTIDUMBRE
        # ----------------------------------------------------

        horizon_factor = (
            1.0
            + 0.045
            * (
                step - 1
            )
        )

        uncertainty = (
            1.96
            * rmse
            * horizon_factor
        )

        lower = float(
            np.clip(
                prediction
                - uncertainty,
                Y_MIN,
                Y_MAX,
            )
        )

        upper = float(
            np.clip(
                prediction
                + uncertainty,
                Y_MIN,
                Y_MAX,
            )
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

                "delta_prediction":
                    prediction
                    - previous,

                "precip_mm":
                    precip,

                "caudal_m3s":
                    caudal,
            }
        )

        row[
            "nivel"
        ] = prediction

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [row]
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(
        output
    )
