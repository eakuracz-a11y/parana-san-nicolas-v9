import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/model.py
# BASE V11.0 ESTABLE
# ============================================================

MIN_OBSERVATIONS = 45
FORECAST_DAYS = 15
RANDOM_STATE = 42

Y_MIN = 0.0
Y_MAX = 7.0


# ============================================================
# PREPARAR NIVEL DE SAN NICOLÁS
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
            "Los datos no contienen la columna datetime."
        )

    # --------------------------------------------------------
    # NIVEL
    # --------------------------------------------------------

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
            "Los datos no contienen nivel ni value."
        )

    # --------------------------------------------------------
    # FECHA
    # --------------------------------------------------------

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

    if x.empty:
        raise ValueError(
            "No quedaron observaciones válidas "
            "de San Nicolás."
        )

    # Una observación diaria
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
# PREPARAR DATASET GENERAL
# ============================================================

def preparar_dataset(
    df,
    exog_history=None,
    upstream_history=None,
):

    x = preparar_nivel_local(df)

    # ========================================================
    # LLUVIA + CAUDAL
    # ========================================================

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

        exog = exog.dropna(
            subset=["datetime"]
        )

        x = x.merge(
            exog,
            on="datetime",
            how="left",
        )

    # ========================================================
    # ESTACIONES AGUAS ARRIBA
    # ========================================================

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

        upstream = upstream.dropna(
            subset=["datetime"]
        )

        x = x.merge(
            upstream,
            on="datetime",
            how="left",
        )

    # ========================================================
    # PRECIPITACIÓN
    # ========================================================

    if "precip_mm" not in x.columns:
        x["precip_mm"] = 0.0

    x["precip_mm"] = pd.to_numeric(
        x["precip_mm"],
        errors="coerce",
    ).fillna(0.0)

    x["precip_mm"] = x["precip_mm"].clip(
        lower=0.0
    )

    # ========================================================
    # CAUDAL
    # ========================================================

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

    # ========================================================
    # NIVELES AGUAS ARRIBA
    # ========================================================

    upstream_cols = [
        col
        for col in x.columns
        if col.startswith("nivel_")
        and col != "nivel"
    ]

    for col in upstream_cols:

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

        # Interpolar pequeños huecos
        if x[col].notna().sum() >= 3:

            x[col] = (
                x[col]
                .interpolate(
                    limit=3,
                    limit_direction="both",
                )
            )

    x = x.sort_values(
        "datetime"
    ).reset_index(drop=True)

    if len(x) < MIN_OBSERVATIONS:

        raise ValueError(
            f"Se requieren al menos "
            f"{MIN_OBSERVATIONS} días para entrenar. "
            f"Disponibles: {len(x)}."
        )

    return x


# ============================================================
# CREAR FEATURES
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

    x["sn_trend3"] = (
        x["nivel"]
        - x["nivel"].shift(3)
    ) / 3.0

    x["sn_trend7"] = (
        x["nivel"]
        - x["nivel"].shift(7)
    ) / 7.0

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

        x["rain_1"] = (
            rain.shift(1)
        )

        x["rain_2"] = (
            rain.shift(2)
        )

        x["rain_3d"] = (
            rain
            .rolling(3)
            .sum()
        )

        x["rain_7d"] = (
            rain
            .rolling(7)
            .sum()
        )

        x["rain_14d"] = (
            rain
            .rolling(14)
            .sum()
        )

    # ========================================================
    # CAUDAL
    # ========================================================

    if (
        "caudal_m3s" in x.columns
        and x["caudal_m3s"].notna().sum() >= 10
    ):

        q = pd.to_numeric(
            x["caudal_m3s"],
            errors="coerce",
        )

        x["q_actual"] = q

        for lag in [
            1,
            3,
            5,
            7,
        ]:

            x[f"q_lag_{lag}"] = (
                q.shift(lag)
            )

        x["q_diff1"] = (
            q.diff()
        )

        x["q_trend3"] = (
            q
            - q.shift(3)
        ) / 3.0

        x["q_trend7"] = (
            q
            - q.shift(7)
        ) / 7.0

    # ========================================================
    # ESTACIONES AGUAS ARRIBA
    # ========================================================

    upstream_cols = [
        col
        for col in x.columns
        if col.startswith("nivel_")
        and col != "nivel"
        and "_lag" not in col
        and "_diff" not in col
        and "_trend" not in col
        and "_media" not in col
        and "_actual" not in col
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

        x[f"{col}_trend3"] = (
            serie
            - serie.shift(3)
        ) / 3.0

        x[f"{col}_trend7"] = (
            serie
            - serie.shift(7)
        ) / 7.0

        for lag in [
            1,
            2,
            3,
            5,
            7,
            10,
            14,
        ]:

            x[f"{col}_lag{lag}"] = (
                serie.shift(lag)
            )

    return x


# ============================================================
# SELECCIONAR FEATURES ÚTILES
# ============================================================

def seleccionar_features(features):

    excluded = {
        "datetime",
        "nivel",
        "precip_mm",
        "caudal_m3s",
    }

    candidatos = [
        col
        for col in features.columns
        if col not in excluded
    ]

    feature_cols = []

    for col in candidatos:

        numeric = pd.to_numeric(
            features[col],
            errors="coerce",
        )

        # Debe tener suficientes datos
        if numeric.notna().sum() < 15:
            continue

        # Evitar columnas constantes
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
# ENTRENAR MODELO
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
            "No se pudieron construir variables "
            "válidas para entrenar el modelo."
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
            "después de construir las variables. "
            f"Disponibles: {len(work)}."
        )

    # ========================================================
    # VALIDACIÓN TEMPORAL 80 / 20
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
            "No quedaron observaciones "
            "para validar el modelo."
        )

    # ========================================================
    # MODELO DE VALIDACIÓN
    # ========================================================

    validation_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    validation_model.fit(
        X_train,
        y_train,
    )

    validation_pred = (
        validation_model.predict(
            X_test
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                validation_pred,
            )
        )
    )

    # Piso mínimo para no mostrar
    # incertidumbre irreal de cero
    rmse = max(
        rmse,
        0.05,
    )

    # ========================================================
    # MODELO FINAL
    # ========================================================

    final_model = RandomForestRegressor(
        n_estimators=700,
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
            "nivel"
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

    models = {
        "model":
            final_model,

        "feature_cols":
            feature_cols,

        "rmse":
            rmse,

        "dataset":
            dataset.copy(),

        "importance":
            importance,

        "uses_rain":
            any(
                col.startswith("rain_")
                for col in feature_cols
            ),

        "uses_caudal":
            any(
                col.startswith("q_")
                for col in feature_cols
            ),

        "uses_upstream":
            any(
                col.startswith("nivel_")
                for col in feature_cols
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
    }

    return (
        models,
        metrics,
    )


# ============================================================
# PREPARAR FUTURO EXÓGENO
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
        or "datetime" not in exog_future.columns
    ):

        return pd.DataFrame()

    future = exog_future.copy()

    future["datetime"] = pd.to_datetime(
        future["datetime"],
        errors="coerce",
    ).dt.normalize()

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
        .reset_index(
            drop=True
        )
    )

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
# CREAR FILA PARA EL MODELO
# ============================================================

def construir_X_futura(
    history,
    feature_cols,
):

    temp = crear_features(
        history
    )

    if temp.empty:

        raise ValueError(
            "No se pudieron construir "
            "variables para el pronóstico."
        )

    latest = temp.iloc[-1]

    values = {}

    for col in feature_cols:

        value = latest.get(
            col,
            np.nan,
        )

        value = pd.to_numeric(
            pd.Series([value]),
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
# PRONÓSTICO RECURSIVO
# ============================================================

def predict(
    df,
    models,
    days=15,
    exog_future=None,
):

    if (
        models is None
        or not isinstance(models, dict)
        or "model" not in models
    ):

        raise ValueError(
            "El modelo no está entrenado."
        )

    days = int(days)

    days = max(
        1,
        min(
            days,
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

    history = models[
        "dataset"
    ].copy()

    history["datetime"] = pd.to_datetime(
        history["datetime"],
        errors="coerce",
    ).dt.normalize()

    history = (
        history
        .dropna(
            subset=[
                "datetime",
                "nivel",
            ]
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    if history.empty:

        raise ValueError(
            "No existe historial para generar "
            "el pronóstico."
        )

    future = preparar_exog_future(
        exog_future
    )

    last_real_date = history[
        "datetime"
    ].iloc[-1]

    output = []

    # ========================================================
    # VARIABLES AGUAS ARRIBA
    # ========================================================

    upstream_cols = [
        col
        for col in history.columns
        if col.startswith("nivel_")
        and col != "nivel"
        and "_lag" not in col
        and "_diff" not in col
        and "_trend" not in col
        and "_media" not in col
        and "_actual" not in col
    ]

    # ========================================================
    # LOOP DÍA POR DÍA
    # ========================================================

    for step in range(
        1,
        days + 1,
    ):

        target_date = (
            last_real_date
            + pd.Timedelta(
                days=step
            )
        )

        # ----------------------------------------------------
        # LLUVIA FUTURA
        # ----------------------------------------------------

        precip = 0.0
        caudal = np.nan

        if not future.empty:

            match = future[
                future["datetime"]
                == target_date.normalize()
            ]

            if not match.empty:

                if "precip_mm" in match.columns:

                    value = pd.to_numeric(
                        match[
                            "precip_mm"
                        ],
                        errors="coerce",
                    )

                    if value.notna().any():

                        precip = float(
                            value
                            .dropna()
                            .iloc[0]
                        )

                if "caudal_m3s" in match.columns:

                    value = pd.to_numeric(
                        match[
                            "caudal_m3s"
                        ],
                        errors="coerce",
                    )

                    if value.notna().any():

                        caudal = float(
                            value
                            .dropna()
                            .iloc[0]
                        )

        # ----------------------------------------------------
        # SI NO HAY CAUDAL FUTURO
        # CONSERVAR ÚLTIMO DISPONIBLE
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

        # ----------------------------------------------------
        # FILA FUTURA
        # ----------------------------------------------------

        row = {
            "datetime":
                target_date,

            # valor provisorio para construir lags
            "nivel":
                float(
                    history[
                        "nivel"
                    ].iloc[-1]
                ),

            "precip_mm":
                float(
                    precip
                ),

            "caudal_m3s":
                caudal,
        }

        # ----------------------------------------------------
        # AGUAS ARRIBA:
        # MANTENER ÚLTIMO NIVEL DISPONIBLE
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

        # ----------------------------------------------------
        # CONSTRUIR X
        # ----------------------------------------------------

        X_future = construir_X_futura(
            trial,
            feature_cols,
        )

        prediction = float(
            model.predict(
                X_future
            )[0]
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

        # La incertidumbre aumenta suavemente
        # con el horizonte.
        horizon_factor = (
            1.0
            + 0.035
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

        previous_level = float(
            history[
                "nivel"
            ].iloc[-1]
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
                    - previous_level,

                "precip_mm":
                    precip,

                "caudal_m3s":
                    caudal,
            }
        )

        # ----------------------------------------------------
        # AGREGAR PREDICCIÓN AL HISTORIAL
        # PARA EL SIGUIENTE DÍA
        # ----------------------------------------------------

        row["nivel"] = prediction

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [row]
                ),
            ],
            ignore_index=True,
        )

        history = (
            history
            .sort_values(
                "datetime"
            )
            .reset_index(
                drop=True
            )
        )

    return pd.DataFrame(
        output
    )
