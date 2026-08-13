import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ============================================================
# CONFIGURACIÓN
# ============================================================

MIN_OBSERVATIONS = 20

DEFAULT_FORECAST_DAYS = 15

RANDOM_STATE = 42


# ============================================================
# PREPARAR SERIE
# ============================================================

def preparar_serie(df):
    """
    Recibe el DataFrame proveniente del INA.

    Columnas esperadas:
        datetime
        nivel

    También admite:
        datetime
        value
    """

    if df is None:
        raise ValueError(
            "No se recibieron datos para el modelo."
        )

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "Los datos del modelo deben ser un DataFrame."
        )

    if df.empty:
        raise ValueError(
            "No hay observaciones disponibles para el modelo."
        )

    work = df.copy()

    # ========================================================
    # FECHA
    # ========================================================

    if "datetime" not in work.columns:
        raise ValueError(
            "No se encontró la columna datetime."
        )

    work["datetime"] = pd.to_datetime(
        work["datetime"],
        errors="coerce",
        utc=True,
    )

    # ========================================================
    # NIVEL
    # ========================================================

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
            "No se encontró la columna nivel o value."
        )

    # ========================================================
    # LIMPIEZA
    # ========================================================

    work = work.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    work = (
        work
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if len(work) < MIN_OBSERVATIONS:

        raise ValueError(
            f"No hay suficientes observaciones para entrenar "
            f"el modelo: {len(work)}. "
            f"Se requieren al menos {MIN_OBSERVATIONS}."
        )

    return work[
        [
            "datetime",
            "nivel",
        ]
    ]


# ============================================================
# DEFINIR LAGS
# ============================================================

def seleccionar_lags(n):
    """
    Selecciona retardos según la cantidad de datos disponibles.
    """

    lags = [
        1,
        2,
        3,
    ]

    if n >= 15:
        lags.append(7)

    if n >= 30:
        lags.append(14)

    return lags


# ============================================================
# CREAR FEATURES
# ============================================================

def crear_features(df, lags):
    """
    Construye variables temporales para Random Forest.
    """

    x = df.copy()

    # ========================================================
    # LAGS
    # ========================================================

    for lag in lags:

        x[
            f"lag_{lag}"
        ] = x[
            "nivel"
        ].shift(lag)

    # ========================================================
    # DIFERENCIAS
    # ========================================================

    x[
        "diff_1"
    ] = x[
        "nivel"
    ].diff(1)

    x[
        "diff_2"
    ] = x[
        "nivel"
    ].diff(2)

    # ========================================================
    # PROMEDIOS MÓVILES
    # ========================================================

    x[
        "media_3"
    ] = (
        x["nivel"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    if len(x) >= 10:

        x[
            "media_7"
        ] = (
            x["nivel"]
            .shift(1)
            .rolling(7)
            .mean()
        )

    # ========================================================
    # TENDENCIA RECIENTE
    # ========================================================

    x[
        "trend_3"
    ] = (
        x["nivel"]
        .shift(1)
        - x["nivel"].shift(4)
    ) / 3.0

    return x


# ============================================================
# ENTRENAR
# ============================================================

def train(df):
    """
    Entrena el modelo con la serie histórica de San Nicolás.

    Retorna:
        models
        metrics
    """

    serie = preparar_serie(
        df
    )

    lags = seleccionar_lags(
        len(serie)
    )

    features_df = crear_features(
        serie,
        lags,
    )

    # ========================================================
    # VARIABLES DEL MODELO
    # ========================================================

    feature_cols = [
        c
        for c in features_df.columns
        if c not in [
            "datetime",
            "nivel",
        ]
    ]

    work = features_df.dropna(
        subset=[
            "nivel",
        ] + feature_cols
    )

    if len(work) < 10:

        raise ValueError(
            "Después de generar los retardos quedan "
            f"solamente {len(work)} observaciones válidas. "
            "Amplíe el período histórico."
        )

    # ========================================================
    # SPLIT CRONOLÓGICO
    # ========================================================

    split = int(
        len(work) * 0.80
    )

    # Garantizar datos de entrenamiento y prueba
    split = max(
        5,
        split,
    )

    split = min(
        split,
        len(work) - 1,
    )

    X_train = work[
        feature_cols
    ].iloc[:split]

    X_test = work[
        feature_cols
    ].iloc[split:]

    y_train = work[
        "nivel"
    ].iloc[:split]

    y_test = work[
        "nivel"
    ].iloc[split:]

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # ========================================================
    # VALIDACIÓN
    # ========================================================

    if len(X_test) > 0:

        prediction_test = model.predict(
            X_test
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    prediction_test,
                )
            )
        )

    else:

        # Valor conservador de respaldo
        rmse = 0.15

    # Evitar incertidumbre cero
    rmse = max(
        rmse,
        0.05,
    )

    # ========================================================
    # REENTRENAR CON TODO EL HISTÓRICO
    # ========================================================

    X_all = work[
        feature_cols
    ]

    y_all = work[
        "nivel"
    ]

    final_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    final_model.fit(
        X_all,
        y_all,
    )

    # ========================================================
    # PAQUETE DEL MODELO
    # ========================================================

    models = {
        "model": final_model,
        "feature_cols": feature_cols,
        "lags": lags,
        "rmse": rmse,
        "observations": len(serie),
        "training_rows": len(work),
        "last_datetime": serie[
            "datetime"
        ].iloc[-1],
        "last_level": float(
            serie[
                "nivel"
            ].iloc[-1]
        ),
    }

    metrics = {
        "RMSE": rmse,
        "observations": len(serie),
        "training_rows": len(work),
        "test_rows": len(X_test),
    }

    return (
        models,
        metrics,
    )


# ============================================================
# CREAR FILA PARA PRONÓSTICO RECURSIVO
# ============================================================

def crear_fila_futura(
    history,
    feature_cols,
    lags,
):
    """
    Construye las variables para pronosticar el próximo día.
    """

    niveles = history[
        "nivel"
    ].to_numpy()

    data = {}

    # ========================================================
    # LAGS
    # ========================================================

    for lag in lags:

        if len(niveles) >= lag:

            data[
                f"lag_{lag}"
            ] = niveles[
                -lag
            ]

        else:

            data[
                f"lag_{lag}"
            ] = niveles[
                0
            ]

    # ========================================================
    # DIFERENCIAS
    # ========================================================

    if len(niveles) >= 2:

        data[
            "diff_1"
        ] = (
            niveles[-1]
            - niveles[-2]
        )

    else:

        data[
            "diff_1"
        ] = 0.0

    if len(niveles) >= 3:

        data[
            "diff_2"
        ] = (
            niveles[-1]
            - niveles[-3]
        )

    else:

        data[
            "diff_2"
        ] = 0.0

    # ========================================================
    # MEDIA 3
    # ========================================================

    if len(niveles) >= 3:

        data[
            "media_3"
        ] = float(
            np.mean(
                niveles[-3:]
            )
        )

    else:

        data[
            "media_3"
        ] = float(
            np.mean(
                niveles
            )
        )

    # ========================================================
    # MEDIA 7
    # ========================================================

    if "media_7" in feature_cols:

        if len(niveles) >= 7:

            data[
                "media_7"
            ] = float(
                np.mean(
                    niveles[-7:]
                )
            )

        else:

            data[
                "media_7"
            ] = float(
                np.mean(
                    niveles
                )
            )

    # ========================================================
    # TENDENCIA 3
    # ========================================================

    if len(niveles) >= 4:

        data[
            "trend_3"
        ] = float(
            (
                niveles[-1]
                - niveles[-4]
            ) / 3.0
        )

    else:

        data[
            "trend_3"
        ] = 0.0

    # ========================================================
    # GARANTIZAR TODAS LAS COLUMNAS
    # ========================================================

    row = {}

    for columna in feature_cols:

        row[
            columna
        ] = data.get(
            columna,
            0.0,
        )

    return pd.DataFrame(
        [
            row
        ]
    )


# ============================================================
# PRONOSTICAR
# ============================================================

def predict(
    df,
    models,
    days=DEFAULT_FORECAST_DAYS,
):
    """
    Genera pronóstico diario recursivo.

    Retorna DataFrame:
        datetime
        prediction
        lower
        upper
    """

    if days < 1:

        raise ValueError(
            "El horizonte debe ser al menos 1 día."
        )

    serie = preparar_serie(
        df
    )

    model = models[
        "model"
    ]

    feature_cols = models[
        "feature_cols"
    ]

    lags = models[
        "lags"
    ]

    rmse = float(
        models.get(
            "rmse",
            0.15,
        )
    )

    history = serie.copy()

    resultados = []

    ultima_fecha = history[
        "datetime"
    ].iloc[-1]

    # ========================================================
    # PRONÓSTICO RECURSIVO
    # ========================================================

    for horizonte in range(
        1,
        days + 1,
    ):

        X_future = crear_fila_futura(
            history,
            feature_cols,
            lags,
        )

        prediction = float(
            model.predict(
                X_future
            )[0]
        )

        fecha_futura = (
            ultima_fecha
            + pd.Timedelta(
                days=horizonte
            )
        )

        # ----------------------------------------------------
        # INCERTIDUMBRE
        # ----------------------------------------------------

        # Aumenta gradualmente con el horizonte
        sigma = (
            rmse
            * np.sqrt(
                horizonte
            )
        )

        lower = float(
            prediction
            - 1.96 * sigma
        )

        upper = float(
            prediction
            + 1.96 * sigma
        )

        resultados.append(
            {
                "datetime": fecha_futura,
                "prediction": prediction,
                "lower": lower,
                "upper": upper,
                "horizon_day": horizonte,
            }
        )

        # ----------------------------------------------------
        # AGREGAR PREDICCIÓN AL HISTÓRICO
        # ----------------------------------------------------

        nueva_fila = pd.DataFrame(
            {
                "datetime": [
                    fecha_futura
                ],
                "nivel": [
                    prediction
                ],
            }
        )

        history = pd.concat(
            [
                history,
                nueva_fila,
            ],
            ignore_index=True,
        )

    return pd.DataFrame(
        resultados
    )


# ============================================================
# PROBABILIDAD DE SUPERAR UMBRAL
# ============================================================

def prob(
    prediction,
    threshold,
    rmse,
):
    """
    Estimación experimental de probabilidad de superar
    determinado nivel.
    """

    sigma = max(
        float(rmse),
        0.05,
    )

    z = (
        float(prediction)
        - float(threshold)
    ) / sigma

    # Evitar overflow numérico
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
