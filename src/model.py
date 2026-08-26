# ============================================================
# model.py
# PARANÁ · SAN NICOLÁS V12.0
# MODELO DE PROPAGACIÓN HIDROLÓGICA
# ============================================================

import re
import unicodedata

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ============================================================
# CONFIGURACIÓN
# ============================================================

RANDOM_STATE = 42

LEVEL_MIN = 0.0
LEVEL_MAX = 7.0

MIN_OBSERVATIONS = 60

MAX_FORECAST_DAYS = 30

MAX_RESPONSE_LAG = 30


STATION_ORDER = [
    "Corrientes",
    "Goya",
    "La Paz",
    "Paraná",
    "Diamante",
    "Rosario",
    "Villa Constitución",
    "San Nicolás",
]


LOCAL_LAGS = [
    1,
    2,
    3,
    5,
    7,
    10,
    14,
    21,
    30,
]


UPSTREAM_LAGS = [
    1,
    2,
    3,
    5,
    7,
    10,
    14,
    21,
    30,
]


Q_LAGS = [
    1,
    2,
    3,
    5,
    7,
    10,
    14,
    21,
    30,
]


RAIN_WINDOWS = [
    1,
    3,
    7,
    14,
]


# ============================================================
# UTILIDADES
# ============================================================

def _slug(text):

    text = str(text).strip().lower()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        c
        for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_")


STATION_SLUGS = {
    station: _slug(station)
    for station in STATION_ORDER
}


def _normalizar_datetime(serie):

    dt = pd.to_datetime(
        serie,
        errors="coerce",
        utc=True,
    )

    return (
        dt
        .dt
        .tz_localize(None)
        .dt
        .normalize()
    )


def _numeric(serie):

    return pd.to_numeric(
        serie,
        errors="coerce",
    )


def _safe_last(serie):

    s = _numeric(
        serie
    ).dropna()

    if s.empty:

        return np.nan

    return float(
        s.iloc[-1]
    )


def _safe_previous(serie):

    s = _numeric(
        serie
    ).dropna()

    if len(s) < 2:

        return np.nan

    return float(
        s.iloc[-2]
    )


def _safe_slope(
    serie,
    window=5,
):

    s = (
        _numeric(
            serie
        )
        .dropna()
        .tail(window)
    )

    if len(s) < 3:

        return 0.0

    x = np.arange(
        len(s),
        dtype=float,
    )

    try:

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


def _clip_level(value):

    return float(
        np.clip(
            value,
            LEVEL_MIN,
            LEVEL_MAX,
        )
    )


# ============================================================
# DETECTAR COLUMNAS
# ============================================================

def _base_level_columns(df):

    result = []

    excluded = [
        "_lag",
        "_diff",
        "_trend",
        "_mean",
        "_actual",
        "_delta",
        "_next",
    ]

    for column in df.columns:

        name = str(column)

        if not name.startswith(
            "nivel_"
        ):
            continue

        if any(
            token in name
            for token in excluded
        ):
            continue

        result.append(
            column
        )

    return result


def _flow_columns(df):

    result = []

    for column in df.columns:

        name = str(
            column
        ).lower()

        if (
            name == "caudal_m3s"
            or "caudal" in name
            or name.startswith("q_")
        ):

            result.append(
                column
            )

    return list(
        dict.fromkeys(
            result
        )
    )


def _rain_columns(df):

    result = []

    for column in df.columns:

        name = str(
            column
        ).lower()

        if (
            name == "precip_mm"
            or "precip" in name
            or "lluvia" in name
            or name.startswith("rain_")
        ):

            result.append(
                column
            )

    return list(
        dict.fromkeys(
            result
        )
    )


def _station_from_column(column):

    if not str(column).startswith(
        "nivel_"
    ):

        return None

    slug = str(column)[
        len("nivel_"):
    ]

    for station, station_slug in STATION_SLUGS.items():

        if slug == station_slug:

            return station

    return slug.replace(
        "_",
        " ",
    ).title()


# ============================================================
# PREPARAR SAN NICOLÁS
# ============================================================

def preparar_nivel_local(
    df,
    fecha_base=None,
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
            "Falta datetime."
        )

    if "nivel" in x.columns:

        x["nivel"] = _numeric(
            x["nivel"]
        )

    elif "value" in x.columns:

        x["nivel"] = _numeric(
            x["value"]
        )

    else:

        raise ValueError(
            "Falta nivel/value."
        )

    x["datetime"] = _normalizar_datetime(
        x["datetime"]
    )

    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    if fecha_base is not None:

        cutoff = pd.to_datetime(
            fecha_base,
            errors="coerce",
        )

        if pd.isna(cutoff):

            raise ValueError(
                "Fecha base no válida."
            )

        cutoff = cutoff.normalize()

        x = x[
            x["datetime"] <= cutoff
        ]

    if x.empty:

        raise ValueError(
            "No existen mediciones de San Nicolás "
            "hasta la fecha seleccionada."
        )

    x = (
        x.groupby(
            "datetime",
            as_index=False,
        )["nivel"]
        .mean()
        .sort_values(
            "datetime"
        )
        .reset_index(drop=True)
    )

    return x


# ============================================================
# NORMALIZAR DATOS EXTERNOS
# ============================================================

def _normalizar_externo(
    df,
    fecha_base=None,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):

        return pd.DataFrame()

    x = df.copy()

    if "datetime" not in x.columns:

        return pd.DataFrame()

    x["datetime"] = _normalizar_datetime(
        x["datetime"]
    )

    x = x.dropna(
        subset=[
            "datetime"
        ]
    )

    if fecha_base is not None:

        cutoff = pd.to_datetime(
            fecha_base,
            errors="coerce",
        )

        if not pd.isna(cutoff):

            cutoff = cutoff.normalize()

            x = x[
                x["datetime"] <= cutoff
            ]

    for column in x.columns:

        if column == "datetime":

            continue

        converted = pd.to_numeric(
            x[column],
            errors="coerce",
        )

        if converted.notna().any():

            x[column] = converted

    numeric_cols = [
        c
        for c in x.columns
        if c != "datetime"
        and pd.api.types.is_numeric_dtype(
            x[c]
        )
    ]

    if numeric_cols:

        x = (
            x.groupby(
                "datetime",
                as_index=False,
            )[numeric_cols]
            .mean()
        )

    return (
        x.sort_values(
            "datetime"
        )
        .reset_index(drop=True)
    )


# ============================================================
# DATASET GENERAL
# ============================================================

def preparar_dataset(
    df,
    exog_history=None,
    upstream_history=None,
    fecha_base=None,
):

    local = preparar_nivel_local(
        df,
        fecha_base=fecha_base,
    )

    result = local.copy()

    exog = _normalizar_externo(
        exog_history,
        fecha_base=fecha_base,
    )

    upstream = _normalizar_externo(
        upstream_history,
        fecha_base=fecha_base,
    )

    if not exog.empty:

        result = result.merge(
            exog,
            on="datetime",
            how="outer",
        )

    if not upstream.empty:

        repeated = [
            c
            for c in upstream.columns
            if (
                c != "datetime"
                and c in result.columns
            )
        ]

        if repeated:

            upstream = upstream.drop(
                columns=repeated,
                errors="ignore",
            )

        result = result.merge(
            upstream,
            on="datetime",
            how="outer",
        )

    result = result.sort_values(
        "datetime"
    )

    start = result[
        "datetime"
    ].min()

    end = result[
        "datetime"
    ].max()

    calendar = pd.DataFrame(
        {
            "datetime":
                pd.date_range(
                    start,
                    end,
                    freq="D",
                )
        }
    )

    result = calendar.merge(
        result,
        on="datetime",
        how="left",
    )

    for column in result.columns:

        if column == "datetime":

            continue

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # San Nicolás:
    # sólo interpolación dentro del período real.
    result["nivel"] = (
        result["nivel"]
        .interpolate(
            limit_area="inside",
        )
    )

    # Estaciones aguas arriba.
    for column in _base_level_columns(
        result
    ):

        result[column] = (
            result[column]
            .interpolate(
                limit=3,
                limit_area="inside",
            )
        )

    # Caudal:
    # sólo huecos cortos.
    for column in _flow_columns(
        result
    ):

        result[column] = (
            result[column]
            .interpolate(
                limit=2,
                limit_area="inside",
            )
        )

    # IMPORTANTE:
    # lluvia faltante NO se transforma en cero.

    return (
        result.sort_values(
            "datetime"
        )
        .reset_index(drop=True)
    )


# ============================================================
# ESTADO ACTUAL DE ESTACIONES
# ============================================================

def resumen_niveles_estaciones(
    upstream_history,
    df_local=None,
    fecha_base=None,
):

    rows = []

    upstream = _normalizar_externo(
        upstream_history,
        fecha_base=fecha_base,
    )

    if not upstream.empty:

        for column in _base_level_columns(
            upstream
        ):

            station = _station_from_column(
                column
            )

            tmp = upstream[
                [
                    "datetime",
                    column,
                ]
            ].dropna()

            if tmp.empty:

                continue

            tmp = tmp.sort_values(
                "datetime"
            )

            current = float(
                tmp[column].iloc[-1]
            )

            previous = (
                float(
                    tmp[column].iloc[-2]
                )
                if len(tmp) >= 2
                else np.nan
            )

            delta = (
                current - previous
                if np.isfinite(previous)
                else np.nan
            )

            if not np.isfinite(delta):

                trend = "Sin comparación"

            elif delta > 0.01:

                trend = "↑ Creciendo"

            elif delta < -0.01:

                trend = "↓ Bajando"

            else:

                trend = "→ Estable"

            rows.append(
                {
                    "Estación": station,
                    "Nivel actual": current,
                    "Nivel anterior": previous,
                    "Variación": delta,
                    "Tendencia": trend,
                    "Fecha":
                        tmp[
                            "datetime"
                        ].iloc[-1],
                }
            )

    if (
        df_local is not None
        and isinstance(
            df_local,
            pd.DataFrame,
        )
        and not df_local.empty
    ):

        local = preparar_nivel_local(
            df_local,
            fecha_base=fecha_base,
        )

        current = _safe_last(
            local["nivel"]
        )

        previous = _safe_previous(
            local["nivel"]
        )

        delta = (
            current - previous
            if np.isfinite(previous)
            else np.nan
        )

        if not np.isfinite(delta):

            trend = "Sin comparación"

        elif delta > 0.01:

            trend = "↑ Creciendo"

        elif delta < -0.01:

            trend = "↓ Bajando"

        else:

            trend = "→ Estable"

        rows.append(
            {
                "Estación": "San Nicolás",
                "Nivel actual": current,
                "Nivel anterior": previous,
                "Variación": delta,
                "Tendencia": trend,
                "Fecha":
                    local[
                        "datetime"
                    ].iloc[-1],
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    order = {
        station: i
        for i, station in enumerate(
            STATION_ORDER
        )
    }

    result["_order"] = (
        result[
            "Estación"
        ]
        .map(order)
        .fillna(999)
    )

    return (
        result.sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(drop=True)
    )


# ============================================================
# ANÁLISIS DE RESPUESTA TEMPORAL
# ============================================================

def _lagged_response(
    driver,
    response,
    max_lag=30,
    only_positive=False,
    min_pairs=30,
):

    driver = _numeric(
        driver
    )

    response = _numeric(
        response
    )

    best = None

    for lag in range(
        max_lag + 1
    ):

        future_response = response.shift(
            -lag
        )

        work = pd.DataFrame(
            {
                "driver": driver,
                "response":
                    future_response,
            }
        ).dropna()

        if only_positive:

            work = work[
                work["driver"] > 0
            ]

        if len(work) < min_pairs:

            continue

        if (
            work["driver"].std() == 0
            or work["response"].std() == 0
        ):

            continue

        corr = float(
            work[
                "driver"
            ].corr(
                work[
                    "response"
                ]
            )
        )

        try:

            slope, intercept = np.polyfit(
                work[
                    "driver"
                ],
                work[
                    "response"
                ],
                1,
            )

            predicted = (
                slope
                * work["driver"]
                + intercept
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        (
                            work[
                                "response"
                            ]
                            - predicted
                        ) ** 2
                    )
                )
            )

        except Exception:

            slope = np.nan
            intercept = np.nan
            rmse = np.nan

        candidate = {
            "lag": lag,
            "correlation": corr,
            "slope": float(slope),
            "intercept":
                float(intercept),
            "pairs":
                int(len(work)),
            "rmse": rmse,
        }

        if best is None:

            best = candidate

        elif candidate[
            "correlation"
        ] > best[
            "correlation"
        ]:

            best = candidate

    return best


# ============================================================
# NIVEL DE CADA ESTACIÓN VS CAUDAL
# ============================================================

def analizar_respuesta_nivel_a_caudal(
    dataset,
    max_lag=30,
):

    if (
        dataset is None
        or dataset.empty
    ):

        return pd.DataFrame()

    x = dataset.copy()

    level_map = {
        "San Nicolás":
            "nivel"
    }

    for column in _base_level_columns(
        x
    ):

        station = _station_from_column(
            column
        )

        level_map[
            station
        ] = column

    flow_cols = _flow_columns(
        x
    )

    if not flow_cols:

        return pd.DataFrame()

    rows = []

    for station in STATION_ORDER:

        level_col = level_map.get(
            station
        )

        if (
            level_col is None
            or level_col not in x.columns
        ):

            continue

        station_slug = STATION_SLUGS[
            station
        ]

        station_flows = [
            c
            for c in flow_cols
            if station_slug in _slug(c)
        ]

        if not station_flows:

            station_flows = flow_cols

        best = None

        delta_level = _numeric(
            x[level_col]
        ).diff()

        for flow_col in station_flows:

            delta_flow = _numeric(
                x[flow_col]
            ).diff()

            result = _lagged_response(
                delta_flow,
                delta_level,
                max_lag=max_lag,
                only_positive=True,
            )

            if result is None:

                continue

            candidate = {
                "Estación":
                    station,

                "Caudal utilizado":
                    flow_col,

                "Desfase días":
                    result["lag"],

                "Correlación":
                    result[
                        "correlation"
                    ],

                "Respuesta nivel/caudal":
                    result[
                        "slope"
                    ],

                "Pares históricos":
                    result[
                        "pairs"
                    ],

                "RMSE":
                    result[
                        "rmse"
                    ],
            }

            if best is None:

                best = candidate

            elif candidate[
                "Correlación"
            ] > best[
                "Correlación"
            ]:

                best = candidate

        if best:

            rows.append(
                best
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PROPAGACIÓN DE NIVEL ENTRE ESTACIONES
# ============================================================

def analizar_propagacion_niveles(
    dataset,
    max_lag=30,
):

    if (
        dataset is None
        or dataset.empty
    ):

        return pd.DataFrame()

    x = dataset.copy()

    level_map = {
        "San Nicolás":
            "nivel"
    }

    for column in _base_level_columns(
        x
    ):

        station = _station_from_column(
            column
        )

        level_map[
            station
        ] = column

    pairs = []

    for i in range(
        len(STATION_ORDER) - 1
    ):

        pairs.append(
            (
                STATION_ORDER[i],
                STATION_ORDER[
                    i + 1
                ],
            )
        )

    # Relación especial fundamental.
    pairs.append(
        (
            "Corrientes",
            "San Nicolás",
        )
    )

    rows = []

    for origin, destination in pairs:

        origin_col = level_map.get(
            origin
        )

        destination_col = level_map.get(
            destination
        )

        if (
            origin_col is None
            or destination_col is None
        ):

            continue

        delta_origin = _numeric(
            x[origin_col]
        ).diff()

        delta_destination = _numeric(
            x[destination_col]
        ).diff()

        result = _lagged_response(
            delta_origin,
            delta_destination,
            max_lag=max_lag,
            only_positive=False,
        )

        if result is None:

            continue

        rows.append(
            {
                "Origen":
                    origin,

                "Destino":
                    destination,

                "Desfase días":
                    result[
                        "lag"
                    ],

                "Correlación":
                    result[
                        "correlation"
                    ],

                "Respuesta destino/origen":
                    result[
                        "slope"
                    ],

                "Pares históricos":
                    result[
                        "pairs"
                    ],

                "RMSE":
                    result[
                        "rmse"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CORRIENTES -> SAN NICOLÁS
# ============================================================

def relacion_corrientes_san_nicolas(
    dataset,
):

    table = analizar_propagacion_niveles(
        dataset
    )

    if table.empty:

        return {}

    row = table[
        (
            table[
                "Origen"
            ]
            == "Corrientes"
        )
        &
        (
            table[
                "Destino"
            ]
            == "San Nicolás"
        )
    ]

    if row.empty:

        return {}

    return row.iloc[
        0
    ].to_dict()


# ============================================================
# CREAR FEATURES
# ============================================================

def crear_features(
    dataset,
):

    x = (
        dataset
        .copy()
        .sort_values(
            "datetime"
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # SAN NICOLÁS
    # ========================================================

    x[
        "nivel_actual"
    ] = x["nivel"]

    x[
        "nivel_diff1"
    ] = x[
        "nivel"
    ].diff()

    x[
        "nivel_trend3"
    ] = (
        x["nivel"]
        - x["nivel"].shift(3)
    ) / 3.0

    x[
        "nivel_trend7"
    ] = (
        x["nivel"]
        - x["nivel"].shift(7)
    ) / 7.0

    for lag in LOCAL_LAGS:

        x[
            f"nivel_lag{lag}"
        ] = x[
            "nivel"
        ].shift(
            lag
        )

    # ========================================================
    # NIVELES AGUAS ARRIBA
    # ========================================================

    for column in _base_level_columns(
        x
    ):

        x[
            f"{column}_actual"
        ] = x[column]

        x[
            f"{column}_diff1"
        ] = x[
            column
        ].diff()

        x[
            f"{column}_trend3"
        ] = (
            x[column]
            - x[column].shift(3)
        ) / 3.0

        x[
            f"{column}_trend7"
        ] = (
            x[column]
            - x[column].shift(7)
        ) / 7.0

        x[
            f"{column}_mean7"
        ] = (
            x[column]
            .rolling(7)
            .mean()
        )

        for lag in UPSTREAM_LAGS:

            x[
                f"{column}_lag{lag}"
            ] = x[
                column
            ].shift(
                lag
            )

    # ========================================================
    # CAUDAL
    # ========================================================

    for column in _flow_columns(
        x
    ):

        q = _numeric(
            x[column]
        )

        x[
            f"{column}_actual"
        ] = q

        x[
            f"{column}_diff1"
        ] = q.diff()

        x[
            f"{column}_trend3"
        ] = (
            q
            - q.shift(3)
        ) / 3.0

        x[
            f"{column}_trend7"
        ] = (
            q
            - q.shift(7)
        ) / 7.0

        x[
            f"{column}_mean7"
        ] = (
            q
            .rolling(7)
            .mean()
        )

        for lag in Q_LAGS:

            x[
                f"{column}_lag{lag}"
            ] = q.shift(
                lag
            )

    # ========================================================
    # LLUVIA
    # ========================================================

    for column in _rain_columns(
        x
    ):

        rain = _numeric(
            x[column]
        )

        for window in RAIN_WINDOWS:

            x[
                f"{column}_sum{window}"
            ] = (
                rain
                .rolling(
                    window,
                    min_periods=max(
                        1,
                        int(
                            window * 0.7
                        ),
                    ),
                )
                .sum()
            )

        x[
            f"{column}_lag1"
        ] = rain.shift(1)

        x[
            f"{column}_lag3"
        ] = rain.shift(3)

        x[
            f"{column}_lag7"
        ] = rain.shift(7)

    # ========================================================
    # ESTACIONALIDAD
    # ========================================================

    doy = x[
        "datetime"
    ].dt.dayofyear

    x[
        "sin_doy"
    ] = np.sin(
        2
        * np.pi
        * doy
        / 365.25
    )

    x[
        "cos_doy"
    ] = np.cos(
        2
        * np.pi
        * doy
        / 365.25
    )

    # ========================================================
    # OBJETIVO
    # ========================================================

    x[
        "target_nivel"
    ] = x[
        "nivel"
    ].shift(-1)

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
# FEATURES UTILIZABLES
# ============================================================

def _seleccionar_features(
    features,
):

    excluded = {
        "datetime",
        "nivel",
        "target_nivel",
        "target_delta",
    }

    excluded.update(
        _base_level_columns(
            features
        )
    )

    excluded.update(
        _flow_columns(
            features
        )
    )

    excluded.update(
        _rain_columns(
            features
        )
    )

    feature_cols = [
        c
        for c in features.columns
        if (
            c not in excluded
            and pd.api.types
            .is_numeric_dtype(
                features[c]
            )
        )
    ]

    feature_cols = [
        c
        for c in feature_cols
        if (
            features[c]
            .notna()
            .sum()
            >= 20
        )
    ]

    feature_cols = [
        c
        for c in feature_cols
        if (
            features[c]
            .nunique(
                dropna=True
            )
            > 1
        )
    ]

    return feature_cols


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train(
    df,
    exog_history=None,
    upstream_history=None,
    fecha_base=None,
):

    dataset = preparar_dataset(
        df=df,
        exog_history=exog_history,
        upstream_history=upstream_history,
        fecha_base=fecha_base,
    )

    features = crear_features(
        dataset
    )

    feature_cols = _seleccionar_features(
        features
    )

    if not feature_cols:

        raise ValueError(
            "No hay variables suficientes "
            "para entrenar el modelo."
        )

    work = (
        features[
            [
                "datetime",
                "nivel",
                "target_delta",
            ]
            + feature_cols
        ]
        .dropna(
            subset=[
                "nivel",
                "target_delta",
            ]
            + feature_cols
        )
        .copy()
    )

    if len(work) < MIN_OBSERVATIONS:

        raise ValueError(
            "No hay suficientes registros "
            "históricos utilizables. "
            f"Disponibles: {len(work)}"
        )

    # ========================================================
    # VALIDACIÓN CRONOLÓGICA
    # ========================================================

    split = int(
        len(work)
        * 0.80
    )

    split = max(
        40,
        split,
    )

    split = min(
        split,
        len(work) - 1,
    )

    train_data = work.iloc[
        :split
    ]

    test_data = work.iloc[
        split:
    ]

    X_train = train_data[
        feature_cols
    ]

    y_train = train_data[
        "target_delta"
    ]

    X_test = test_data[
        feature_cols
    ]

    y_test = test_data[
        "target_delta"
    ]

    # ========================================================
    # MODELO VALIDACIÓN
    # ========================================================

    validation_model = (
        RandomForestRegressor(
            n_estimators=800,
            max_depth=14,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    validation_model.fit(
        X_train,
        y_train,
    )

    pred_delta = (
        validation_model
        .predict(
            X_test
        )
    )

    base_test = test_data[
        "nivel"
    ].to_numpy(
        dtype=float
    )

    true_level = (
        base_test
        + y_test.to_numpy(
            dtype=float
        )
    )

    pred_level = (
        base_test
        + pred_delta
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                true_level,
                pred_level,
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
            n_estimators=1200,
            max_depth=14,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
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

    # ========================================================
    # ANÁLISIS HISTÓRICO HIDROLÓGICO
    # ========================================================

    response_flow = (
        analizar_respuesta_nivel_a_caudal(
            dataset
        )
    )

    propagation = (
        analizar_propagacion_niveles(
            dataset
        )
    )

    corrientes_sn = (
        relacion_corrientes_san_nicolas(
            dataset
        )
    )

    models = {

        "version":
            "V12.0",

        "model":
            final_model,

        "feature_cols":
            feature_cols,

        "dataset":
            dataset,

        "rmse":
            rmse,

        "importance":
            importance,

        "response_flow":
            response_flow,

        "propagation":
            propagation,

        "corrientes_san_nicolas":
            corrientes_sn,

        "fecha_base":
            dataset[
                "datetime"
            ].max(),

        "uses_rain":
            any(
                (
                    "precip" in c.lower()
                    or "lluvia" in c.lower()
                    or "rain_" in c.lower()
                )
                for c in feature_cols
            ),

        "uses_caudal":
            any(
                (
                    "caudal" in c.lower()
                    or c.lower().startswith(
                        "q_"
                    )
                )
                for c in feature_cols
            ),

        "uses_upstream":
            any(
                c.startswith(
                    "nivel_"
                )
                for c in feature_cols
            ),
    }

    metrics = {

        "RMSE":
            rmse,

        "observations":
            int(
                dataset[
                    "nivel"
                ]
                .notna()
                .sum()
            ),

        "training_rows":
            int(
                len(work)
            ),

        "test_rows":
            int(
                len(test_data)
            ),

        "fecha_base":
            dataset[
                "datetime"
            ].max(),
    }

    return (
        models,
        metrics,
    )


# ============================================================
# PRONÓSTICO FUTURO
# ============================================================

def _get_future_exog(
    exog_future,
    future_date,
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

        return {}

    x = exog_future.copy()

    x[
        "datetime"
    ] = _normalizar_datetime(
        x[
            "datetime"
        ]
    )

    day = x[
        x[
            "datetime"
        ]
        ==
        pd.Timestamp(
            future_date
        ).normalize()
    ]

    if day.empty:

        return {}

    result = {}

    row = day.iloc[-1]

    for column in day.columns:

        if column == "datetime":

            continue

        value = pd.to_numeric(
            pd.Series(
                [
                    row[column]
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if not pd.isna(value):

            result[
                column
            ] = float(value)

    return result


def _project_upstream(
    history,
    step,
):

    result = {}

    for column in _base_level_columns(
        history
    ):

        current = _safe_last(
            history[column]
        )

        if not np.isfinite(
            current
        ):

            continue

        slope = _safe_slope(
            history[column],
            window=5,
        )

        damping = np.exp(
            -0.18
            * max(
                0,
                step - 1,
            )
        )

        projected = (
            current
            + slope
            * damping
        )

        result[column] = (
            _clip_level(
                projected
            )
        )

    return result


def predict(
    df,
    models,
    days=15,
    exog_future=None,
    fecha_base=None,
):

    if (
        models is None
        or "model" not in models
    ):

        raise ValueError(
            "Modelo no disponible."
        )

    model = models[
        "model"
    ]

    feature_cols = models[
        "feature_cols"
    ]

    history = models[
        "dataset"
    ].copy()

    days = max(
        1,
        min(
            int(days),
            MAX_FORECAST_DAYS,
        ),
    )

    if fecha_base is not None:

        cutoff = pd.to_datetime(
            fecha_base,
            errors="coerce",
        )

        if not pd.isna(cutoff):

            cutoff = cutoff.normalize()

            history = history[
                history[
                    "datetime"
                ] <= cutoff
            ]

    history = (
        history
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    history = history[
        history[
            "nivel"
        ].notna()
    ].copy()

    if history.empty:

        raise ValueError(
            "No existe nivel base "
            "de San Nicolás."
        )

    base_date = history[
        "datetime"
    ].iloc[-1]

    base_level = float(
        history[
            "nivel"
        ].iloc[-1]
    )

    rmse = float(
        models.get(
            "rmse",
            0.10,
        )
    )

    rows = []

    for step in range(
        1,
        days + 1,
    ):

        future_date = (
            base_date
            + pd.Timedelta(
                days=step
            )
        )

        new_row = {
            column: np.nan
            for column in history.columns
        }

        new_row[
            "datetime"
        ] = future_date

        new_row[
            "nivel"
        ] = float(
            history[
                "nivel"
            ].iloc[-1]
        )

        # ====================================================
        # LLUVIA + CAUDAL FUTUROS
        # ====================================================

        future_exog = _get_future_exog(
            exog_future,
            future_date,
        )

        for column, value in future_exog.items():

            if column not in history.columns:

                history[
                    column
                ] = np.nan

            new_row[
                column
            ] = value

        # ====================================================
        # CAUDAL
        # ====================================================

        for column in _flow_columns(
            history
        ):

            if (
                column not in new_row
                or pd.isna(
                    new_row[
                        column
                    ]
                )
            ):

                new_row[
                    column
                ] = _safe_last(
                    history[
                        column
                    ]
                )

        # ====================================================
        # NIVEL AGUAS ARRIBA
        # ====================================================

        projected = _project_upstream(
            history,
            step,
        )

        for column, value in projected.items():

            new_row[
                column
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

        features = crear_features(
            history
        )

        latest = features.iloc[
            -1
        ]

        values = {}

        valid = True

        for column in feature_cols:

            value = latest.get(
                column,
                np.nan,
            )

            if pd.isna(value):

                valid = False
                break

            values[
                column
            ] = float(value)

        if valid:

            X_future = pd.DataFrame(
                [
                    values
                ]
            )

            delta = float(
                model.predict(
                    X_future
                )[0]
            )

        else:

            # Respaldo conservador.
            delta = _safe_slope(
                history[
                    "nivel"
                ].iloc[:-1],
                window=5,
            )

        # Protección ante saltos computacionales extremos.
        delta = float(
            np.clip(
                delta,
                -0.75,
                0.75,
            )
        )

        previous = float(
            history[
                "nivel"
            ].iloc[-2]
        )

        prediction = _clip_level(
            previous
            + delta
        )

        history.loc[
            history.index[-1],
            "nivel",
        ] = prediction

        uncertainty = (
            rmse
            * 1.35
            * np.sqrt(step)
        )

        lower = _clip_level(
            prediction
            - uncertainty
        )

        upper = _clip_level(
            prediction
            + uncertainty
        )

        rows.append(
            {
                "datetime":
                    future_date,

                "prediction":
                    prediction,

                "lower":
                    lower,

                "upper":
                    upper,

                "delta_prediction":
                    delta,

                "base_level":
                    base_level,

                "base_date":
                    base_date,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# METADATOS
# ============================================================

def model_meta():

    return {

        "version":
            "V12.0",

        "nombre":
            "Modelo de propagación hidrológica",

        "objetivo":
            "Pronóstico del nivel del río Paraná "
            "en San Nicolás",

        "metodologia":
            "Predicción de variación diaria utilizando "
            "niveles aguas arriba, caudal, lluvia y "
            "retardos históricos de propagación.",

        "fecha_base":
            "La fecha seleccionada por el usuario es "
            "el último dato permitido para iniciar "
            "el pronóstico.",

        "corrientes":
            "Se calcula específicamente la relación "
            "histórica Corrientes -> San Nicolás.",

        "caudal":
            "Se analiza históricamente cómo responde "
            "el nivel de cada estación frente a "
            "aumentos del caudal.",
    }
