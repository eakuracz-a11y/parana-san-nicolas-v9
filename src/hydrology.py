import numpy as np
import pandas as pd


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/hydrology.py
# V11.5
#
# Módulo hidrológico:
# - Corrientes ↔ San Nicolás
# - retardos históricos
# - máximos históricos
# - eventos de crecida
# - lluvia
# - caudal
# - variables para modelo predictivo
# ============================================================


DEFAULT_MAX_LAG = 20

MIN_EVENT_DISTANCE_DAYS = 7

DEFAULT_EVENT_WINDOW_BEFORE = 5
DEFAULT_EVENT_WINDOW_AFTER = 20


# ============================================================
# UTILIDADES
# ============================================================


def _to_datetime_naive(series):

    x = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    return x.dt.tz_localize(None)


def _to_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _safe_corr(a, b):

    temp = pd.DataFrame(
        {
            "a": _to_numeric(a),
            "b": _to_numeric(b),
        }
    ).dropna()

    if len(temp) < 10:
        return np.nan

    if (
        temp["a"].std() == 0
        or temp["b"].std() == 0
    ):
        return np.nan

    return float(
        temp["a"].corr(
            temp["b"]
        )
    )


def _normalize_daily(
    df,
    datetime_col="datetime",
    value_col="value",
):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )

    if (
        datetime_col not in df.columns
        or value_col not in df.columns
    ):
        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )

    x = df[
        [
            datetime_col,
            value_col,
        ]
    ].copy()

    x["datetime"] = _to_datetime_naive(
        x[
            datetime_col
        ]
    )

    x["value"] = _to_numeric(
        x[
            value_col
        ]
    )

    x = x.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )

    if x.empty:
        return x

    x["datetime"] = (
        x["datetime"]
        .dt.normalize()
    )

    x = (
        x.groupby(
            "datetime",
            as_index=False,
        )["value"]
        .mean()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return x


# ============================================================
# PREPARAR SAN NICOLÁS
# ============================================================


def preparar_san_nicolas(df):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel_san_nicolas",
            ]
        )

    value_col = None

    for col in [
        "nivel",
        "value",
        "nivel_san_nicolas",
    ]:

        if col in df.columns:
            value_col = col
            break

    if value_col is None:
        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel_san_nicolas",
            ]
        )

    x = _normalize_daily(
        df,
        datetime_col="datetime",
        value_col=value_col,
    )

    if x.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel_san_nicolas",
            ]
        )

    return x.rename(
        columns={
            "value":
                "nivel_san_nicolas"
        }
    )


# ============================================================
# PREPARAR CORRIENTES
# ============================================================


def preparar_corrientes(
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
        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel_corrientes",
            ]
        )

    if (
        "datetime"
        not in upstream_history.columns
        or "nivel_corrientes"
        not in upstream_history.columns
    ):
        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel_corrientes",
            ]
        )

    return _normalize_daily(
        upstream_history,
        datetime_col="datetime",
        value_col="nivel_corrientes",
    ).rename(
        columns={
            "value":
                "nivel_corrientes"
        }
    )


# ============================================================
# UNIR CORRIENTES + SAN NICOLÁS
# ============================================================


def construir_relacion_base(
    san_nicolas,
    upstream_history,
):

    sn = preparar_san_nicolas(
        san_nicolas
    )

    corrientes = preparar_corrientes(
        upstream_history
    )

    if (
        sn.empty
        or corrientes.empty
    ):
        return pd.DataFrame()

    result = sn.merge(
        corrientes,
        on="datetime",
        how="inner",
    )

    result = (
        result
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "delta_corrientes_1d"
    ] = (
        result[
            "nivel_corrientes"
        ]
        .diff()
    )

    result[
        "delta_san_nicolas_1d"
    ] = (
        result[
            "nivel_san_nicolas"
        ]
        .diff()
    )

    result[
        "corrientes_change_3d"
    ] = (
        result[
            "nivel_corrientes"
        ]
        .diff(
            3
        )
    )

    result[
        "corrientes_change_7d"
    ] = (
        result[
            "nivel_corrientes"
        ]
        .diff(
            7
        )
    )

    result[
        "san_nicolas_change_3d"
    ] = (
        result[
            "nivel_san_nicolas"
        ]
        .diff(
            3
        )
    )

    result[
        "san_nicolas_change_7d"
    ] = (
        result[
            "nivel_san_nicolas"
        ]
        .diff(
            7
        )
    )

    return result


# ============================================================
# MEJOR RETARDO CORRIENTES → SAN NICOLÁS
# ============================================================


def calcular_lag_corrientes_san_nicolas(
    san_nicolas,
    upstream_history,
    max_lag=DEFAULT_MAX_LAG,
):

    base = construir_relacion_base(
        san_nicolas,
        upstream_history,
    )

    if base.empty:

        return {
            "best_lag_days": None,
            "correlation": np.nan,
            "samples": 0,
            "lag_table": pd.DataFrame(),
        }

    rows = []

    best_lag = None
    best_corr = np.nan
    best_score = -np.inf

    for lag in range(
        0,
        int(max_lag) + 1,
    ):

        temp = base[
            [
                "datetime",
                "nivel_corrientes",
                "nivel_san_nicolas",
            ]
        ].copy()

        # ----------------------------------------------------
        # Corrientes hoy se compara con San Nicolás
        # varios días después.
        # ----------------------------------------------------

        temp[
            "san_nicolas_future"
        ] = (
            temp[
                "nivel_san_nicolas"
            ]
            .shift(
                -lag
            )
        )

        temp = temp.dropna(
            subset=[
                "nivel_corrientes",
                "san_nicolas_future",
            ]
        )

        corr = _safe_corr(
            temp[
                "nivel_corrientes"
            ],
            temp[
                "san_nicolas_future"
            ],
        )

        score = (
            abs(corr)
            if np.isfinite(corr)
            else -np.inf
        )

        rows.append(
            {
                "lag_days":
                    lag,

                "correlation":
                    corr,

                "samples":
                    len(temp),
            }
        )

        if score > best_score:

            best_score = score

            best_lag = lag

            best_corr = corr

    return {
        "best_lag_days":
            best_lag,

        "correlation":
            best_corr,

        "samples":
            (
                int(
                    max(
                        [
                            row[
                                "samples"
                            ]
                            for row in rows
                        ],
                        default=0,
                    )
                )
            ),

        "lag_table":
            pd.DataFrame(
                rows
            ),
    }


# ============================================================
# DETECCIÓN DE MÁXIMOS
# ============================================================


def detectar_maximos_locales(
    df,
    value_col,
    min_distance_days=
        MIN_EVENT_DISTANCE_DAYS,
    quantile=0.85,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or value_col
        not in df.columns
    ):
        return pd.DataFrame()

    x = df[
        [
            "datetime",
            value_col,
        ]
    ].copy()

    x[
        value_col
    ] = _to_numeric(
        x[
            value_col
        ]
    )

    x["datetime"] = (
        _to_datetime_naive(
            x[
                "datetime"
            ]
        )
    )

    x = (
        x.dropna()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    if len(x) < 5:
        return pd.DataFrame()

    threshold = float(
        x[
            value_col
        ].quantile(
            quantile
        )
    )

    candidates = []

    values = x[
        value_col
    ].to_numpy(
        dtype=float
    )

    for i in range(
        1,
        len(x) - 1,
    ):

        current = values[
            i
        ]

        if current < threshold:
            continue

        if (
            current
            >= values[
                i - 1
            ]
            and current
            >= values[
                i + 1
            ]
        ):

            candidates.append(
                {
                    "datetime":
                        x[
                            "datetime"
                        ].iloc[
                            i
                        ],

                    "value":
                        current,
                }
            )

    if not candidates:
        return pd.DataFrame()

    candidates = sorted(
        candidates,
        key=lambda row:
            row["value"],
        reverse=True,
    )

    selected = []

    for candidate in candidates:

        fecha = pd.Timestamp(
            candidate[
                "datetime"
            ]
        )

        demasiado_cerca = False

        for previous in selected:

            distancia = abs(
                (
                    fecha
                    - pd.Timestamp(
                        previous[
                            "datetime"
                        ]
                    )
                ).days
            )

            if (
                distancia
                < min_distance_days
            ):
                demasiado_cerca = True
                break

        if not demasiado_cerca:

            selected.append(
                candidate
            )

    selected = sorted(
        selected,
        key=lambda row:
            row[
                "datetime"
            ],
    )

    return pd.DataFrame(
        selected
    )


# ============================================================
# EVENTOS HISTÓRICOS CORRIENTES → SAN NICOLÁS
# ============================================================


def construir_eventos_corrientes_san_nicolas(
    san_nicolas,
    upstream_history,
    lag_days=None,
    window_before=
        DEFAULT_EVENT_WINDOW_BEFORE,
    window_after=
        DEFAULT_EVENT_WINDOW_AFTER,
    quantile=0.85,
):

    base = construir_relacion_base(
        san_nicolas,
        upstream_history,
    )

    if base.empty:
        return pd.DataFrame()

    if lag_days is None:

        lag_info = (
            calcular_lag_corrientes_san_nicolas(
                san_nicolas,
                upstream_history,
            )
        )

        lag_days = (
            lag_info.get(
                "best_lag_days"
            )
        )

    if lag_days is None:
        lag_days = 7

    peaks = detectar_maximos_locales(
        base,
        value_col=
            "nivel_corrientes",
        quantile=
            quantile,
    )

    if peaks.empty:
        return pd.DataFrame()

    rows = []

    for _, peak in peaks.iterrows():

        corrientes_date = (
            pd.Timestamp(
                peak[
                    "datetime"
                ]
            )
        )

        corrientes_max = float(
            peak[
                "value"
            ]
        )

        target_date = (
            corrientes_date
            + pd.Timedelta(
                days=int(
                    lag_days
                )
            )
        )

        start = (
            target_date
            - pd.Timedelta(
                days=
                    int(
                        window_before
                    )
            )
        )

        end = (
            target_date
            + pd.Timedelta(
                days=
                    int(
                        window_after
                    )
            )
        )

        sn_window = base[
            (
                base[
                    "datetime"
                ]
                >= start
            )
            &
            (
                base[
                    "datetime"
                ]
                <= end
            )
        ].copy()

        if sn_window.empty:
            continue

        idx_max = (
            sn_window[
                "nivel_san_nicolas"
            ]
            .idxmax()
        )

        sn_max = float(
            sn_window.loc[
                idx_max,
                "nivel_san_nicolas",
            ]
        )

        sn_max_date = pd.Timestamp(
            sn_window.loc[
                idx_max,
                "datetime",
            ]
        )

        real_lag = int(
            (
                sn_max_date
                - corrientes_date
            ).days
        )

        sn_before = base[
            base[
                "datetime"
            ]
            <= corrientes_date
        ].tail(
            7
        )

        if sn_before.empty:

            sn_base = np.nan

        else:

            sn_base = float(
                sn_before[
                    "nivel_san_nicolas"
                ].mean()
            )

        response = (
            sn_max
            - sn_base
            if np.isfinite(
                sn_base
            )
            else np.nan
        )

        rows.append(
            {
                "fecha_max_corrientes":
                    corrientes_date,

                "max_corrientes_m":
                    corrientes_max,

                "fecha_max_san_nicolas":
                    sn_max_date,

                "max_san_nicolas_m":
                    sn_max,

                "lag_real_dias":
                    real_lag,

                "nivel_base_san_nicolas_m":
                    sn_base,

                "respuesta_san_nicolas_m":
                    response,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    result = (
        result
        .sort_values(
            "fecha_max_corrientes"
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# PREPARAR CAUDAL Y LLUVIA
# ============================================================


def preparar_exogenas(
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

    x = exog_history.copy()

    x[
        "datetime"
    ] = _to_datetime_naive(
        x[
            "datetime"
        ]
    )

    x[
        "datetime"
    ] = (
        x[
            "datetime"
        ]
        .dt.normalize()
    )

    keep = [
        "datetime"
    ]

    for col in [
        "caudal_m3s",
        "precip_mm",
    ]:

        if col in x.columns:

            x[col] = (
                _to_numeric(
                    x[col]
                )
            )

            keep.append(
                col
            )

    x = x[
        keep
    ]

    agg = {}

    if (
        "caudal_m3s"
        in x.columns
    ):

        agg[
            "caudal_m3s"
        ] = "mean"

    if (
        "precip_mm"
        in x.columns
    ):

        agg[
            "precip_mm"
        ] = "sum"

    if not agg:
        return pd.DataFrame()

    return (
        x.groupby(
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
# AGREGAR CAUDAL Y LLUVIA A EVENTOS
# ============================================================


def enriquecer_eventos_con_exogenas(
    events,
    exog_history,
    rain_window_days=15,
    flow_window_days=7,
):

    if (
        events is None
        or not isinstance(
            events,
            pd.DataFrame,
        )
        or events.empty
    ):
        return pd.DataFrame()

    exog = preparar_exogenas(
        exog_history
    )

    result = events.copy()

    if exog.empty:
        return result

    rain_values = []
    flow_mean_values = []
    flow_max_values = []

    for _, row in result.iterrows():

        event_date = pd.Timestamp(
            row[
                "fecha_max_corrientes"
            ]
        )

        # ----------------------------------------------------
        # LLUVIA PREVIA
        # ----------------------------------------------------

        rain_start = (
            event_date
            - pd.Timedelta(
                days=int(
                    rain_window_days
                )
            )
        )

        rain_window = exog[
            (
                exog[
                    "datetime"
                ]
                >= rain_start
            )
            &
            (
                exog[
                    "datetime"
                ]
                <= event_date
            )
        ]

        if (
            "precip_mm"
            in rain_window.columns
        ):

            rain_total = float(
                rain_window[
                    "precip_mm"
                ]
                .fillna(
                    0
                )
                .sum()
            )

        else:

            rain_total = np.nan

        # ----------------------------------------------------
        # CAUDAL ALREDEDOR DEL EVENTO
        # ----------------------------------------------------

        flow_start = (
            event_date
            - pd.Timedelta(
                days=int(
                    flow_window_days
                )
            )
        )

        flow_end = (
            event_date
            + pd.Timedelta(
                days=int(
                    flow_window_days
                )
            )
        )

        flow_window = exog[
            (
                exog[
                    "datetime"
                ]
                >= flow_start
            )
            &
            (
                exog[
                    "datetime"
                ]
                <= flow_end
            )
        ]

        if (
            "caudal_m3s"
            in flow_window.columns
        ):

            q = (
                flow_window[
                    "caudal_m3s"
                ]
                .dropna()
            )

            if not q.empty:

                flow_mean = float(
                    q.mean()
                )

                flow_max = float(
                    q.max()
                )

            else:

                flow_mean = np.nan
                flow_max = np.nan

        else:

            flow_mean = np.nan
            flow_max = np.nan

        rain_values.append(
            rain_total
        )

        flow_mean_values.append(
            flow_mean
        )

        flow_max_values.append(
            flow_max
        )

    result[
        "lluvia_previa_mm"
    ] = rain_values

    result[
        "caudal_medio_m3s"
    ] = flow_mean_values

    result[
        "caudal_max_m3s"
    ] = flow_max_values

    return result


# ============================================================
# RESUMEN DE EVENTOS MÁXIMOS
# ============================================================


def resumen_eventos_maximos(
    events,
    top_n=10,
):

    if (
        events is None
        or not isinstance(
            events,
            pd.DataFrame,
        )
        or events.empty
    ):
        return pd.DataFrame()

    x = events.copy()

    if (
        "max_corrientes_m"
        in x.columns
    ):

        x = x.sort_values(
            "max_corrientes_m",
            ascending=False,
        )

    return (
        x.head(
            int(
                top_n
            )
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# ESTADÍSTICAS DE PROPAGACIÓN
# ============================================================


def estadisticas_propagacion(
    events,
):

    if (
        events is None
        or not isinstance(
            events,
            pd.DataFrame,
        )
        or events.empty
    ):

        return {
            "events": 0,
            "median_lag_days": None,
            "mean_lag_days": None,
            "median_response_m": None,
            "mean_response_m": None,
            "correlation_maxima": np.nan,
        }

    result = {
        "events":
            len(
                events
            ),

        "median_lag_days":
            None,

        "mean_lag_days":
            None,

        "median_response_m":
            None,

        "mean_response_m":
            None,

        "correlation_maxima":
            np.nan,
    }

    if (
        "lag_real_dias"
        in events.columns
    ):

        lag = _to_numeric(
            events[
                "lag_real_dias"
            ]
        ).dropna()

        if not lag.empty:

            result[
                "median_lag_days"
            ] = float(
                lag.median()
            )

            result[
                "mean_lag_days"
            ] = float(
                lag.mean()
            )

    if (
        "respuesta_san_nicolas_m"
        in events.columns
    ):

        response = _to_numeric(
            events[
                "respuesta_san_nicolas_m"
            ]
        ).dropna()

        if not response.empty:

            result[
                "median_response_m"
            ] = float(
                response.median()
            )

            result[
                "mean_response_m"
            ] = float(
                response.mean()
            )

    if (
        "max_corrientes_m"
        in events.columns
        and "max_san_nicolas_m"
        in events.columns
    ):

        result[
            "correlation_maxima"
        ] = _safe_corr(
            events[
                "max_corrientes_m"
            ],
            events[
                "max_san_nicolas_m"
            ],
        )

    return result


# ============================================================
# SIMILITUD ENTRE CONDICIÓN ACTUAL Y EVENTOS HISTÓRICOS
# ============================================================


def buscar_eventos_similares(
    events,
    current_corrientes=None,
    current_flow=None,
    recent_rain=None,
    top_n=5,
):

    if (
        events is None
        or not isinstance(
            events,
            pd.DataFrame,
        )
        or events.empty
    ):
        return pd.DataFrame()

    x = events.copy()

    components = []

    # --------------------------------------------------------
    # NIVEL CORRIENTES
    # --------------------------------------------------------

    if (
        current_corrientes
        is not None
        and "max_corrientes_m"
        in x.columns
    ):

        values = _to_numeric(
            x[
                "max_corrientes_m"
            ]
        )

        scale = (
            values.std()
        )

        if (
            not np.isfinite(
                scale
            )
            or scale <= 0
        ):
            scale = 1.0

        diff = (
            values
            - float(
                current_corrientes
            )
        ).abs() / scale

        components.append(
            diff.rename(
                "corrientes_score"
            )
        )

    # --------------------------------------------------------
    # CAUDAL
    # --------------------------------------------------------

    if (
        current_flow
        is not None
        and "caudal_medio_m3s"
        in x.columns
    ):

        values = _to_numeric(
            x[
                "caudal_medio_m3s"
            ]
        )

        scale = (
            values.std()
        )

        if (
            not np.isfinite(
                scale
            )
            or scale <= 0
        ):
            scale = 1.0

        diff = (
            values
            - float(
                current_flow
            )
        ).abs() / scale

        components.append(
            diff.rename(
                "flow_score"
            )
        )

    # --------------------------------------------------------
    # LLUVIA
    # --------------------------------------------------------

    if (
        recent_rain
        is not None
        and "lluvia_previa_mm"
        in x.columns
    ):

        values = _to_numeric(
            x[
                "lluvia_previa_mm"
            ]
        )

        scale = (
            values.std()
        )

        if (
            not np.isfinite(
                scale
            )
            or scale <= 0
        ):
            scale = 1.0

        diff = (
            values
            - float(
                recent_rain
            )
        ).abs() / scale

        components.append(
            diff.rename(
                "rain_score"
            )
        )

    if not components:

        return pd.DataFrame()

    score_df = pd.concat(
        components,
        axis=1,
    )

    x[
        "similarity_score"
    ] = (
        score_df
        .mean(
            axis=1,
            skipna=True,
        )
    )

    x = (
        x.sort_values(
            "similarity_score"
        )
        .head(
            int(
                top_n
            )
        )
        .reset_index(
            drop=True
        )
    )

    return x


# ============================================================
# VARIABLES HIDROLÓGICAS PARA EL MODELO
# ============================================================


def crear_features_hidrologicas(
    san_nicolas,
    upstream_history=None,
    exog_history=None,
):

    sn = preparar_san_nicolas(
        san_nicolas
    )

    if sn.empty:
        return pd.DataFrame()

    result = sn.copy()

    # --------------------------------------------------------
    # CORRIENTES
    # --------------------------------------------------------

    corrientes = preparar_corrientes(
        upstream_history
    )

    if not corrientes.empty:

        result = result.merge(
            corrientes,
            on="datetime",
            how="left",
        )

        for lag in [
            1,
            3,
            5,
            7,
            10,
            14,
        ]:

            result[
                f"corrientes_lag_{lag}"
            ] = (
                result[
                    "nivel_corrientes"
                ]
                .shift(
                    lag
                )
            )

        result[
            "corrientes_diff_1d"
        ] = (
            result[
                "nivel_corrientes"
            ]
            .diff()
        )

        result[
            "corrientes_diff_3d"
        ] = (
            result[
                "nivel_corrientes"
            ]
            .diff(
                3
            )
        )

        result[
            "corrientes_diff_7d"
        ] = (
            result[
                "nivel_corrientes"
            ]
            .diff(
                7
            )
        )

        result[
            "corrientes_roll_max_7d"
        ] = (
            result[
                "nivel_corrientes"
            ]
            .rolling(
                7,
                min_periods=1,
            )
            .max()
        )

        result[
            "corrientes_roll_max_15d"
        ] = (
            result[
                "nivel_corrientes"
            ]
            .rolling(
                15,
                min_periods=1,
            )
            .max()
        )

    # --------------------------------------------------------
    # CAUDAL Y LLUVIA
    # --------------------------------------------------------

    exog = preparar_exogenas(
        exog_history
    )

    if not exog.empty:

        result = result.merge(
            exog,
            on="datetime",
            how="left",
        )

        if (
            "caudal_m3s"
            in result.columns
        ):

            result[
                "caudal_diff_1d"
            ] = (
                result[
                    "caudal_m3s"
                ]
                .diff()
            )

            result[
                "caudal_diff_3d"
            ] = (
                result[
                    "caudal_m3s"
                ]
                .diff(
                    3
                )
            )

            result[
                "caudal_mean_3d"
            ] = (
                result[
                    "caudal_m3s"
                ]
                .rolling(
                    3,
                    min_periods=1,
                )
                .mean()
            )

            result[
                "caudal_mean_7d"
            ] = (
                result[
                    "caudal_m3s"
                ]
                .rolling(
                    7,
                    min_periods=1,
                )
                .mean()
            )

            result[
                "caudal_max_7d"
            ] = (
                result[
                    "caudal_m3s"
                ]
                .rolling(
                    7,
                    min_periods=1,
                )
                .max()
            )

        if (
            "precip_mm"
            in result.columns
        ):

            result[
                "lluvia_3d"
            ] = (
                result[
                    "precip_mm"
                ]
                .fillna(
                    0
                )
                .rolling(
                    3,
                    min_periods=1,
                )
                .sum()
            )

            result[
                "lluvia_7d"
            ] = (
                result[
                    "precip_mm"
                ]
                .fillna(
                    0
                )
                .rolling(
                    7,
                    min_periods=1,
                )
                .sum()
            )

            result[
                "lluvia_15d"
            ] = (
                result[
                    "precip_mm"
                ]
                .fillna(
                    0
                )
                .rolling(
                    15,
                    min_periods=1,
                )
                .sum()
            )

    # --------------------------------------------------------
    # NIVEL LOCAL
    # --------------------------------------------------------

    for lag in [
        1,
        2,
        3,
        5,
        7,
        10,
        14,
    ]:

        result[
            f"san_nicolas_lag_{lag}"
        ] = (
            result[
                "nivel_san_nicolas"
            ]
            .shift(
                lag
            )
        )

    result[
        "san_nicolas_diff_1d"
    ] = (
        result[
            "nivel_san_nicolas"
        ]
        .diff()
    )

    result[
        "san_nicolas_diff_3d"
    ] = (
        result[
            "nivel_san_nicolas"
        ]
        .diff(
            3
        )
    )

    result[
        "san_nicolas_mean_7d"
    ] = (
        result[
            "nivel_san_nicolas"
        ]
        .rolling(
            7,
            min_periods=1,
        )
        .mean()
    )

    return result


# ============================================================
# ANÁLISIS COMPLETO
# ============================================================


def analizar_corrientes_san_nicolas(
    san_nicolas,
    upstream_history,
    exog_history=None,
    max_lag=DEFAULT_MAX_LAG,
):

    lag = (
        calcular_lag_corrientes_san_nicolas(
            san_nicolas,
            upstream_history,
            max_lag=max_lag,
        )
    )

    events = (
        construir_eventos_corrientes_san_nicolas(
            san_nicolas,
            upstream_history,
            lag_days=
                lag.get(
                    "best_lag_days"
                ),
        )
    )

    if (
        exog_history is not None
        and not events.empty
    ):

        events = (
            enriquecer_eventos_con_exogenas(
                events,
                exog_history,
            )
        )

    stats = (
        estadisticas_propagacion(
            events
        )
    )

    return {
        "lag":
            lag,

        "events":
            events,

        "statistics":
            stats,

        "top_events":
            resumen_eventos_maximos(
                events,
                top_n=10,
            ),

        "features":
            crear_features_hidrologicas(
                san_nicolas,
                upstream_history,
                exog_history,
            ),
    }
