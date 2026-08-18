import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# CONFIGURACIÓN
# ============================================================

STRESS_DAYS = 60

LEVEL_MIN = 0.0

# Límite interno.
# La escala visual del app.py puede continuar en 0–7 m.
LEVEL_MAX_INTERNAL = 12.0

EPS = 1e-9


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
        .tz_localize(None)
        .dt
        .normalize()
    )


def _numeric(
    serie,
):

    return pd.to_numeric(
        serie,
        errors="coerce",
    )


def _safe_float(
    value,
    default=np.nan,
):

    try:

        value = float(value)

        if np.isfinite(value):

            return value

    except Exception:

        pass

    return default


def _format_number(
    value,
    decimals=1,
):

    if value is None:

        return "--"

    try:

        if pd.isna(value):

            return "--"

    except Exception:

        pass

    text = f"{float(value):,.{decimals}f}"

    return (
        text
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def _upstream_cols(
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

        return []

    return [
        c
        for c
        in df.columns
        if (
            c.startswith("nivel_")
            and c != "nivel"
            and "_lag" not in c
            and "_diff" not in c
            and "_trend" not in c
            and "_mean" not in c
            and "_actual" not in c
            and "_next" not in c
        )
    ]


def _percentile_rank(
    serie,
):

    s = _numeric(
        serie
    )

    if s.notna().sum() < 2:

        return pd.Series(
            0.0,
            index=s.index,
            dtype=float,
        )

    return (
        s
        .rank(
            pct=True,
            method="average",
        )
        .fillna(0.0)
    )


# ============================================================
# NIVEL SAN NICOLÁS
# ============================================================

def _preparar_nivel(
    df,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "datetime"
        not in df.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel",
            ]
        )

    x = df.copy()

    if "nivel" in x.columns:

        x[
            "nivel"
        ] = _numeric(
            x[
                "nivel"
            ]
        )

    elif "value" in x.columns:

        x[
            "nivel"
        ] = _numeric(
            x[
                "value"
            ]
        )

    else:

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel",
            ]
        )

    x[
        "datetime"
    ] = _normalizar_datetime(
        x[
            "datetime"
        ]
    )

    x = (
        x
        .dropna(
            subset=[
                "datetime",
                "nivel",
            ]
        )
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

    return x


# ============================================================
# VARIABLES EXÓGENAS
# ============================================================

def _preparar_exog(
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

        return pd.DataFrame(
            columns=[
                "datetime",
                "precip_mm",
                "caudal_m3s",
            ]
        )

    x = exog_history.copy()

    x[
        "datetime"
    ] = _normalizar_datetime(
        x[
            "datetime"
        ]
    )

    if "precip_mm" not in x.columns:

        x[
            "precip_mm"
        ] = np.nan

    if "caudal_m3s" not in x.columns:

        x[
            "caudal_m3s"
        ] = np.nan

    x[
        "precip_mm"
    ] = (
        _numeric(
            x[
                "precip_mm"
            ]
        )
        .clip(
            lower=0.0
        )
    )

    x[
        "caudal_m3s"
    ] = _numeric(
        x[
            "caudal_m3s"
        ]
    )

    x = (
        x[
            [
                "datetime",
                "precip_mm",
                "caudal_m3s",
            ]
        ]
        .dropna(
            subset=[
                "datetime"
            ]
        )
        .groupby(
            "datetime",
            as_index=False,
        )
        .agg(
            {
                "precip_mm":
                    "mean",

                "caudal_m3s":
                    "mean",
            }
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return x


# ============================================================
# AGUAS ARRIBA
# ============================================================

def _preparar_upstream(
    upstream_history,
):

    if (
        upstream_history is None
        or not isinstance(
            upstream_history,
            pd.DataFrame,
        )
        or upstream_history.empty
        or "datetime"
        not in upstream_history.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "upstream_mean",
                "upstream_max",
            ]
        )

    x = upstream_history.copy()

    x[
        "datetime"
    ] = _normalizar_datetime(
        x[
            "datetime"
        ]
    )

    cols = _upstream_cols(
        x
    )

    if not cols:

        return pd.DataFrame(
            {
                "datetime":
                    x[
                        "datetime"
                    ],

                "upstream_mean":
                    np.nan,

                "upstream_max":
                    np.nan,
            }
        )

    for col in cols:

        x[
            col
        ] = _numeric(
            x[
                col
            ]
        )

        if (
            x[
                col
            ]
            .notna()
            .sum()
            >= 2
        ):

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

    x[
        "upstream_mean"
    ] = (
        x[
            cols
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    x[
        "upstream_max"
    ] = (
        x[
            cols
        ]
        .max(
            axis=1,
            skipna=True,
        )
    )

    return (
        x[
            [
                "datetime",
                "upstream_mean",
                "upstream_max",
            ]
        ]
        .dropna(
            subset=[
                "datetime"
            ]
        )
        .groupby(
            "datetime",
            as_index=False,
        )
        .agg(
            {
                "upstream_mean":
                    "mean",

                "upstream_max":
                    "max",
            }
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# HISTÓRICO INTEGRADO
# ============================================================

def _armar_historico(
    df,
    exog_history,
    upstream_history,
):

    nivel = _preparar_nivel(
        df
    )

    exog = _preparar_exog(
        exog_history
    )

    upstream = _preparar_upstream(
        upstream_history
    )

    if nivel.empty:

        return pd.DataFrame()

    # Partimos de las fechas de San Nicolás.
    hist = nivel.copy()

    if not exog.empty:

        hist = hist.merge(
            exog,
            on="datetime",
            how="left",
        )

    if not upstream.empty:

        hist = hist.merge(
            upstream,
            on="datetime",
            how="left",
        )

    for col in [
        "precip_mm",
        "caudal_m3s",
        "upstream_mean",
        "upstream_max",
    ]:

        if col not in hist.columns:

            hist[
                col
            ] = np.nan

        hist[
            col
        ] = _numeric(
            hist[
                col
            ]
        )

    # Lluvia:
    # cero sólo es válido si viene informado como cero.
    # No transformamos todos los faltantes en cero.

    if (
        hist[
            "caudal_m3s"
        ]
        .notna()
        .sum()
        >= 7
    ):

        hist[
            "caudal_m3s"
        ] = (
            hist[
                "caudal_m3s"
            ]
            .interpolate(
                limit=5,
                limit_direction="both",
            )
        )

    for col in [
        "upstream_mean",
        "upstream_max",
    ]:

        if (
            hist[
                col
            ]
            .notna()
            .sum()
            >= 7
        ):

            hist[
                col
            ] = (
                hist[
                    col
                ]
                .interpolate(
                    limit=3,
                    limit_direction="both",
                )
            )

    return (
        hist
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


# ============================================================
# VARIABLES DE SEVERIDAD
# ============================================================

def _calcular_severidad(
    hist,
):

    x = hist.copy()

    x[
        "delta_nivel"
    ] = x[
        "nivel"
    ].diff()

    x[
        "rain_7d"
    ] = (
        x[
            "precip_mm"
        ]
        .fillna(0.0)
        .rolling(
            7,
            min_periods=1,
        )
        .sum()
    )

    x[
        "rain_15d"
    ] = (
        x[
            "precip_mm"
        ]
        .fillna(0.0)
        .rolling(
            15,
            min_periods=1,
        )
        .sum()
    )

    x[
        "rain_30d"
    ] = (
        x[
            "precip_mm"
        ]
        .fillna(0.0)
        .rolling(
            30,
            min_periods=1,
        )
        .sum()
    )

    x[
        "q_7d"
    ] = (
        x[
            "caudal_m3s"
        ]
        .rolling(
            7,
            min_periods=1,
        )
        .mean()
    )

    x[
        "q_15d"
    ] = (
        x[
            "caudal_m3s"
        ]
        .rolling(
            15,
            min_periods=1,
        )
        .mean()
    )

    x[
        "up_7d"
    ] = (
        x[
            "upstream_mean"
        ]
        .rolling(
            7,
            min_periods=1,
        )
        .mean()
    )

    x[
        "nivel_7d"
    ] = (
        x[
            "nivel"
        ]
        .rolling(
            7,
            min_periods=1,
        )
        .mean()
    )

    # Crecida histórica desde mínimo móvil reciente.
    rolling_min = (
        x[
            "nivel"
        ]
        .rolling(
            30,
            min_periods=1,
        )
        .min()
    )

    x[
        "historical_rise"
    ] = (
        x[
            "nivel"
        ]
        - rolling_min
    )

    components = []

    weights = []

    candidates = [

        (
            "rain_7d",
            0.13,
        ),

        (
            "rain_15d",
            0.10,
        ),

        (
            "rain_30d",
            0.08,
        ),

        (
            "q_7d",
            0.17,
        ),

        (
            "q_15d",
            0.12,
        ),

        (
            "up_7d",
            0.13,
        ),

        (
            "nivel_7d",
            0.10,
        ),

        (
            "historical_rise",
            0.17,
        ),
    ]

    for col, weight in candidates:

        if (
            x[
                col
            ]
            .notna()
            .sum()
            >= 10
        ):

            components.append(
                _percentile_rank(
                    x[
                        col
                    ]
                )
                * weight
            )

            weights.append(
                weight
            )

    if not components:

        x[
            "severity"
        ] = 0.0

        return x

    total = components[
        0
    ].copy()

    for component in components[
        1:
    ]:

        total = (
            total
            + component
        )

    x[
        "severity"
    ] = (
        total
        / max(
            sum(weights),
            EPS,
        )
    )

    return x


# ============================================================
# SELECCIONAR PEOR SECUENCIA HISTÓRICA
# ============================================================

def _seleccionar_peor_bloque(
    hist,
    days=STRESS_DAYS,
):

    x = _calcular_severidad(
        hist
    )

    if x.empty:

        return pd.DataFrame()

    if len(x) <= days:

        return x.copy()

    scores = []

    # Evaluamos cada ventana continua de 60 días.
    for end_position in range(
        days - 1,
        len(x),
    ):

        start_position = (
            end_position
            - days
            + 1
        )

        block = x.iloc[
            start_position:
            end_position + 1
        ]

        if len(block) < days:

            continue

        nivel_block = (
            block[
                "nivel"
            ]
            .dropna()
        )

        if nivel_block.empty:

            continue

        nivel_inicio = float(
            nivel_block.iloc[0]
        )

        nivel_max = float(
            nivel_block.max()
        )

        rise = max(
            nivel_max
            - nivel_inicio,
            0.0,
        )

        severity_mean = float(
            block[
                "severity"
            ]
            .fillna(0.0)
            .mean()
        )

        # Priorizamos eventos que históricamente generaron
        # una crecida real, no solamente valores altos aislados.

        rise_rank_component = rise

        score = (
            0.62
            * severity_mean
            + 0.38
            * min(
                rise_rank_component
                / 2.0,
                1.0,
            )
        )

        scores.append(
            (
                score,
                start_position,
                end_position,
                rise,
            )
        )

    if not scores:

        return (
            x
            .tail(days)
            .copy()
            .reset_index(
                drop=True
            )
        )

    scores.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )

    _, start_position, end_position, _ = (
        scores[0]
    )

    return (
        x.iloc[
            start_position:
            end_position + 1
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# RESPUESTA EMPÍRICA
# ============================================================

def _ajustar_respuesta_empirica(
    hist,
):

    x = hist.copy()

    x[
        "delta_nivel"
    ] = x[
        "nivel"
    ].diff()

    x[
        "rain_3d"
    ] = (
        x[
            "precip_mm"
        ]
        .fillna(0.0)
        .rolling(
            3,
            min_periods=1,
        )
        .sum()
    )

    x[
        "rain_7d"
    ] = (
        x[
            "precip_mm"
        ]
        .fillna(0.0)
        .rolling(
            7,
            min_periods=1,
        )
        .sum()
    )

    x[
        "q_diff1"
    ] = (
        x[
            "caudal_m3s"
        ]
        .diff()
    )

    x[
        "q_diff7"
    ] = (
        x[
            "caudal_m3s"
        ]
        - x[
            "caudal_m3s"
        ].shift(7)
    )

    x[
        "up_diff1"
    ] = (
        x[
            "upstream_mean"
        ]
        .diff()
    )

    x[
        "up_diff7"
    ] = (
        x[
            "upstream_mean"
        ]
        - x[
            "upstream_mean"
        ].shift(7)
    )

    candidates = [
        "rain_3d",
        "rain_7d",
        "caudal_m3s",
        "q_diff1",
        "q_diff7",
        "upstream_mean",
        "up_diff1",
        "up_diff7",
    ]

    feature_names = []

    for col in candidates:

        if (
            x[
                col
            ]
            .notna()
            .sum()
            >= 30
        ):

            feature_names.append(
                col
            )

    if not feature_names:

        return None

    work = (
        x[
            [
                "delta_nivel",
            ]
            + feature_names
        ]
        .dropna()
    )

    if len(work) < 40:

        return None

    means = (
        work[
            feature_names
        ]
        .mean()
    )

    stds = (
        work[
            feature_names
        ]
        .std()
        .replace(
            0.0,
            1.0,
        )
        .fillna(
            1.0
        )
    )

    X = (
        (
            work[
                feature_names
            ]
            - means
        )
        / stds
    ).to_numpy(
        dtype=float
    )

    X = np.column_stack(
        [
            np.ones(
                len(X)
            ),
            X,
        ]
    )

    y = (
        work[
            "delta_nivel"
        ]
        .to_numpy(
            dtype=float
        )
    )

    try:

        coef, _, _, _ = (
            np.linalg.lstsq(
                X,
                y,
                rcond=None,
            )
        )

    except Exception:

        return None

    abs_delta = (
        x[
            "delta_nivel"
        ]
        .abs()
        .dropna()
    )

    if abs_delta.empty:

        daily_limit = 0.25

    else:

        daily_limit = float(
            abs_delta.quantile(
                0.995
            )
        )

    daily_limit = float(
        np.clip(
            daily_limit,
            0.08,
            0.60,
        )
    )

    return {

        "feature_names":
            feature_names,

        "means":
            means,

        "stds":
            stds,

        "coef":
            coef,

        "daily_limit":
            daily_limit,
    }


# ============================================================
# ESCENARIO SEVERO 60 DÍAS
# ============================================================

def _crear_escenario_60_dias(
    hist,
    block,
):

    if (
        hist.empty
        or block.empty
    ):

        return pd.DataFrame()

    nivel_valid = (
        hist[
            "nivel"
        ]
        .dropna()
    )

    if nivel_valid.empty:

        return pd.DataFrame()

    nivel_actual = float(
        nivel_valid.iloc[-1]
    )

    ultima_fecha = hist[
        "datetime"
    ].max()

    future_dates = pd.date_range(
        ultima_fecha
        + pd.Timedelta(days=1),
        periods=STRESS_DAYS,
        freq="D",
    )

    response = (
        _ajustar_respuesta_empirica(
            hist
        )
    )

    block = (
        block
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(block) < STRESS_DAYS:

        repeat_count = int(
            np.ceil(
                STRESS_DAYS
                / len(block)
            )
        )

        block = pd.concat(
            [
                block
            ]
            * repeat_count,
            ignore_index=True,
        )

    block = (
        block
        .head(
            STRESS_DAYS
        )
        .copy()
    )

    block[
        "historical_delta"
    ] = (
        block[
            "nivel"
        ]
        .diff()
    )

    # ========================================================
    # PERCENTILES HISTÓRICOS ALTOS
    # ========================================================

    q95_rain = (
        float(
            hist[
                "precip_mm"
            ]
            .dropna()
            .quantile(0.95)
        )
        if hist[
            "precip_mm"
        ].notna().any()
        else 0.0
    )

    q99_rain = (
        float(
            hist[
                "precip_mm"
            ]
            .dropna()
            .quantile(0.99)
        )
        if hist[
            "precip_mm"
        ].notna().any()
        else 0.0
    )

    q95_flow = (
        float(
            hist[
                "caudal_m3s"
            ]
            .dropna()
            .quantile(0.95)
        )
        if hist[
            "caudal_m3s"
        ].notna().any()
        else np.nan
    )

    q99_flow = (
        float(
            hist[
                "caudal_m3s"
            ]
            .dropna()
            .quantile(0.99)
        )
        if hist[
            "caudal_m3s"
        ].notna().any()
        else np.nan
    )

    q95_up = (
        float(
            hist[
                "upstream_mean"
            ]
            .dropna()
            .quantile(0.95)
        )
        if hist[
            "upstream_mean"
        ].notna().any()
        else np.nan
    )

    q99_up = (
        float(
            hist[
                "upstream_mean"
            ]
            .dropna()
            .quantile(0.99)
        )
        if hist[
            "upstream_mean"
        ].notna().any()
        else np.nan
    )

    # ========================================================
    # CONSTRUIR TRAYECTORIA
    # ========================================================

    results = []

    nivel = nivel_actual

    rain_buffer = []

    prev_q = np.nan

    prev_up = np.nan

    for i in range(
        STRESS_DAYS
    ):

        row = block.iloc[i]

        rain_hist = _safe_float(
            row.get(
                "precip_mm",
                np.nan,
            )
        )

        q_hist = _safe_float(
            row.get(
                "caudal_m3s",
                np.nan,
            )
        )

        up_hist = _safe_float(
            row.get(
                "upstream_mean",
                np.nan,
            )
        )

        hist_delta = _safe_float(
            row.get(
                "historical_delta",
                np.nan,
            )
        )

        # ====================================================
        # LLUVIA SEVERA
        # ====================================================

        if pd.isna(
            rain_hist
        ):

            rain = q95_rain

        elif rain_hist > 0:

            # Conservamos la forma histórica del evento,
            # elevándolo dentro del rango alto registrado.

            rain = max(
                rain_hist,
                min(
                    rain_hist
                    * 1.10,
                    q99_rain,
                ),
            )

        else:

            # No transformamos todos los días secos
            # en lluvias máximas ficticias.

            rain = 0.0

        # ====================================================
        # CAUDAL SEVERO
        # ====================================================

        if pd.isna(
            q_hist
        ):

            q = q95_flow

        else:

            q = q_hist

            if pd.notna(
                q95_flow
            ):

                q = max(
                    q,
                    min(
                        q
                        * 1.03,
                        q99_flow
                        if pd.notna(
                            q99_flow
                        )
                        else q95_flow,
                    ),
                )

        # ====================================================
        # AGUAS ARRIBA SEVERO
        # ====================================================

        if pd.isna(
            up_hist
        ):

            up = q95_up

        else:

            up = up_hist

            if pd.notna(
                q95_up
            ):

                up = max(
                    up,
                    min(
                        up
                        * 1.02,
                        q99_up
                        if pd.notna(
                            q99_up
                        )
                        else q95_up,
                    ),
                )

        rain_buffer.append(
            float(
                rain
                if pd.notna(rain)
                else 0.0
            )
        )

        rain_3d = float(
            sum(
                rain_buffer[-3:]
            )
        )

        rain_7d = float(
            sum(
                rain_buffer[-7:]
            )
        )

        empirical_delta = np.nan

        # ====================================================
        # RESPUESTA EMPÍRICA
        # ====================================================

        if response is not None:

            feature_values = {

                "rain_3d":
                    rain_3d,

                "rain_7d":
                    rain_7d,

                "caudal_m3s":
                    q,

                "q_diff1":
                    (
                        q - prev_q
                        if (
                            pd.notna(q)
                            and pd.notna(prev_q)
                        )
                        else 0.0
                    ),

                "q_diff7":
                    0.0,

                "upstream_mean":
                    up,

                "up_diff1":
                    (
                        up - prev_up
                        if (
                            pd.notna(up)
                            and pd.notna(prev_up)
                        )
                        else 0.0
                    ),

                "up_diff7":
                    0.0,
            }

            z = []

            for name in response[
                "feature_names"
            ]:

                value = (
                    feature_values
                    .get(
                        name,
                        np.nan,
                    )
                )

                mean = float(
                    response[
                        "means"
                    ][name]
                )

                std = float(
                    response[
                        "stds"
                    ][name]
                )

                if pd.isna(value):

                    value = mean

                z.append(
                    (
                        float(value)
                        - mean
                    )
                    / max(
                        std,
                        EPS,
                    )
                )

            vector = np.array(
                [
                    1.0
                ]
                + z,
                dtype=float,
            )

            empirical_delta = float(
                vector
                @ response[
                    "coef"
                ]
            )

        # ====================================================
        # COMBINAR RESPUESTA HISTÓRICA + EMPÍRICA
        # ====================================================

        if (
            pd.notna(
                empirical_delta
            )
            and pd.notna(
                hist_delta
            )
        ):

            delta = (
                0.60
                * empirical_delta
                + 0.40
                * hist_delta
            )

        elif pd.notna(
            empirical_delta
        ):

            delta = empirical_delta

        elif pd.notna(
            hist_delta
        ):

            delta = hist_delta

        else:

            delta = 0.0

        # ====================================================
        # REFUERZO DE CONDICIÓN SEVERA
        # ====================================================

        driver_score = 0.0

        driver_count = 0

        if (
            q95_rain > 0
            and rain_7d > 0
        ):

            driver_score += min(
                rain_7d
                / max(
                    q95_rain * 3.0,
                    EPS,
                ),
                2.0,
            )

            driver_count += 1

        if (
            pd.notna(q)
            and pd.notna(q95_flow)
            and q95_flow > 0
        ):

            driver_score += (
                q
                / q95_flow
            )

            driver_count += 1

        if (
            pd.notna(up)
            and pd.notna(q95_up)
            and q95_up > 0
        ):

            driver_score += (
                up
                / q95_up
            )

            driver_count += 1

        if driver_count > 0:

            driver_score = (
                driver_score
                / driver_count
            )

        # Si todos los drivers están en condición extrema
        # pero el ajuste lineal produce una bajante exagerada,
        # evitamos que ignore completamente el evento severo.

        if (
            driver_score >= 1.0
            and delta < -0.08
        ):

            delta = -0.08

        # ====================================================
        # LÍMITE DIARIO HISTÓRICO
        # ====================================================

        if response is not None:

            daily_limit = response[
                "daily_limit"
            ]

        else:

            delta_hist = (
                hist[
                    "nivel"
                ]
                .diff()
                .abs()
                .dropna()
            )

            if delta_hist.empty:

                daily_limit = 0.25

            else:

                daily_limit = float(
                    np.clip(
                        delta_hist.quantile(
                            0.995
                        ),
                        0.08,
                        0.60,
                    )
                )

        delta = float(
            np.clip(
                delta,
                -daily_limit,
                daily_limit,
            )
        )

        nivel_base = nivel

        nivel = float(
            np.clip(
                nivel
                + delta,
                LEVEL_MIN,
                LEVEL_MAX_INTERNAL,
            )
        )

        results.append(
            {

                "datetime":
                    future_dates[i],

                "prediction":
                    nivel,

                "nivel_base":
                    nivel_base,

                "variacion_dia":
                    delta,

                "precip_mm":
                    float(
                        rain
                        if pd.notna(rain)
                        else 0.0
                    ),

                "caudal_m3s":
                    (
                        float(q)
                        if pd.notna(q)
                        else np.nan
                    ),

                "upstream_mean":
                    (
                        float(up)
                        if pd.notna(up)
                        else np.nan
                    ),

                "severity_driver":
                    driver_score,

                "historical_source_date":
                    row[
                        "datetime"
                    ],
            }
        )

        prev_q = q

        prev_up = up

    return pd.DataFrame(
        results
    )


# ============================================================
# MÁXIMO / MÍNIMO HISTÓRICO PARA CADA FECHA
# ============================================================

def _envolvente_historica_diaria(
    df,
    future_dates,
):

    nivel = _preparar_nivel(
        df
    )

    if nivel.empty:

        return pd.DataFrame()

    nivel[
        "month"
    ] = (
        nivel[
            "datetime"
        ]
        .dt
        .month
    )

    nivel[
        "day"
    ] = (
        nivel[
            "datetime"
        ]
        .dt
        .day
    )

    env = (
        nivel
        .groupby(
            [
                "month",
                "day",
            ],
            as_index=False,
        )
        .agg(

            nivel_min_historico=(
                "nivel",
                "min",
            ),

            nivel_max_historico=(
                "nivel",
                "max",
            ),

            nivel_promedio_historico=(
                "nivel",
                "mean",
            ),

            registros=(
                "nivel",
                "count",
            ),
        )
    )

    future = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    future_dates,
                    errors="coerce",
                )
        }
    )

    future[
        "month"
    ] = (
        future[
            "datetime"
        ]
        .dt
        .month
    )

    future[
        "day"
    ] = (
        future[
            "datetime"
        ]
        .dt
        .day
    )

    return future.merge(
        env,
        on=[
            "month",
            "day",
        ],
        how="left",
    )


# ============================================================
# RENDER
# ============================================================

def render_stress_scenario(
    df,
    models=None,
    exog_history=None,
    upstream_history=None,
):

    st.subheader(
        "⚠️ Escenario de peor condición histórica · 60 días"
    )

    st.caption(
        "Simulación de estrés construida a partir de una "
        "secuencia histórica severa de nivel, lluvia, caudal "
        "y condiciones aguas arriba. No representa un "
        "pronóstico meteorológico convencional."
    )

    hist = _armar_historico(
        df=df,
        exog_history=exog_history,
        upstream_history=upstream_history,
    )

    if hist.empty:

        st.info(
            "No hay información histórica suficiente "
            "para calcular el escenario de 60 días."
        )

        return

    nivel_valid = (
        hist[
            "nivel"
        ]
        .dropna()
    )

    if nivel_valid.empty:

        st.info(
            "No existen niveles históricos válidos "
            "de San Nicolás."
        )

        return

    block = (
        _seleccionar_peor_bloque(
            hist,
            days=STRESS_DAYS,
        )
    )

    if block.empty:

        st.info(
            "No fue posible identificar "
            "una secuencia histórica severa."
        )

        return

    scenario = (
        _crear_escenario_60_dias(
            hist,
            block,
        )
    )

    if scenario.empty:

        st.info(
            "No fue posible calcular "
            "el escenario de 60 días."
        )

        return

    envelope = (
        _envolvente_historica_diaria(
            df,
            scenario[
                "datetime"
            ],
        )
    )

    if not envelope.empty:

        scenario = (
            scenario
            .merge(
                envelope[
                    [
                        "datetime",
                        "nivel_min_historico",
                        "nivel_max_historico",
                        "nivel_promedio_historico",
                        "registros",
                    ]
                ],
                on="datetime",
                how="left",
            )
        )

    # ========================================================
    # MÉTRICAS PRINCIPALES
    # ========================================================

    nivel_actual = float(
        nivel_valid.iloc[-1]
    )

    idx_max = (
        scenario[
            "prediction"
        ]
        .idxmax()
    )

    nivel_max_escenario = float(
        scenario.loc[
            idx_max,
            "prediction"
        ]
    )

    fecha_max_escenario = pd.to_datetime(
        scenario.loc[
            idx_max,
            "datetime"
        ]
    )

    crecimiento_maximo = (
        nivel_max_escenario
        - nivel_actual
    )

    nivel_final = float(
        scenario[
            "prediction"
        ]
        .iloc[-1]
    )

    max_hist_periodo = np.nan

    if (
        "nivel_max_historico"
        in scenario.columns
        and scenario[
            "nivel_max_historico"
        ]
        .notna()
        .any()
    ):

        max_hist_periodo = float(
            scenario[
                "nivel_max_historico"
            ]
            .max()
        )

    diferencia_max_hist = np.nan

    if pd.notna(
        max_hist_periodo
    ):

        diferencia_max_hist = (
            max_hist_periodo
            - nivel_max_escenario
        )

    # ========================================================
    # FILA 1
    # ========================================================

    st.markdown(
        "**Posición del escenario**"
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Nivel inicial",
        f"{nivel_actual:.2f} m",
    )

    c2.metric(
        "Máximo escenario",
        f"{nivel_max_escenario:.2f} m",
        f"{crecimiento_maximo:+.2f} m",
    )

    c3.metric(
        "Máximo histórico período",
        (
            f"{max_hist_periodo:.2f} m"
            if pd.notna(
                max_hist_periodo
            )
            else "--"
        ),
    )

    c4.metric(
        "Distancia al máximo histórico",
        (
            f"{diferencia_max_hist:.2f} m"
            if pd.notna(
                diferencia_max_hist
            )
            else "--"
        ),
    )

    # ========================================================
    # FILA 2
    # ========================================================

    d1, d2, d3, d4 = st.columns(
        4
    )

    d1.metric(
        "Nivel día 60",
        f"{nivel_final:.2f} m",
        f"{nivel_final - nivel_actual:+.2f} m",
    )

    d2.metric(
        "Crecimiento máximo",
        f"{crecimiento_maximo:+.2f} m",
    )

    d3.metric(
        "Fecha del máximo",
        fecha_max_escenario.strftime(
            "%d/%m/%Y"
        ),
    )

    block_start = (
        pd.to_datetime(
            block[
                "datetime"
            ].min()
        )
    )

    block_end = (
        pd.to_datetime(
            block[
                "datetime"
            ].max()
        )
    )

    d4.metric(
        "Secuencia histórica",
        (
            f"{block_start.strftime('%d/%m/%y')} "
            f"→ {block_end.strftime('%d/%m/%y')}"
        ),
    )

    # ========================================================
    # DRIVERS DEL ESCENARIO
    # ========================================================

    lluvia_acum = float(
        scenario[
            "precip_mm"
        ]
        .fillna(0.0)
        .sum()
    )

    lluvia_max = float(
        scenario[
            "precip_mm"
        ]
        .fillna(0.0)
        .max()
    )

    caudal_max = (
        float(
            scenario[
                "caudal_m3s"
            ]
            .max()
        )
        if scenario[
            "caudal_m3s"
        ]
        .notna()
        .any()
        else np.nan
    )

    upstream_max = (
        float(
            scenario[
                "upstream_mean"
            ]
            .max()
        )
        if scenario[
            "upstream_mean"
        ]
        .notna()
        .any()
        else np.nan
    )

    st.markdown(
        "**Variables de estrés utilizadas**"
    )

    v1, v2, v3, v4 = st.columns(
        4
    )

    v1.metric(
        "Lluvia acumulada 60 d",
        f"{_format_number(lluvia_acum, 1)} mm",
    )

    v2.metric(
        "Máxima lluvia diaria",
        f"{_format_number(lluvia_max, 1)} mm",
    )

    v3.metric(
        "Caudal máximo utilizado",
        (
            f"{_format_number(caudal_max, 0)} m³/s"
            if pd.notna(
                caudal_max
            )
            else "--"
        ),
    )

    v4.metric(
        "Nivel medio aguas arriba máx.",
        (
            f"{upstream_max:.2f} m"
            if pd.notna(
                upstream_max
            )
            else "--"
        ),
    )

    # ========================================================
    # GRÁFICO PRINCIPAL
    # ========================================================

    fig = go.Figure()

    # Banda histórica
    if (
        "nivel_max_historico"
        in scenario.columns
        and "nivel_min_historico"
        in scenario.columns
    ):

        fig.add_trace(
            go.Scatter(
                x=scenario[
                    "datetime"
                ],
                y=scenario[
                    "nivel_max_historico"
                ],
                mode="lines",
                name="Máximo histórico del día",
                line=dict(
                    color="#d62728",
                    width=2,
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Máximo histórico: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=scenario[
                    "datetime"
                ],
                y=scenario[
                    "nivel_min_historico"
                ],
                mode="lines",
                name="Mínimo histórico del día",
                line=dict(
                    color="#2ca02c",
                    width=2,
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Mínimo histórico: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    # Escenario
    fig.add_trace(
        go.Scatter(
            x=scenario[
                "datetime"
            ],
            y=scenario[
                "prediction"
            ],
            mode="lines+markers",
            name="Escenario severo 60 días",
            line=dict(
                color="#1f77b4",
                width=3,
            ),
            marker=dict(
                size=5,
            ),
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>"
                "Nivel escenario: %{y:.2f} m"
                "<extra></extra>"
            ),
        )
    )

    # Nivel real
    fig.add_hline(
        y=nivel_actual,
        line_dash="dash",
        line_width=2,
        annotation_text=(
            f"Nivel real de partida: "
            f"{nivel_actual:.2f} m"
        ),
    )

    # Máximo escenario
    fig.add_vline(
        x=fecha_max_escenario,
        line_dash="dot",
        line_width=1,
    )

    fig.add_annotation(
        x=fecha_max_escenario,
        y=nivel_max_escenario,
        text=(
            f"Máximo escenario<br>"
            f"{nivel_max_escenario:.2f} m"
        ),
        showarrow=True,
        arrowhead=2,
        yshift=20,
    )

    # Escala automática con máximo histórico visible
    valores_max = [
        nivel_actual,
        nivel_max_escenario,
    ]

    if pd.notna(
        max_hist_periodo
    ):

        valores_max.append(
            max_hist_periodo
        )

    y_top = min(
        LEVEL_MAX_INTERNAL,
        max(
            7.0,
            max(valores_max)
            + 0.5,
        ),
    )

    fig.update_layout(
        height=520,
        margin=dict(
            l=10,
            r=10,
            t=35,
            b=10,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.08,
        ),
    )

    fig.update_xaxes(
        title_text="Fecha",
        tickformat="%d/%m/%Y",
    )

    fig.update_yaxes(
        title_text="Nivel hidrométrico (m)",
        range=[
            LEVEL_MIN,
            y_top,
        ],
        dtick=0.5,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # DIAGNÓSTICO AUTOMÁTICO
    # ========================================================

    st.markdown(
        "**Lectura automática del escenario**"
    )

    if crecimiento_maximo > 0.50:

        st.warning(
            "El escenario histórico severo produce una "
            f"crecida máxima estimada de "
            f"**{crecimiento_maximo:+.2f} m**, "
            f"alcanzando **{nivel_max_escenario:.2f} m** "
            f"el {fecha_max_escenario.strftime('%d/%m/%Y')}."
        )

    elif crecimiento_maximo > 0.15:

        st.info(
            "El escenario genera una creciente moderada: "
            f"**{crecimiento_maximo:+.2f} m** respecto "
            "del nivel actual."
        )

    elif crecimiento_maximo >= 0:

        st.info(
            "Aun utilizando la secuencia histórica severa "
            "seleccionada, el modelo no proyecta una creciente "
            "significativa desde el nivel actual."
        )

    else:

        st.info(
            "La secuencia histórica seleccionada no produce "
            "una creciente respecto del nivel actual. Conviene "
            "revisar lluvia, caudal y niveles aguas arriba del "
            "bloque seleccionado."
        )

    if (
        pd.notna(
            max_hist_periodo
        )
        and nivel_max_escenario
        < max_hist_periodo
    ):

        st.caption(
            "El máximo histórico rojo es una referencia "
            "observada para cada fecha del calendario. "
            "El escenario azul no está obligado a alcanzar "
            "ese máximo: parte del nivel real actual y calcula "
            "la respuesta a los drivers severos seleccionados."
        )

    # ========================================================
    # LLUVIA
    # ========================================================

    rain_fig = go.Figure()

    rain_fig.add_trace(
        go.Bar(
            x=scenario[
                "datetime"
            ],
            y=scenario[
                "precip_mm"
            ],
            name="Lluvia escenario severo",
            marker_color="#17becf",
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>"
                "Lluvia: %{y:.1f} mm"
                "<extra></extra>"
            ),
        )
    )

    rain_fig.update_layout(
        height=280,
        margin=dict(
            l=5,
            r=5,
            t=10,
            b=5,
        ),
        yaxis_title="Precipitación (mm/día)",
        showlegend=False,
    )

    rain_fig.update_xaxes(
        tickformat="%d/%m",
    )

    st.plotly_chart(
        rain_fig,
        use_container_width=True,
    )

    # ========================================================
    # CAUDAL
    # ========================================================

    if (
        scenario[
            "caudal_m3s"
        ]
        .notna()
        .any()
    ):

        q_fig = go.Figure()

        q_fig.add_trace(
            go.Scatter(
                x=scenario[
                    "datetime"
                ],
                y=scenario[
                    "caudal_m3s"
                ],
                mode="lines+markers",
                name="Caudal escenario severo",
                line=dict(
                    color="#9467bd",
                    width=2,
                ),
                marker=dict(
                    size=4,
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Caudal: %{y:,.0f} m³/s"
                    "<extra></extra>"
                ),
            )
        )

        q_fig.update_layout(
            height=280,
            margin=dict(
                l=5,
                r=5,
                t=10,
                b=5,
            ),
            yaxis_title="Caudal (m³/s)",
            showlegend=False,
        )

        q_fig.update_xaxes(
            tickformat="%d/%m",
        )

        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )

    # ========================================================
    # TABLA DETALLADA
    # ========================================================

    with st.expander(
        "🔎 Auditoría diaria del escenario severo"
    ):

        tabla = scenario.copy()

        tabla[
            "Fecha"
        ] = (
            pd.to_datetime(
                tabla[
                    "datetime"
                ]
            )
            .dt
            .strftime(
                "%d/%m/%Y"
            )
        )

        tabla[
            "Nivel base (m)"
        ] = (
            tabla[
                "nivel_base"
            ]
            .round(2)
        )

        tabla[
            "Variación (m)"
        ] = (
            tabla[
                "variacion_dia"
            ]
            .round(3)
        )

        tabla[
            "Nivel escenario (m)"
        ] = (
            tabla[
                "prediction"
            ]
            .round(2)
        )

        tabla[
            "Lluvia (mm)"
        ] = (
            tabla[
                "precip_mm"
            ]
            .round(1)
        )

        tabla[
            "Caudal (m³/s)"
        ] = (
            tabla[
                "caudal_m3s"
            ]
            .round(0)
        )

        tabla[
            "Aguas arriba medio (m)"
        ] = (
            tabla[
                "upstream_mean"
            ]
            .round(2)
        )

        tabla[
            "Fecha histórica origen"
        ] = (
            pd.to_datetime(
                tabla[
                    "historical_source_date"
                ],
                errors="coerce",
            )
            .dt
            .strftime(
                "%d/%m/%Y"
            )
        )

        columnas = [

            "Fecha",
            "Nivel base (m)",
            "Lluvia (mm)",
            "Caudal (m³/s)",
            "Aguas arriba medio (m)",
            "Variación (m)",
            "Nivel escenario (m)",
            "Fecha histórica origen",
        ]

        if (
            "nivel_max_historico"
            in tabla.columns
        ):

            tabla[
                "Máximo histórico (m)"
            ] = (
                tabla[
                    "nivel_max_historico"
                ]
                .round(2)
            )

            columnas.append(
                "Máximo histórico (m)"
            )

        if (
            "nivel_min_historico"
            in tabla.columns
        ):

            tabla[
                "Mínimo histórico (m)"
            ] = (
                tabla[
                    "nivel_min_historico"
                ]
                .round(2)
            )

            columnas.append(
                "Mínimo histórico (m)"
            )

        st.dataframe(
            tabla[
                columnas
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # EXPLICACIÓN DEL BLOQUE HISTÓRICO
    # ========================================================

    with st.expander(
        "ℹ️ Cómo se determina la peor condición histórica"
    ):

        st.markdown(
            f"""
            El escenario comienza siempre en el **nivel real
            disponible de San Nicolás: {nivel_actual:.2f} m**.

            El sistema busca dentro del historial una secuencia
            continua de aproximadamente **60 días** que combine:

            - precipitaciones acumuladas elevadas;
            - caudales elevados;
            - niveles elevados aguas arriba;
            - comportamiento elevado de San Nicolás;
            - y, especialmente, períodos que históricamente
              produjeron una **crecida real**.

            La secuencia seleccionada corresponde aproximadamente a:

            **{block_start.strftime('%d/%m/%Y')} →
            {block_end.strftime('%d/%m/%Y')}**

            Después se aplica esa condición severa partiendo del
            **nivel actual**, por lo que la curva azul no reproduce
            literalmente el nivel histórico de aquel año.

            El máximo histórico rojo representa la mayor altura
            observada históricamente para cada día y mes y funciona
            como referencia de comparación.
            """
        )

    st.caption(
        "Escenario experimental de estrés histórico. "
        "No reemplaza pronósticos, avisos ni alertas oficiales."
    )
