import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/stress_ui.py
# V11.6
#
# ESCENARIO HIDROLÓGICO 60 DÍAS
#
# Objetivos:
# - Mantener compatibilidad con app.py actual
# - Generar escenario 60 días
# - Devolver DataFrame para gráfico unificado
# - Usar máximos históricos disponibles
# - Evitar precipitaciones máximas artificialmente en cero
# - Incorporar señal de caudal
# - Incorporar señal de niveles aguas arriba
#
# IMPORTANTE:
# Este módulo NO consulta INA directamente.
# Trabaja solamente con los datos que recibe.
# ============================================================


STRESS_DAYS = 60

Y_MIN = 0.0
Y_MAX = 7.0


# ============================================================
# UTILIDADES
# ============================================================


def _to_numeric(series):
    """
    Convierte una Serie a valores numéricos.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _to_datetime(series):
    """
    Convierte fechas a datetime sin zona horaria.
    """

    x = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    return x.dt.tz_localize(None)


def _safe_float(value, default=np.nan):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


# ============================================================
# IDENTIFICAR COLUMNA DE NIVEL
# ============================================================


def _get_level_column(df):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return None

    candidates = [
        "nivel",
        "value",
        "nivel_san_nicolas",
        "prediction",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    return None


# ============================================================
# CUANTIL SEGURO
# ============================================================


def _safe_quantile(
    values,
    q,
    default=0.0,
):

    if values is None:
        return default

    s = _to_numeric(
        pd.Series(values)
    ).dropna()

    if s.empty:
        return default

    try:

        value = float(
            s.quantile(q)
        )

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


# ============================================================
# PREPARAR NIVEL SAN NICOLÁS
# ============================================================


def _prepare_level_history(df):

    level_col = _get_level_column(
        df
    )

    if (
        level_col is None
        or "datetime" not in df.columns
    ):
        return pd.DataFrame()

    x = df[
        [
            "datetime",
            level_col,
        ]
    ].copy()

    x["datetime"] = _to_datetime(
        x["datetime"]
    )

    x["nivel"] = _to_numeric(
        x[level_col]
    )

    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    if x.empty:
        return pd.DataFrame()

    x["date"] = (
        x["datetime"]
        .dt.normalize()
    )

    x = (
        x.groupby(
            "date",
            as_index=False,
        )["nivel"]
        .mean()
        .rename(
            columns={
                "date":
                    "datetime"
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
# PREPARAR EXÓGENAS
# ============================================================


def _prepare_exogenous(
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

    x["datetime"] = _to_datetime(
        x["datetime"]
    )

    x = x.dropna(
        subset=[
            "datetime"
        ]
    )

    x["date"] = (
        x["datetime"]
        .dt.normalize()
    )

    agg = {}

    if "precip_mm" in x.columns:

        x["precip_mm"] = _to_numeric(
            x["precip_mm"]
        )

        agg["precip_mm"] = "sum"

    if "caudal_m3s" in x.columns:

        x["caudal_m3s"] = _to_numeric(
            x["caudal_m3s"]
        )

        agg["caudal_m3s"] = "mean"

    if not agg:
        return pd.DataFrame()

    x = (
        x.groupby(
            "date",
            as_index=False,
        )
        .agg(
            agg
        )
        .rename(
            columns={
                "date":
                    "datetime"
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
# MÁXIMO HISTÓRICO DE LLUVIA PARA FECHAS FUTURAS
# ============================================================


def _historical_rain_calendar(
    exog_history,
    future_dates,
):

    """
    Para cada fecha futura busca el máximo histórico
    registrado para el mismo día del año.

    Si no existe dato para ese día exacto:
    utiliza una ventana estacional de ±7 días.

    Si tampoco existe:
    utiliza un máximo histórico de referencia.

    Esto evita colocar automáticamente cero cuando
    simplemente falta información histórica.
    """

    exog = _prepare_exogenous(
        exog_history
    )

    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    future_dates
                )
        }
    )

    result[
        "rain_historical_max_mm"
    ] = 0.0

    if (
        exog.empty
        or "precip_mm"
        not in exog.columns
    ):
        return result

    rain = exog[
        [
            "datetime",
            "precip_mm",
        ]
    ].copy()

    rain["precip_mm"] = (
        _to_numeric(
            rain[
                "precip_mm"
            ]
        )
    )

    rain = rain.dropna(
        subset=[
            "precip_mm"
        ]
    )

    if rain.empty:
        return result

    rain["doy"] = (
        rain[
            "datetime"
        ].dt.dayofyear
    )

    global_reference = (
        _safe_quantile(
            rain[
                "precip_mm"
            ],
            0.95,
            default=0.0,
        )
    )

    values = []

    for future_date in result[
        "datetime"
    ]:

        doy = int(
            future_date.dayofyear
        )

        exact = rain[
            rain[
                "doy"
            ]
            == doy
        ][
            "precip_mm"
        ].dropna()

        if not exact.empty:

            value = float(
                exact.max()
            )

            values.append(
                max(
                    value,
                    0.0,
                )
            )

            continue

        # ----------------------------------------------------
        # Ventana estacional ±7 días
        # ----------------------------------------------------

        distance = np.minimum(
            np.abs(
                rain[
                    "doy"
                ]
                - doy
            ),
            365
            - np.abs(
                rain[
                    "doy"
                ]
                - doy
            ),
        )

        seasonal = rain[
            distance <= 7
        ][
            "precip_mm"
        ].dropna()

        if not seasonal.empty:

            value = float(
                seasonal.max()
            )

        else:

            value = (
                global_reference
            )

        values.append(
            max(
                _safe_float(
                    value,
                    0.0,
                ),
                0.0,
            )
        )

    result[
        "rain_historical_max_mm"
    ] = values

    return result


# ============================================================
# MÁXIMO HISTÓRICO DE CAUDAL PARA FECHAS FUTURAS
# ============================================================


def _historical_flow_calendar(
    exog_history,
    future_dates,
):

    """
    Busca caudal máximo histórico para el mismo
    día del año.

    Si no existe, utiliza ventana estacional ±10 días.
    """

    exog = _prepare_exogenous(
        exog_history
    )

    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    future_dates
                )
        }
    )

    result[
        "flow_historical_max_m3s"
    ] = np.nan

    if (
        exog.empty
        or "caudal_m3s"
        not in exog.columns
    ):
        return result

    q = exog[
        [
            "datetime",
            "caudal_m3s",
        ]
    ].copy()

    q["caudal_m3s"] = (
        _to_numeric(
            q[
                "caudal_m3s"
            ]
        )
    )

    q = q.dropna(
        subset=[
            "caudal_m3s"
        ]
    )

    if q.empty:
        return result

    q["doy"] = (
        q[
            "datetime"
        ].dt.dayofyear
    )

    global_reference = (
        _safe_quantile(
            q[
                "caudal_m3s"
            ],
            0.95,
            default=np.nan,
        )
    )

    values = []

    for future_date in result[
        "datetime"
    ]:

        doy = int(
            future_date.dayofyear
        )

        exact = q[
            q[
                "doy"
            ]
            == doy
        ][
            "caudal_m3s"
        ].dropna()

        if not exact.empty:

            values.append(
                float(
                    exact.max()
                )
            )

            continue

        distance = np.minimum(
            np.abs(
                q[
                    "doy"
                ]
                - doy
            ),
            365
            - np.abs(
                q[
                    "doy"
                ]
                - doy
            ),
        )

        seasonal = q[
            distance <= 10
        ][
            "caudal_m3s"
        ].dropna()

        if not seasonal.empty:

            value = float(
                seasonal.max()
            )

        else:

            value = (
                global_reference
            )

        values.append(
            _safe_float(
                value,
                np.nan,
            )
        )

    result[
        "flow_historical_max_m3s"
    ] = values

    return result


# ============================================================
# ENVOLVENTE HISTÓRICA DE NIVEL
# ============================================================


def _historical_level_envelope(
    df,
    future_dates,
):

    """
    Máximo histórico de San Nicolás para la misma
    época del año.

    No es el pronóstico.
    Es una referencia superior histórica.
    """

    level = _prepare_level_history(
        df
    )

    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    future_dates
                )
        }
    )

    result[
        "historical_level_max_m"
    ] = np.nan

    if level.empty:
        return result

    level["doy"] = (
        level[
            "datetime"
        ].dt.dayofyear
    )

    global_max = float(
        level[
            "nivel"
        ].max()
    )

    values = []

    for future_date in result[
        "datetime"
    ]:

        doy = int(
            future_date.dayofyear
        )

        exact = level[
            level[
                "doy"
            ]
            == doy
        ][
            "nivel"
        ].dropna()

        if not exact.empty:

            value = float(
                exact.max()
            )

        else:

            distance = np.minimum(
                np.abs(
                    level[
                        "doy"
                    ]
                    - doy
                ),
                365
                - np.abs(
                    level[
                        "doy"
                    ]
                    - doy
                ),
            )

            seasonal = level[
                distance <= 10
            ][
                "nivel"
            ].dropna()

            if not seasonal.empty:

                value = float(
                    seasonal.max()
                )

            else:

                value = global_max

        values.append(
            float(
                np.clip(
                    value,
                    Y_MIN,
                    Y_MAX,
                )
            )
        )

    result[
        "historical_level_max_m"
    ] = values

    return result


# ============================================================
# SEÑAL AGUAS ARRIBA
# ============================================================


def _upstream_signal(
    upstream_history,
):

    """
    Genera una señal agregada de tendencia de
    las estaciones aguas arriba.

    Valor positivo:
    predominio creciente.

    Valor negativo:
    predominio descendente.
    """

    if (
        upstream_history is None
        or not isinstance(
            upstream_history,
            pd.DataFrame,
        )
        or upstream_history.empty
    ):
        return 0.0

    level_cols = [
        col
        for col
        in upstream_history.columns
        if str(col).startswith(
            "nivel_"
        )
    ]

    if not level_cols:
        return 0.0

    signals = []

    for col in level_cols:

        values = (
            _to_numeric(
                upstream_history[
                    col
                ]
            )
            .dropna()
        )

        if len(values) < 2:
            continue

        # ----------------------------------------------------
        # Cambio reciente
        # ----------------------------------------------------

        if len(values) >= 7:

            recent = float(
                values.iloc[-1]
                - values.iloc[-7]
            )

        else:

            recent = float(
                values.iloc[-1]
                - values.iloc[0]
            )

        signals.append(
            recent
        )

    if not signals:
        return 0.0

    signal = float(
        np.nanmedian(
            signals
        )
    )

    return float(
        np.clip(
            signal,
            -1.5,
            1.5,
        )
    )


# ============================================================
# SEÑAL DE CAUDAL
# ============================================================


def _flow_signal(
    exog_history,
):

    exog = _prepare_exogenous(
        exog_history
    )

    if (
        exog.empty
        or "caudal_m3s"
        not in exog.columns
    ):
        return {
            "current":
                np.nan,

            "historical_high":
                np.nan,

            "ratio":
                1.0,

            "trend":
                0.0,
        }

    q = (
        _to_numeric(
            exog[
                "caudal_m3s"
            ]
        )
        .dropna()
    )

    if q.empty:

        return {
            "current":
                np.nan,

            "historical_high":
                np.nan,

            "ratio":
                1.0,

            "trend":
                0.0,
        }

    current = float(
        q.iloc[-1]
    )

    historical_high = (
        _safe_quantile(
            q,
            0.95,
            default=current,
        )
    )

    if (
        historical_high
        and historical_high > 0
    ):

        ratio = (
            current
            / historical_high
        )

    else:

        ratio = 1.0

    if len(q) >= 7:

        base = float(
            q.iloc[-7]
        )

        if base != 0:

            trend = (
                current
                - base
            ) / abs(base)

        else:

            trend = 0.0

    elif len(q) >= 2:

        base = float(
            q.iloc[0]
        )

        if base != 0:

            trend = (
                current
                - base
            ) / abs(base)

        else:

            trend = 0.0

    else:

        trend = 0.0

    return {
        "current":
            current,

        "historical_high":
            historical_high,

        "ratio":
            float(
                np.clip(
                    ratio,
                    0.2,
                    2.0,
                )
            ),

        "trend":
            float(
                np.clip(
                    trend,
                    -1.0,
                    1.0,
                )
            ),
    }


# ============================================================
# SEÑAL DE LLUVIA
# ============================================================


def _rain_signal(
    exog_history,
):

    exog = _prepare_exogenous(
        exog_history
    )

    if (
        exog.empty
        or "precip_mm"
        not in exog.columns
    ):
        return {
            "recent_7d":
                0.0,

            "recent_15d":
                0.0,

            "historical_high_15d":
                0.0,

            "ratio":
                0.0,
        }

    rain = (
        _to_numeric(
            exog[
                "precip_mm"
            ]
        )
        .fillna(
            0.0
        )
    )

    recent_7d = float(
        rain.tail(
            7
        ).sum()
    )

    recent_15d = float(
        rain.tail(
            15
        ).sum()
    )

    rolling_15 = (
        rain
        .rolling(
            15,
            min_periods=1,
        )
        .sum()
    )

    historical_high = (
        _safe_quantile(
            rolling_15,
            0.95,
            default=recent_15d,
        )
    )

    if historical_high > 0:

        ratio = (
            recent_15d
            / historical_high
        )

    else:

        ratio = 0.0

    return {
        "recent_7d":
            recent_7d,

        "recent_15d":
            recent_15d,

        "historical_high_15d":
            historical_high,

        "ratio":
            float(
                np.clip(
                    ratio,
                    0.0,
                    2.0,
                )
            ),
    }


# ============================================================
# PENDIENTE RECIENTE DEL NIVEL
# ============================================================


def _recent_level_slope(
    level_history,
):

    if (
        level_history is None
        or level_history.empty
    ):
        return 0.0

    values = (
        _to_numeric(
            level_history[
                "nivel"
            ]
        )
        .dropna()
        .tail(
            10
        )
    )

    if len(values) < 3:
        return 0.0

    y = values.to_numpy(
        dtype=float
    )

    x = np.arange(
        len(y),
        dtype=float,
    )

    try:

        slope = float(
            np.polyfit(
                x,
                y,
                1,
            )[0]
        )

    except Exception:

        slope = 0.0

    return float(
        np.clip(
            slope,
            -0.15,
            0.15,
        )
    )


# ============================================================
# CONSTRUIR ESCENARIO 60 DÍAS
# ============================================================


def build_stress_scenario(
    df,
    models=None,
    exog_history=None,
    upstream_history=None,
    days=STRESS_DAYS,
):
    """
    Construye y DEVUELVE la serie completa del escenario.

    Esta función no dibuja.

    Resultado:
    DataFrame con columnas:

    datetime
    stress_level
    historical_level_max_m
    rain_historical_max_mm
    flow_historical_max_m3s
    upstream_signal
    scenario_day
    """

    level_history = (
        _prepare_level_history(
            df
        )
    )

    if level_history.empty:
        return pd.DataFrame()

    days = int(
        max(
            1,
            days,
        )
    )

    last_date = pd.Timestamp(
        level_history[
            "datetime"
        ].iloc[-1]
    )

    current_level = float(
        level_history[
            "nivel"
        ].iloc[-1]
    )

    future_dates = pd.date_range(
        start=(
            last_date
            + pd.Timedelta(
                days=1
            )
        ),
        periods=days,
        freq="D",
    )


    # ========================================================
    # REFERENCIAS HISTÓRICAS
    # ========================================================

    level_envelope = (
        _historical_level_envelope(
            df,
            future_dates,
        )
    )

    rain_calendar = (
        _historical_rain_calendar(
            exog_history,
            future_dates,
        )
    )

    flow_calendar = (
        _historical_flow_calendar(
            exog_history,
            future_dates,
        )
    )


    scenario = pd.DataFrame(
        {
            "datetime":
                future_dates
        }
    )

    scenario = scenario.merge(
        level_envelope,
        on="datetime",
        how="left",
    )

    scenario = scenario.merge(
        rain_calendar,
        on="datetime",
        how="left",
    )

    scenario = scenario.merge(
        flow_calendar,
        on="datetime",
        how="left",
    )


    # ========================================================
    # SEÑALES ACTUALES
    # ========================================================

    upstream = _upstream_signal(
        upstream_history
    )

    flow = _flow_signal(
        exog_history
    )

    rain = _rain_signal(
        exog_history
    )

    local_slope = (
        _recent_level_slope(
            level_history
        )
    )


    # ========================================================
    # NORMALIZAR REFERENCIAS FUTURAS
    # ========================================================

    rain_values = (
        _to_numeric(
            scenario[
                "rain_historical_max_mm"
            ]
        )
        .fillna(
            0.0
        )
    )

    rain_scale = (
        _safe_quantile(
            rain_values,
            0.90,
            default=1.0,
        )
    )

    if (
        not np.isfinite(
            rain_scale
        )
        or rain_scale <= 0
    ):
        rain_scale = 1.0


    flow_values = (
        _to_numeric(
            scenario[
                "flow_historical_max_m3s"
            ]
        )
    )

    flow_reference = (
        _safe_quantile(
            flow_values,
            0.50,
            default=flow.get(
                "historical_high",
                np.nan,
            ),
        )
    )


    # ========================================================
    # CURVA DEL ESCENARIO
    # ========================================================

    levels = []

    previous_level = (
        current_level
    )

    for i, row in scenario.iterrows():

        day = i + 1


        # ----------------------------------------------------
        # 1. Persistencia de tendencia local
        # Se va amortiguando con el tiempo.
        # ----------------------------------------------------

        local_component = (
            local_slope
            * np.exp(
                -day / 15.0
            )
        )


        # ----------------------------------------------------
        # 2. Influencia aguas arriba
        #
        # No se aplica totalmente el primer día.
        # Aumenta gradualmente para simular propagación.
        # ----------------------------------------------------

        propagation_factor = (
            1.0
            - np.exp(
                -day / 9.0
            )
        )

        upstream_component = (
            0.025
            * upstream
            * propagation_factor
        )


        # ----------------------------------------------------
        # 3. Caudal
        # ----------------------------------------------------

        flow_component = 0.0

        q_future = _safe_float(
            row.get(
                "flow_historical_max_m3s"
            ),
            np.nan,
        )

        if (
            np.isfinite(
                q_future
            )
            and np.isfinite(
                flow_reference
            )
            and flow_reference > 0
        ):

            flow_ratio_future = (
                q_future
                / flow_reference
            )

            flow_component = (
                0.018
                * (
                    flow_ratio_future
                    - 1.0
                )
            )

        flow_component += (
            0.012
            * flow.get(
                "trend",
                0.0,
            )
        )


        # ----------------------------------------------------
        # 4. Lluvia histórica máxima estacional
        # ----------------------------------------------------

        rain_future = max(
            _safe_float(
                row.get(
                    "rain_historical_max_mm"
                ),
                0.0,
            ),
            0.0,
        )

        rain_ratio_future = (
            rain_future
            / rain_scale
        )

        rain_component = (
            0.012
            * np.clip(
                rain_ratio_future,
                0.0,
                3.0,
            )
        )


        # ----------------------------------------------------
        # 5. Lluvia reciente
        # ----------------------------------------------------

        recent_rain_component = (
            0.006
            * rain.get(
                "ratio",
                0.0,
            )
            * np.exp(
                -day / 10.0
            )
        )


        # ----------------------------------------------------
        # Cambio diario combinado
        # ----------------------------------------------------

        daily_change = (
            local_component
            + upstream_component
            + flow_component
            + rain_component
            + recent_rain_component
        )


        # ----------------------------------------------------
        # Límite de cambio diario para evitar saltos físicos
        # irreales en este escenario experimental.
        # ----------------------------------------------------

        daily_change = float(
            np.clip(
                daily_change,
                -0.12,
                0.16,
            )
        )


        proposed_level = (
            previous_level
            + daily_change
        )


        # ----------------------------------------------------
        # Referencia histórica de nivel
        #
        # Acercamos progresivamente la proyección al máximo
        # histórico estacional, pero no copiamos directamente
        # ese máximo.
        # ----------------------------------------------------

        historical_target = (
            _safe_float(
                row.get(
                    "historical_level_max_m"
                ),
                previous_level,
            )
        )

        if (
            historical_target
            > proposed_level
        ):

            historical_pull = (
                (
                    historical_target
                    - proposed_level
                )
                * 0.015
            )

            proposed_level += (
                historical_pull
            )


        # ----------------------------------------------------
        # Límite físico/gráfico
        # ----------------------------------------------------

        proposed_level = float(
            np.clip(
                proposed_level,
                Y_MIN,
                Y_MAX,
            )
        )

        levels.append(
            proposed_level
        )

        previous_level = (
            proposed_level
        )


    scenario[
        "stress_level"
    ] = levels

    scenario[
        "scenario_day"
    ] = np.arange(
        1,
        len(
            scenario
        ) + 1,
    )

    scenario[
        "upstream_signal"
    ] = upstream

    scenario[
        "current_flow_m3s"
    ] = flow.get(
        "current",
        np.nan,
    )

    scenario[
        "recent_rain_15d_mm"
    ] = rain.get(
        "recent_15d",
        0.0,
    )

    scenario[
        "scenario_type"
    ] = "historical_stress"

    return scenario


# ============================================================
# ALIAS PARA COMPATIBILIDAD FUTURA
# ============================================================


def get_stress_scenario(
    df,
    models=None,
    exog_history=None,
    upstream_history=None,
    days=STRESS_DAYS,
):

    return build_stress_scenario(
        df=df,
        models=models,
        exog_history=
            exog_history,
        upstream_history=
            upstream_history,
        days=days,
    )


# ============================================================
# RENDER DEL ESCENARIO
# ============================================================


def render_stress_scenario(
    df,
    models=None,
    exog_history=None,
    upstream_history=None,
    days=STRESS_DAYS,
):
    """
    Mantiene compatibilidad con app.py actual.

    Además de mostrar el escenario, DEVUELVE
    el DataFrame calculado.
    """

    st.subheader(
        "⚠️ Escenario hidrológico histórico · 60 días"
    )

    st.caption(
        "Escenario experimental que combina nivel actual, "
        "tendencia reciente, niveles aguas arriba, caudal "
        "y máximos históricos estacionales de precipitación "
        "y caudal."
    )


    scenario = build_stress_scenario(
        df=df,
        models=models,
        exog_history=
            exog_history,
        upstream_history=
            upstream_history,
        days=days,
    )


    if scenario.empty:

        st.info(
            "No hay suficientes datos para generar "
            "el escenario histórico de 60 días."
        )

        return scenario


    # ========================================================
    # MÉTRICAS
    # ========================================================

    level_history = (
        _prepare_level_history(
            df
        )
    )

    current_level = float(
        level_history[
            "nivel"
        ].iloc[-1]
    )

    projected_end = float(
        scenario[
            "stress_level"
        ].iloc[-1]
    )

    projected_max = float(
        scenario[
            "stress_level"
        ].max()
    )

    growth = (
        projected_end
        - current_level
    )


    c1, c2 = st.columns(
        2
    )

    c1.metric(
        "Nivel actual",
        f"{current_level:.2f} m",
    )

    c2.metric(
        "Nivel estimado día 60",
        f"{projected_end:.2f} m",
        f"{growth:+.2f} m",
    )


    c3, c4 = st.columns(
        2
    )

    c3.metric(
        "Máximo escenario",
        f"{projected_max:.2f} m",
    )

    upstream_signal = float(
        scenario[
            "upstream_signal"
        ].iloc[0]
    )

    if upstream_signal > 0.05:

        upstream_text = (
            "↑ Creciente"
        )

    elif upstream_signal < -0.05:

        upstream_text = (
            "↓ Descendente"
        )

    else:

        upstream_text = (
            "→ Estable"
        )

    c4.metric(
        "Señal aguas arriba",
        upstream_text,
    )


    # ========================================================
    # GRÁFICO
    # ========================================================

    fig = go.Figure()


    # --------------------------------------------------------
    # Nivel observado reciente
    # --------------------------------------------------------

    recent = (
        level_history.tail(
            60
        )
    )

    fig.add_trace(
        go.Scatter(
            x=recent[
                "datetime"
            ],
            y=recent[
                "nivel"
            ],
            mode="lines",
            name="Observado",
        )
    )


    # --------------------------------------------------------
    # Escenario
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=scenario[
                "datetime"
            ],
            y=scenario[
                "stress_level"
            ],
            mode="lines+markers",
            name="Escenario 60 días",
        )
    )


    # --------------------------------------------------------
    # Máximo histórico estacional
    # --------------------------------------------------------

    if (
        "historical_level_max_m"
        in scenario.columns
    ):

        fig.add_trace(
            go.Scatter(
                x=scenario[
                    "datetime"
                ],
                y=scenario[
                    "historical_level_max_m"
                ],
                mode="lines",
                line=dict(
                    dash="dot"
                ),
                name=(
                    "Máximo histórico "
                    "misma época"
                ),
            )
        )


    fig.update_layout(
        height=420,
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
        legend=dict(
            orientation="h",
            y=1.08,
        ),
    )

    fig.update_xaxes(
        tickformat="%d/%m",
    )

    fig.update_yaxes(
        title_text="Nivel (m)",
        range=[
            Y_MIN,
            Y_MAX,
        ],
        dtick=0.5,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # LLUVIAS MÁXIMAS HISTÓRICAS
    # ========================================================

    if (
        "rain_historical_max_mm"
        in scenario.columns
    ):

        with st.expander(
            "🌧️ Máximos históricos de precipitación utilizados"
        ):

            rain_table = scenario[
                [
                    "datetime",
                    "rain_historical_max_mm",
                ]
            ].copy()

            rain_table[
                "datetime"
            ] = (
                rain_table[
                    "datetime"
                ]
                .dt.strftime(
                    "%d/%m/%Y"
                )
            )

            rain_table = (
                rain_table.rename(
                    columns={
                        "datetime":
                            "Fecha",

                        "rain_historical_max_mm":
                            "Máx. histórico lluvia (mm)",
                    }
                )
            )

            rain_table[
                "Máx. histórico lluvia (mm)"
            ] = (
                rain_table[
                    "Máx. histórico lluvia (mm)"
                ]
                .round(
                    1
                )
            )

            st.dataframe(
                rain_table,
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # CAUDALES MÁXIMOS HISTÓRICOS
    # ========================================================

    if (
        "flow_historical_max_m3s"
        in scenario.columns
        and scenario[
            "flow_historical_max_m3s"
        ].notna().any()
    ):

        with st.expander(
            "💧 Máximos históricos de caudal utilizados"
        ):

            flow_table = scenario[
                [
                    "datetime",
                    "flow_historical_max_m3s",
                ]
            ].copy()

            flow_table[
                "datetime"
            ] = (
                flow_table[
                    "datetime"
                ]
                .dt.strftime(
                    "%d/%m/%Y"
                )
            )

            flow_table = (
                flow_table.rename(
                    columns={
                        "datetime":
                            "Fecha",

                        "flow_historical_max_m3s":
                            "Máx. histórico caudal (m³/s)",
                    }
                )
            )

            flow_table[
                "Máx. histórico caudal (m³/s)"
            ] = (
                flow_table[
                    "Máx. histórico caudal (m³/s)"
                ]
                .round(
                    0
                )
            )

            st.dataframe(
                flow_table,
                use_container_width=True,
                hide_index=True,
            )


    st.caption(
        "El escenario de 60 días es una simulación "
        "experimental basada en condiciones históricas "
        "y no constituye un pronóstico oficial."
    )


    # ========================================================
    # MUY IMPORTANTE
    #
    # Ahora app.py podrá recibir esta serie y utilizarla
    # dentro del gráfico único 15 + 30 + 60 días.
    # ========================================================

    return scenario
