import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import (
    date,
    timedelta,
    datetime,
)

from src.ina import observed

from src.model import (
    train,
    predict,
)

from src.exogenous import get_exogenous_data

from src.upstream import (
    get_upstream_history,
)

from src.stress_ui import (
    render_stress_scenario,
)

from src.hydrology import (
    analizar_corrientes_san_nicolas,
)


# ============================================================
# PARANÁ · SAN NICOLÁS
# APP V11.5 MOBILE + HIDROLOGÍA
#
# - Selector móvil visible
# - Pronóstico 15 días
# - Tendencia 30 días
# - Escenario 60 días
# - Relación histórica Corrientes -> San Nicolás
# - Retardo de propagación
# - Eventos máximos históricos
# ============================================================

APP_VERSION = "V11.5"

FORECAST_DAYS = 15
TREND_DAYS = 30

Y_MIN = 0.0
Y_MAX = 7.0
Y_STEP = 0.5


# ============================================================
# CONFIGURACIÓN STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.40rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.82rem;
    }

    div[data-testid="stAlert"] {
        padding-top: 0.55rem;
        padding-bottom: 0.55rem;
    }

    div[data-testid="stButton"] > button {
        min-height: 3rem;
        border-radius: 0.65rem;
        font-weight: 600;
    }

    .period-title {
        font-size: 1.15rem;
        font-weight: 650;
        margin-top: 0.25rem;
        margin-bottom: 0.15rem;
    }

    .period-help {
        font-size: 0.88rem;
        opacity: 0.75;
        margin-bottom: 0.5rem;
    }

    [data-testid="stDataFrame"] {
        overflow-x: auto;
    }

    @media (max-width: 768px) {

        .block-container {
            padding-top: 0.60rem !important;
            padding-left: 0.70rem !important;
            padding-right: 0.70rem !important;
            padding-bottom: 2rem !important;
        }

        h1 {
            font-size: 1.55rem !important;
            line-height: 1.15 !important;
            margin-bottom: 0.35rem !important;
        }

        h2 {
            font-size: 1.28rem !important;
        }

        h3 {
            font-size: 1.10rem !important;
        }

        p {
            font-size: 0.90rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.10rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.70rem !important;
        }

        [data-testid="stMetricDelta"] {
            font-size: 0.70rem !important;
        }

        div[data-testid="stButton"] > button {
            min-height: 3.2rem !important;
            font-size: 1rem !important;
            width: 100% !important;
        }

        div[data-testid="stDateInput"] input {
            font-size: 0.95rem !important;
        }

        div[data-testid="stAlert"] {
            font-size: 0.82rem !important;
        }

        .period-title {
            font-size: 1.05rem !important;
        }

        .period-help {
            font-size: 0.78rem !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ENCABEZADO
# ============================================================

st.title("🌊 PARANÁ · SAN NICOLÁS")

st.caption(
    f"{APP_VERSION} · Monitoreo y pronóstico experimental"
)

st.markdown(
    """
    Nivel del río Paraná en **San Nicolás de los Arroyos**,
    incorporando información hidrológica, precipitación,
    caudal y estaciones aguas arriba.
    """
)


# ============================================================
# SELECTOR DE PERÍODO
# ============================================================

st.markdown(
    '<div class="period-title">📅 Período de análisis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="period-help">
    Seleccioná las fechas y presioná Actualizar modelo.
    </div>
    """,
    unsafe_allow_html=True,
)


fecha_hasta_default = date.today()

fecha_desde_default = (
    fecha_hasta_default
    - timedelta(days=365)
)


fecha_col1, fecha_col2 = st.columns(
    2,
    gap="small",
)


with fecha_col1:

    desde = st.date_input(
        "Desde",
        value=fecha_desde_default,
        format="DD/MM/YYYY",
        key="fecha_desde_principal",
    )


with fecha_col2:

    hasta = st.date_input(
        "Hasta",
        value=fecha_hasta_default,
        format="DD/MM/YYYY",
        key="fecha_hasta_principal",
    )


actualizar = st.button(
    "🔄 Actualizar modelo",
    use_container_width=True,
    type="primary",
)


info1, info2, info3 = st.columns(3)

info1.caption(
    "Pronóstico: 15 días"
)

info2.caption(
    "Tendencia: 30 días"
)

info3.caption(
    "Escenario: 60 días"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌊 Paraná"
)

st.sidebar.subheader(
    "San Nicolás"
)

st.sidebar.caption(
    f"Versión {APP_VERSION}"
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Horizontes**

    • Pronóstico: 15 días  
    • Tendencia: 30 días  
    • Escenario histórico: 60 días
    """
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Análisis hidrológico**

    • Corrientes → San Nicolás  
    • Retardo histórico  
    • Máximos comparativos  
    • Caudal  
    • Precipitación
    """
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Fuentes**

    Nivel:
    INA A5

    Precipitación:
    Open-Meteo

    Variables hidrológicas:
    INA
    """
)

st.sidebar.caption(
    "Escala de nivel: 0 a 7 m"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def preparar_datos(df):

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

    x["datetime"] = pd.to_datetime(
        x["datetime"],
        errors="coerce",
        utc=True,
    )

    if "value" in x.columns:

        x["nivel"] = pd.to_numeric(
            x["value"],
            errors="coerce",
        )

    elif "nivel" in x.columns:

        x["nivel"] = pd.to_numeric(
            x["nivel"],
            errors="coerce",
        )

    else:

        return pd.DataFrame()

    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    x = (
        x
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

    return x


def texto_tendencia(
    delta,
):

    if delta is None:
        return "Sin comparación"

    try:
        delta = float(
            delta
        )
    except Exception:
        return "Sin comparación"

    if not np.isfinite(
        delta
    ):
        return "Sin comparación"

    if delta > 0.01:
        return "↑ Creciendo"

    if delta < -0.01:
        return "↓ Bajando"

    return "→ Estable"


def normalizar_estacion(
    texto,
):

    return (
        str(texto)
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace(" ", "_")
    )


# ============================================================
# TABLA AGUAS ARRIBA
# ============================================================

def construir_tabla_upstream(
    upstream_history,
    upstream_meta,
):

    rows = []

    if not isinstance(
        upstream_meta,
        dict,
    ):
        return pd.DataFrame()

    for station, info in upstream_meta.items():

        col = (
            "nivel_"
            + normalizar_estacion(
                station
            )
        )

        actual = None
        anterior = None
        variacion = None
        fecha_ultima = None

        if (
            isinstance(
                upstream_history,
                pd.DataFrame,
            )
            and not upstream_history.empty
            and col
            in upstream_history.columns
        ):

            temp = upstream_history[
                [
                    "datetime",
                    col,
                ]
            ].copy()

            temp[col] = pd.to_numeric(
                temp[col],
                errors="coerce",
            )

            temp = (
                temp
                .dropna(
                    subset=[col]
                )
                .sort_values(
                    "datetime"
                )
            )

            if not temp.empty:

                actual = float(
                    temp[col].iloc[-1]
                )

                fecha_ultima = (
                    temp[
                        "datetime"
                    ].iloc[-1]
                )

                if len(temp) >= 2:

                    anterior = float(
                        temp[col].iloc[-2]
                    )

                    variacion = (
                        actual
                        - anterior
                    )

        series_id = None
        status = None

        if isinstance(
            info,
            dict,
        ):

            series_id = info.get(
                "series_id"
            )

            status = info.get(
                "status"
            )

        rows.append(
            {
                "Estación":
                    station,

                "Nivel actual":
                    (
                        f"{actual:.2f} m"
                        if actual is not None
                        else "—"
                    ),

                "Variación":
                    (
                        f"{variacion:+.2f} m"
                        if variacion is not None
                        else "—"
                    ),

                "Tendencia":
                    (
                        texto_tendencia(
                            variacion
                        )
                        if actual is not None
                        else "Sin datos"
                    ),

                "Fecha":
                    (
                        pd.to_datetime(
                            fecha_ultima
                        ).strftime(
                            "%d/%m/%Y"
                        )
                        if fecha_ultima
                        is not None
                        else "—"
                    ),

                "Serie":
                    (
                        series_id
                        if series_id is not None
                        else "—"
                    ),

                "Estado":
                    (
                        status
                        if status
                        else "—"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# EXTENDER PRONÓSTICO 30 DÍAS
# ============================================================

def extender_pronostico_30(
    forecast,
    df,
):

    if (
        forecast is None
        or not isinstance(
            forecast,
            pd.DataFrame,
        )
        or forecast.empty
    ):
        return pd.DataFrame()

    f = forecast.copy()

    f["datetime"] = pd.to_datetime(
        f["datetime"],
        errors="coerce",
    )

    f["prediction"] = pd.to_numeric(
        f["prediction"],
        errors="coerce",
    )

    f = f.dropna(
        subset=[
            "datetime",
            "prediction",
        ]
    )

    if f.empty:
        return pd.DataFrame()

    result = f.copy()

    last_date = (
        f["datetime"].iloc[-1]
    )

    last_level = float(
        f["prediction"].iloc[-1]
    )

    if len(f) >= 5:

        recent = (
            f["prediction"]
            .tail(5)
            .to_numpy(
                dtype=float
            )
        )

        slope = float(
            np.polyfit(
                np.arange(
                    len(recent)
                ),
                recent,
                1,
            )[0]
        )

    else:

        recent = (
            pd.to_numeric(
                df["nivel"],
                errors="coerce",
            )
            .dropna()
            .tail(7)
        )

        if len(recent) >= 3:

            slope = float(
                np.polyfit(
                    np.arange(
                        len(recent)
                    ),
                    recent.to_numpy(
                        dtype=float
                    ),
                    1,
                )[0]
            )

        else:

            slope = 0.0

    slope = float(
        np.clip(
            slope,
            -0.10,
            0.10,
        )
    )

    extra = []

    for step in range(
        FORECAST_DAYS + 1,
        TREND_DAYS + 1,
    ):

        damping = np.exp(
            -0.14
            * (
                step
                - FORECAST_DAYS
            )
        )

        daily_change = (
            slope
            * damping
        )

        last_level = float(
            np.clip(
                last_level
                + daily_change,
                Y_MIN,
                Y_MAX,
            )
        )

        last_date = (
            last_date
            + pd.Timedelta(
                days=1
            )
        )

        extra.append(
            {
                "datetime":
                    last_date,

                "prediction":
                    last_level,

                "lower":
                    np.nan,

                "upper":
                    np.nan,
            }
        )

    if extra:

        result = pd.concat(
            [
                result,
                pd.DataFrame(
                    extra
                ),
            ],
            ignore_index=True,
        )

    return result


# ============================================================
# PREPARAR EVENTOS PARA MOSTRAR
# ============================================================

def preparar_tabla_eventos(
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
        return pd.DataFrame()

    tabla = events.copy()

    rename = {
        "fecha_max_corrientes":
            "Máx. Corrientes",

        "max_corrientes_m":
            "Corrientes (m)",

        "fecha_max_san_nicolas":
            "Máx. San Nicolás",

        "max_san_nicolas_m":
            "San Nicolás (m)",

        "lag_real_dias":
            "Retardo (días)",

        "nivel_base_san_nicolas_m":
            "Nivel base SN (m)",

        "respuesta_san_nicolas_m":
            "Crecimiento SN (m)",

        "lluvia_previa_mm":
            "Lluvia previa (mm)",

        "caudal_medio_m3s":
            "Caudal medio (m³/s)",

        "caudal_max_m3s":
            "Caudal máx. (m³/s)",
    }

    tabla = tabla.rename(
        columns=rename
    )

    for col in [
        "Máx. Corrientes",
        "Máx. San Nicolás",
    ]:

        if col in tabla.columns:

            tabla[col] = pd.to_datetime(
                tabla[col],
                errors="coerce",
            ).dt.strftime(
                "%d/%m/%Y"
            )

    for col in [
        "Corrientes (m)",
        "San Nicolás (m)",
        "Nivel base SN (m)",
        "Crecimiento SN (m)",
    ]:

        if col in tabla.columns:

            tabla[col] = pd.to_numeric(
                tabla[col],
                errors="coerce",
            ).round(
                2
            )

    for col in [
        "Lluvia previa (mm)",
        "Caudal medio (m³/s)",
        "Caudal máx. (m³/s)",
    ]:

        if col in tabla.columns:

            tabla[col] = pd.to_numeric(
                tabla[col],
                errors="coerce",
            ).round(
                1
            )

    return tabla


# ============================================================
# VALIDACIÓN FECHAS
# ============================================================

if desde > hasta:

    st.error(
        "⚠️ La fecha inicial no puede ser posterior "
        "a la fecha final."
    )


# ============================================================
# ACTUALIZACIÓN
# ============================================================

if actualizar:

    if desde > hasta:

        st.error(
            "El período seleccionado no es válido."
        )

    else:

        inicio = desde.strftime(
            "%Y-%m-%d"
        )

        fin = hasta.strftime(
            "%Y-%m-%d"
        )


        # ====================================================
        # SAN NICOLÁS
        # ====================================================

        with st.spinner(
            "Consultando nivel de San Nicolás..."
        ):

            try:

                (
                    df_ina,
                    error_ina,
                ) = observed(
                    inicio,
                    fin,
                )

            except Exception as exc:

                df_ina = pd.DataFrame()

                error_ina = str(
                    exc
                )


        if error_ina:

            st.error(
                error_ina
            )

        else:

            df = preparar_datos(
                df_ina
            )

            if df.empty:

                st.error(
                    "No se obtuvieron niveles válidos "
                    "de San Nicolás."
                )

            else:

                # ============================================
                # LLUVIA + CAUDAL
                # ============================================

                with st.spinner(
                    "Consultando precipitación y caudal..."
                ):

                    try:

                        (
                            exog_history,
                            exog_future,
                            exog_meta,
                        ) = get_exogenous_data(
                            inicio,
                            fin,
                            FORECAST_DAYS,
                        )

                    except Exception as exc:

                        exog_history = pd.DataFrame()

                        exog_future = pd.DataFrame()

                        exog_meta = {}

                        st.warning(
                            "Variables externas parcialmente "
                            f"disponibles: {exc}"
                        )


                # ============================================
                # AGUAS ARRIBA
                # ============================================

                with st.spinner(
                    "Consultando niveles aguas arriba..."
                ):

                    try:

                        upstream_result = (
                            get_upstream_history(
                                inicio,
                                fin,
                            )
                        )

                        if (
                            isinstance(
                                upstream_result,
                                tuple,
                            )
                            and len(
                                upstream_result
                            ) >= 2
                        ):

                            upstream_history = (
                                upstream_result[0]
                            )

                            upstream_meta = (
                                upstream_result[1]
                            )

                        else:

                            upstream_history = (
                                upstream_result
                                if isinstance(
                                    upstream_result,
                                    pd.DataFrame,
                                )
                                else pd.DataFrame()
                            )

                            upstream_meta = {}

                    except Exception as exc:

                        upstream_history = (
                            pd.DataFrame()
                        )

                        upstream_meta = {}

                        st.warning(
                            "No fue posible recuperar "
                            "todas las estaciones aguas arriba. "
                            f"Detalle: {exc}"
                        )


                # ============================================
                # ANÁLISIS HIDROLÓGICO
                # ============================================

                with st.spinner(
                    "Analizando relación Corrientes → San Nicolás..."
                ):

                    try:

                        hydro_analysis = (
                            analizar_corrientes_san_nicolas(
                                san_nicolas=df,
                                upstream_history=
                                    upstream_history,
                                exog_history=
                                    exog_history,
                                max_lag=20,
                            )
                        )

                    except Exception as exc:

                        hydro_analysis = {}

                        st.warning(
                            "No fue posible completar "
                            "el análisis histórico "
                            "Corrientes → San Nicolás. "
                            f"Detalle: {exc}"
                        )


                # ============================================
                # MODELO ACTUAL
                #
                # IMPORTANTE:
                # todavía NO incorporamos hydrology.py
                # al Random Forest.
                # Primero validamos la relación.
                # ============================================

                with st.spinner(
                    "Entrenando modelo..."
                ):

                    try:

                        (
                            models,
                            metrics,
                        ) = train(
                            df=df,
                            exog_history=
                                exog_history,
                            upstream_history=
                                upstream_history,
                        )

                        forecast = predict(
                            df=df,
                            models=models,
                            days=FORECAST_DAYS,
                            exog_future=
                                exog_future,
                        )

                        forecast30 = (
                            extender_pronostico_30(
                                forecast,
                                df,
                            )
                        )

                    except Exception as exc:

                        models = {}

                        metrics = {}

                        forecast = (
                            pd.DataFrame()
                        )

                        forecast30 = (
                            pd.DataFrame()
                        )

                        st.error(
                            "No fue posible generar el modelo. "
                            f"Detalle: {exc}"
                        )


                # ============================================
                # SESSION STATE
                # ============================================

                st.session_state[
                    "datos"
                ] = df

                st.session_state[
                    "forecast"
                ] = forecast

                st.session_state[
                    "forecast30"
                ] = forecast30

                st.session_state[
                    "models"
                ] = models

                st.session_state[
                    "metrics"
                ] = metrics

                st.session_state[
                    "exog_history"
                ] = exog_history

                st.session_state[
                    "exog_future"
                ] = exog_future

                st.session_state[
                    "exog_meta"
                ] = exog_meta

                st.session_state[
                    "upstream_history"
                ] = upstream_history

                st.session_state[
                    "upstream_meta"
                ] = upstream_meta

                st.session_state[
                    "hydro_analysis"
                ] = hydro_analysis

                st.session_state[
                    "actualizado"
                ] = datetime.now()


                st.success(
                    "✅ Datos y modelo actualizados correctamente."
                )


# ============================================================
# ESTADO INICIAL
# ============================================================

if "datos" not in st.session_state:

    st.info(
        "👆 Seleccioná el período arriba y presioná "
        "**Actualizar modelo**."
    )


# ============================================================
# RESULTADOS
# ============================================================

else:

    df = st.session_state.get(
        "datos",
        pd.DataFrame(),
    )

    forecast = st.session_state.get(
        "forecast",
        pd.DataFrame(),
    )

    forecast30 = st.session_state.get(
        "forecast30",
        pd.DataFrame(),
    )

    models = st.session_state.get(
        "models",
        {},
    )

    metrics = st.session_state.get(
        "metrics",
        {},
    )

    exog_history = st.session_state.get(
        "exog_history",
        pd.DataFrame(),
    )

    exog_future = st.session_state.get(
        "exog_future",
        pd.DataFrame(),
    )

    upstream_history = st.session_state.get(
        "upstream_history",
        pd.DataFrame(),
    )

    upstream_meta = st.session_state.get(
        "upstream_meta",
        {},
    )

    hydro_analysis = st.session_state.get(
        "hydro_analysis",
        {},
    )

    actualizado = st.session_state.get(
        "actualizado"
    )


    # ========================================================
    # SITUACIÓN SAN NICOLÁS
    # ========================================================

    niveles = pd.to_numeric(
        df["nivel"],
        errors="coerce",
    ).dropna()

    if niveles.empty:

        st.error(
            "No hay niveles válidos de San Nicolás."
        )

        st.stop()


    nivel_actual = float(
        niveles.iloc[-1]
    )

    delta_actual = None

    if len(niveles) >= 2:

        delta_actual = (
            nivel_actual
            - float(
                niveles.iloc[-2]
            )
        )

    ultima_fecha = (
        df["datetime"].iloc[-1]
    )


    st.subheader(
        "📊 Situación actual"
    )

    c1, c2 = st.columns(
        2
    )

    c1.metric(
        "Nivel actual",
        f"{nivel_actual:.2f} m",
        (
            f"{delta_actual:+.2f} m"
            if delta_actual is not None
            else None
        ),
    )

    c2.metric(
        "Tendencia",
        texto_tendencia(
            delta_actual
        ),
    )

    c3, c4, c5 = st.columns(
        3
    )

    c3.metric(
        "Mínimo",
        f"{niveles.min():.2f} m",
    )

    c4.metric(
        "Promedio",
        f"{niveles.mean():.2f} m",
    )

    c5.metric(
        "Máximo",
        f"{niveles.max():.2f} m",
    )

    st.caption(
        "Última observación INA: "
        + pd.to_datetime(
            ultima_fecha
        ).strftime(
            "%d/%m/%Y %H:%M"
        )
    )


    # ========================================================
    # PRONÓSTICO 15 DÍAS
    # ========================================================

    st.subheader(
        "📈 Pronóstico · 15 días"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["nivel"],
            mode="lines",
            name="Observado",
        )
    )

    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
    ):

        fig.add_trace(
            go.Scatter(
                x=forecast[
                    "datetime"
                ],
                y=forecast[
                    "prediction"
                ],
                mode="lines+markers",
                name="Pronóstico",
            )
        )

        if (
            "upper" in forecast.columns
            and "lower"
            in forecast.columns
        ):

            fig.add_trace(
                go.Scatter(
                    x=forecast[
                        "datetime"
                    ],
                    y=forecast[
                        "upper"
                    ],
                    mode="lines",
                    line=dict(
                        width=0
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast[
                        "datetime"
                    ],
                    y=forecast[
                        "lower"
                    ],
                    mode="lines",
                    line=dict(
                        width=0
                    ),
                    fill="tonexty",
                    name="Incertidumbre",
                )
            )

    fig.update_layout(
        height=420,
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=35,
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
        dtick=Y_STEP,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # ESTACIONES AGUAS ARRIBA
    # ========================================================

    st.subheader(
        "🌊 Estaciones aguas arriba"
    )

    tabla_upstream = (
        construir_tabla_upstream(
            upstream_history,
            upstream_meta,
        )
    )

    if not tabla_upstream.empty:

        st.dataframe(
            tabla_upstream,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Todavía no se recuperaron niveles "
            "de las estaciones aguas arriba."
        )


    # ========================================================
    # CORRIENTES -> SAN NICOLÁS
    # ========================================================

    st.subheader(
        "🌊 Corrientes → San Nicolás · comportamiento histórico"
    )

    st.caption(
        "El análisis busca cuánto tiempo suele transcurrir "
        "entre una señal en Corrientes y la respuesta observada "
        "posteriormente en San Nicolás."
    )


    lag_info = {}

    events = pd.DataFrame()

    hydro_stats = {}

    top_events = pd.DataFrame()


    if isinstance(
        hydro_analysis,
        dict,
    ):

        lag_info = hydro_analysis.get(
            "lag",
            {},
        )

        events = hydro_analysis.get(
            "events",
            pd.DataFrame(),
        )

        hydro_stats = hydro_analysis.get(
            "statistics",
            {},
        )

        top_events = hydro_analysis.get(
            "top_events",
            pd.DataFrame(),
        )


    best_lag = None
    correlation = np.nan

    if isinstance(
        lag_info,
        dict,
    ):

        best_lag = lag_info.get(
            "best_lag_days"
        )

        correlation = lag_info.get(
            "correlation",
            np.nan,
        )


    if (
        best_lag is not None
        and np.isfinite(
            correlation
        )
    ):

        h1, h2, h3 = st.columns(
            3
        )

        h1.metric(
            "Retardo estadístico",
            f"{int(best_lag)} días",
        )

        h2.metric(
            "Correlación",
            f"{float(correlation):.2f}",
        )

        h3.metric(
            "Eventos detectados",
            int(
                hydro_stats.get(
                    "events",
                    len(events),
                )
            ),
        )

    else:

        st.info(
            "Todavía no hay suficientes datos de Corrientes "
            "para calcular una relación histórica confiable."
        )


    # ========================================================
    # ESTADÍSTICAS DE EVENTOS
    # ========================================================

    if (
        isinstance(
            hydro_stats,
            dict,
        )
        and hydro_stats.get(
            "events",
            0,
        ) > 0
    ):

        median_lag = (
            hydro_stats.get(
                "median_lag_days"
            )
        )

        response = (
            hydro_stats.get(
                "median_response_m"
            )
        )

        corr_max = (
            hydro_stats.get(
                "correlation_maxima"
            )
        )

        s1, s2, s3 = st.columns(
            3
        )

        s1.metric(
            "Retardo mediano eventos",
            (
                f"{median_lag:.1f} días"
                if median_lag
                is not None
                else "—"
            ),
        )

        s2.metric(
            "Respuesta mediana SN",
            (
                f"{response:+.2f} m"
                if response
                is not None
                else "—"
            ),
        )

        s3.metric(
            "Correlación de máximos",
            (
                f"{corr_max:.2f}"
                if corr_max is not None
                and np.isfinite(
                    corr_max
                )
                else "—"
            ),
        )


    # ========================================================
    # GRÁFICO COMPARATIVO CORRIENTES + SAN NICOLÁS
    # ========================================================

    if (
        isinstance(
            upstream_history,
            pd.DataFrame,
        )
        and not upstream_history.empty
        and "nivel_corrientes"
        in upstream_history.columns
    ):

        corrientes_plot = (
            upstream_history[
                [
                    "datetime",
                    "nivel_corrientes",
                ]
            ]
            .copy()
        )

        corrientes_plot[
            "datetime"
        ] = pd.to_datetime(
            corrientes_plot[
                "datetime"
            ],
            errors="coerce",
            utc=True,
        )

        corrientes_plot[
            "nivel_corrientes"
        ] = pd.to_numeric(
            corrientes_plot[
                "nivel_corrientes"
            ],
            errors="coerce",
        )

        corrientes_plot = (
            corrientes_plot.dropna()
        )


        relation_fig = go.Figure()


        relation_fig.add_trace(
            go.Scatter(
                x=corrientes_plot[
                    "datetime"
                ],
                y=corrientes_plot[
                    "nivel_corrientes"
                ],
                mode="lines",
                name="Corrientes",
            )
        )


        relation_fig.add_trace(
            go.Scatter(
                x=df[
                    "datetime"
                ],
                y=df[
                    "nivel"
                ],
                mode="lines",
                name="San Nicolás",
            )
        )


        relation_fig.update_layout(
            height=430,
            hovermode="x unified",
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
        )


        relation_fig.update_yaxes(
            title_text="Nivel (m)",
            range=[
                Y_MIN,
                Y_MAX,
            ],
            dtick=Y_STEP,
        )


        st.plotly_chart(
            relation_fig,
            use_container_width=True,
        )


        # ====================================================
        # GRÁFICO CORRIENTES DESPLAZADO
        # ====================================================

        if best_lag is not None:

            st.markdown(
                "#### Propagación temporal estimada"
            )

            shifted = (
                corrientes_plot.copy()
            )

            shifted[
                "datetime_propagado"
            ] = (
                shifted[
                    "datetime"
                ]
                + pd.to_timedelta(
                    int(best_lag),
                    unit="D",
                )
            )


            propagation_fig = go.Figure()


            propagation_fig.add_trace(
                go.Scatter(
                    x=shifted[
                        "datetime_propagado"
                    ],
                    y=shifted[
                        "nivel_corrientes"
                    ],
                    mode="lines",
                    name=(
                        "Corrientes desplazado "
                        f"+{best_lag} días"
                    ),
                )
            )


            propagation_fig.add_trace(
                go.Scatter(
                    x=df[
                        "datetime"
                    ],
                    y=df[
                        "nivel"
                    ],
                    mode="lines",
                    name="San Nicolás",
                )
            )


            propagation_fig.update_layout(
                height=400,
                hovermode="x unified",
                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=10,
                ),
            )


            propagation_fig.update_yaxes(
                title_text="Nivel (m)",
                range=[
                    Y_MIN,
                    Y_MAX,
                ],
                dtick=Y_STEP,
            )


            st.plotly_chart(
                propagation_fig,
                use_container_width=True,
            )


    # ========================================================
    # EVENTOS MÁXIMOS
    # ========================================================

    st.markdown(
        "#### Máximos históricos comparativos"
    )

    st.caption(
        "Cada fila representa un máximo detectado en Corrientes "
        "y el máximo observado posteriormente en San Nicolás."
    )


    tabla_eventos = (
        preparar_tabla_eventos(
            top_events
        )
    )


    if not tabla_eventos.empty:

        st.dataframe(
            tabla_eventos,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Todavía no hay eventos históricos suficientes "
            "para construir la comparación de máximos."
        )


    # ========================================================
    # GRÁFICO DE MÁXIMOS
    # ========================================================

    if (
        isinstance(
            events,
            pd.DataFrame,
        )
        and not events.empty
        and "max_corrientes_m"
        in events.columns
        and "max_san_nicolas_m"
        in events.columns
    ):

        scatter_events = events.copy()

        scatter_events[
            "max_corrientes_m"
        ] = pd.to_numeric(
            scatter_events[
                "max_corrientes_m"
            ],
            errors="coerce",
        )

        scatter_events[
            "max_san_nicolas_m"
        ] = pd.to_numeric(
            scatter_events[
                "max_san_nicolas_m"
            ],
            errors="coerce",
        )

        scatter_events = (
            scatter_events.dropna(
                subset=[
                    "max_corrientes_m",
                    "max_san_nicolas_m",
                ]
            )
        )


        if len(
            scatter_events
        ) >= 2:

            st.markdown(
                "#### Relación entre máximos"
            )

            scatter_fig = go.Figure()

            scatter_fig.add_trace(
                go.Scatter(
                    x=scatter_events[
                        "max_corrientes_m"
                    ],
                    y=scatter_events[
                        "max_san_nicolas_m"
                    ],
                    mode="markers",
                    name="Eventos históricos",
                    customdata=(
                        scatter_events[
                            "lag_real_dias"
                        ]
                        if "lag_real_dias"
                        in scatter_events.columns
                        else None
                    ),
                    hovertemplate=(
                        "Corrientes: %{x:.2f} m<br>"
                        "San Nicolás: %{y:.2f} m<br>"
                        "<extra></extra>"
                    ),
                )
            )


            if len(
                scatter_events
            ) >= 3:

                x_values = (
                    scatter_events[
                        "max_corrientes_m"
                    ].to_numpy(
                        dtype=float
                    )
                )

                y_values = (
                    scatter_events[
                        "max_san_nicolas_m"
                    ].to_numpy(
                        dtype=float
                    )
                )

                try:

                    slope, intercept = (
                        np.polyfit(
                            x_values,
                            y_values,
                            1,
                        )
                    )

                    xx = np.linspace(
                        x_values.min(),
                        x_values.max(),
                        100,
                    )

                    yy = (
                        slope
                        * xx
                        + intercept
                    )

                    scatter_fig.add_trace(
                        go.Scatter(
                            x=xx,
                            y=yy,
                            mode="lines",
                            name="Relación histórica",
                        )
                    )

                except Exception:

                    pass


            scatter_fig.update_layout(
                height=400,
                xaxis_title=
                    "Máximo Corrientes (m)",
                yaxis_title=
                    "Máximo San Nicolás (m)",
                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=10,
                ),
            )


            st.plotly_chart(
                scatter_fig,
                use_container_width=True,
            )


    # ========================================================
    # PRÓXIMA ETAPA DEL MODELO
    # ========================================================

    with st.expander(
        "🧠 Cómo se utilizará este análisis en el pronóstico"
    ):

        st.markdown(
            """
            El análisis Corrientes → San Nicolás todavía se utiliza
            como **diagnóstico histórico**.

            Una vez validada la calidad de estos resultados,
            el modelo de 15 días incorporará:

            - nivel actual de Corrientes;
            - crecimiento de Corrientes en 1, 3 y 7 días;
            - retardo histórico de propagación;
            - nivel y tendencia de las demás estaciones;
            - caudal diario disponible;
            - tendencia del caudal;
            - lluvia observada reciente;
            - lluvia estimada para los próximos 15 días;
            - similitud con crecidas históricas anteriores.

            De esta manera el pronóstico no dependerá solamente
            de la tendencia reciente de San Nicolás.
            """
        )


    # ========================================================
    # TENDENCIA 30 DÍAS
    # ========================================================

    st.subheader(
        "📆 Tendencia extendida · 30 días"
    )

    if (
        isinstance(
            forecast30,
            pd.DataFrame,
        )
        and not forecast30.empty
    ):

        trend_fig = go.Figure()

        recent = df.tail(
            30
        )

        trend_fig.add_trace(
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

        f15 = forecast30.head(
            FORECAST_DAYS
        )

        trend_fig.add_trace(
            go.Scatter(
                x=f15[
                    "datetime"
                ],
                y=f15[
                    "prediction"
                ],
                mode="lines+markers",
                name="Pronóstico 1–15 días",
            )
        )

        f16_30 = forecast30.iloc[
            FORECAST_DAYS:
        ]

        if not f16_30.empty:

            trend_fig.add_trace(
                go.Scatter(
                    x=f16_30[
                        "datetime"
                    ],
                    y=f16_30[
                        "prediction"
                    ],
                    mode="lines+markers",
                    line=dict(
                        dash="dot"
                    ),
                    name="Tendencia 16–30 días",
                )
            )

        trend_fig.update_layout(
            height=390,
            hovermode="x unified",
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
        )

        trend_fig.update_yaxes(
            title_text="Nivel (m)",
            range=[
                Y_MIN,
                Y_MAX,
            ],
            dtick=Y_STEP,
        )

        st.plotly_chart(
            trend_fig,
            use_container_width=True,
        )


    # ========================================================
    # LLUVIA 15 DÍAS
    # ========================================================

    st.subheader(
        "🌧️ Precipitación prevista · 15 días"
    )

    if (
        isinstance(
            exog_future,
            pd.DataFrame,
        )
        and not exog_future.empty
        and "precip_mm"
        in exog_future.columns
    ):

        rain = (
            exog_future
            .head(
                FORECAST_DAYS
            )
            .copy()
        )

        rain[
            "precip_mm"
        ] = pd.to_numeric(
            rain[
                "precip_mm"
            ],
            errors="coerce",
        ).fillna(
            0.0
        )

        r1, r2 = st.columns(
            2
        )

        r1.metric(
            "Acumulado",
            f"{rain['precip_mm'].sum():.1f} mm",
        )

        r2.metric(
            "Máximo diario",
            f"{rain['precip_mm'].max():.1f} mm",
        )

        st.metric(
            "Días con lluvia ≥ 1 mm",
            int(
                (
                    rain[
                        "precip_mm"
                    ]
                    >= 1
                ).sum()
            ),
        )

        rain_fig = go.Figure()

        rain_fig.add_trace(
            go.Bar(
                x=rain[
                    "datetime"
                ],
                y=rain[
                    "precip_mm"
                ],
                name="Precipitación",
            )
        )

        rain_fig.update_layout(
            height=280,
            yaxis_title="mm/día",
            margin=dict(
                l=10,
                r=10,
                t=25,
                b=10,
            ),
        )

        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )

    else:

        st.info(
            "No hay pronóstico de precipitación disponible."
        )


    # ========================================================
    # CAUDAL
    # ========================================================

    st.subheader(
        "💧 Caudal"
    )

    if (
        isinstance(
            exog_history,
            pd.DataFrame,
        )
        and not exog_history.empty
        and "caudal_m3s"
        in exog_history.columns
    ):

        q = exog_history[
            [
                "datetime",
                "caudal_m3s",
            ]
        ].copy()

        q[
            "caudal_m3s"
        ] = pd.to_numeric(
            q[
                "caudal_m3s"
            ],
            errors="coerce",
        )

        q = q.dropna(
            subset=[
                "caudal_m3s"
            ]
        )

        if not q.empty:

            q_actual = float(
                q[
                    "caudal_m3s"
                ].iloc[-1]
            )

            q_anterior = None

            if len(q) >= 2:

                q_anterior = float(
                    q[
                        "caudal_m3s"
                    ].iloc[-2]
                )

            q_delta = (
                q_actual
                - q_anterior
                if q_anterior
                is not None
                else None
            )

            q1, q2 = st.columns(
                2
            )

            q1.metric(
                "Caudal actual",
                f"{q_actual:,.0f} m³/s",
            )

            q2.metric(
                "Tendencia",
                texto_tendencia(
                    q_delta
                ),
            )

            q_fig = go.Figure()

            q_fig.add_trace(
                go.Scatter(
                    x=q[
                        "datetime"
                    ],
                    y=q[
                        "caudal_m3s"
                    ],
                    mode="lines",
                    name="Caudal",
                )
            )

            q_fig.update_layout(
                height=300,
                yaxis_title="m³/s",
                margin=dict(
                    l=10,
                    r=10,
                    t=25,
                    b=10,
                ),
            )

            st.plotly_chart(
                q_fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No hay valores válidos de caudal."
            )

    else:

        st.info(
            "No se encontró serie de caudal."
        )


    # ========================================================
    # ESCENARIO 60 DÍAS
    # ========================================================

    try:

        render_stress_scenario(
            df=df,
            models=models,
            exog_history=
                exog_history,
            upstream_history=
                upstream_history,
        )

    except Exception as exc:

        st.warning(
            "No fue posible construir "
            "el escenario de 60 días. "
            f"Detalle: {exc}"
        )


    # ========================================================
    # DIAGNÓSTICO
    # ========================================================

    with st.expander(
        "🧠 Diagnóstico del modelo"
    ):

        rmse = (
            metrics.get(
                "RMSE"
            )
            if isinstance(
                metrics,
                dict,
            )
            else None
        )

        if rmse is not None:

            st.metric(
                "RMSE histórico",
                f"{float(rmse):.3f} m",
            )

        if isinstance(
            models,
            dict,
        ):

            st.write(
                "**Precipitación:**",
                (
                    "Incluida"
                    if models.get(
                        "uses_rain",
                        False,
                    )
                    else "No disponible"
                ),
            )

            st.write(
                "**Caudal:**",
                (
                    "Incluido"
                    if models.get(
                        "uses_caudal",
                        False,
                    )
                    else "No disponible"
                ),
            )

            st.write(
                "**Estaciones aguas arriba:**",
                (
                    "Incluidas"
                    if models.get(
                        "uses_upstream",
                        False,
                    )
                    else "No disponibles"
                ),
            )

            importance = (
                models.get(
                    "importance"
                )
            )

            if (
                isinstance(
                    importance,
                    pd.DataFrame,
                )
                and not importance.empty
            ):

                st.markdown(
                    "#### Variables más influyentes"
                )

                st.dataframe(
                    importance.head(
                        20
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


        # ----------------------------------------------------
        # Diagnóstico hidrológico
        # ----------------------------------------------------

        st.markdown(
            "#### Diagnóstico Corrientes → San Nicolás"
        )

        st.write(
            "**Retardo encontrado:**",
            (
                f"{best_lag} días"
                if best_lag is not None
                else "No disponible"
            ),
        )

        st.write(
            "**Correlación temporal:**",
            (
                f"{correlation:.3f}"
                if np.isfinite(
                    correlation
                )
                else "No disponible"
            ),
        )

        st.write(
            "**Eventos históricos:**",
            (
                len(events)
                if isinstance(
                    events,
                    pd.DataFrame,
                )
                else 0
            ),
        )


    if actualizado:

        st.caption(
            "Última actualización de la aplicación: "
            f"{actualizado.strftime('%d/%m/%Y %H:%M')}"
        )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    f"Paraná · San Nicolás · {APP_VERSION}"
)

st.caption(
    "Nivel observado: INA A5 · "
    "Pronóstico experimental"
)

st.warning(
    "Las proyecciones son experimentales y no reemplazan "
    "alertas, mediciones ni pronósticos emitidos por "
    "organismos oficiales."
)
