import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import date, timedelta, datetime


from src.ina import observed

from src.model import (
    train,
    predict,
)

from src.exogenous import (
    get_exogenous_data,
)

from src.upstream import (
    get_upstream_data,
)

from src.stress_ui import (
    render_stress_scenario,
)


# ============================================================
# VERSIÓN
# ============================================================

APP_VERSION = "V11.1"


# ============================================================
# HORIZONTES
# ============================================================

FORECAST_DAYS = 15
TREND_DAYS = 30
STRESS_DAYS = 60


# ============================================================
# HISTÓRICO
# ============================================================

FULL_HISTORY_START = "1900-01-01"

MODEL_HISTORY_YEARS = 15


# ============================================================
# ESCALA
# ============================================================

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
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 2rem;
        max-width: 1700px;
    }

    [data-testid="stMetric"] {
        padding: 0.35rem 0.50rem;
        border: 1px solid rgba(120, 120, 120, 0.22);
        border-radius: 8px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.40rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.60rem;
    }

    h1 {
        margin-bottom: 0.1rem;
    }

    h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
    }

    .compact-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .status-box {
        border: 1px solid rgba(130, 130, 130, 0.25);
        border-radius: 8px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ENCABEZADO
# ============================================================

header_left, header_right = st.columns(
    [
        5,
        2,
    ]
)

with header_left:

    st.title(
        "🌊 PARANÁ · SAN NICOLÁS"
    )

    st.caption(
        f"{APP_VERSION} · Monitoreo y pronóstico hidrométrico experimental"
    )


with header_right:

    st.caption(
        "📍 San Nicolás de los Arroyos"
    )

    st.caption(
        f"📅 {date.today().strftime('%d/%m/%Y')}"
    )


st.markdown(
    """
    **Nivel INA · lluvia corredor Paraná · caudal · estaciones aguas arriba ·
    pronóstico recursivo diario**
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Control"
)


fecha_hasta = date.today()

fecha_desde = (
    fecha_hasta
    - timedelta(
        days=120
    )
)


desde = st.sidebar.date_input(
    "Desde",
    value=fecha_desde,
    format="DD/MM/YYYY",
)


hasta = st.sidebar.date_input(
    "Hasta",
    value=fecha_hasta,
    format="DD/MM/YYYY",
)


actualizar = st.sidebar.button(
    "🔄 Actualizar modelo",
    use_container_width=True,
    type="primary",
)


st.sidebar.divider()


st.sidebar.markdown(
    "**Horizontes**"
)

st.sidebar.caption(
    "15 días · Pronóstico principal"
)

st.sidebar.caption(
    "30 días · Extensión experimental"
)

st.sidebar.caption(
    "60 días · Escenario histórico severo"
)


st.sidebar.divider()


st.sidebar.markdown(
    "**Fuentes**"
)

st.sidebar.caption(
    "INA · niveles y caudal"
)

st.sidebar.caption(
    "Open-Meteo · precipitación"
)

st.sidebar.caption(
    "Modelo propio · Random Forest"
)


# ============================================================
# UTILIDADES
# ============================================================

def preparar_datos(
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

        return pd.DataFrame()

    x = df.copy()

    if "datetime" not in x.columns:

        return pd.DataFrame()

    x[
        "datetime"
    ] = pd.to_datetime(
        x[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    if "value" in x.columns:

        x[
            "nivel"
        ] = pd.to_numeric(
            x[
                "value"
            ],
            errors="coerce",
        )

    elif "nivel" in x.columns:

        x[
            "nivel"
        ] = pd.to_numeric(
            x[
                "nivel"
            ],
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


# ============================================================
# NIVEL DIARIO
# ============================================================

def nivel_diario(
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

        return pd.DataFrame()

    x = df.copy()

    x[
        "datetime"
    ] = (
        pd.to_datetime(
            x[
                "datetime"
            ],
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(
            None
        )
        .dt
        .normalize()
    )

    x[
        "nivel"
    ] = pd.to_numeric(
        x[
            "nivel"
        ],
        errors="coerce",
    )

    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    x = (
        x
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
# ENVOLVENTE HISTÓRICA
# ============================================================

def calcular_envolvente_historica(
    df_historico,
    fechas,
):

    hist = nivel_diario(
        df_historico
    )

    if hist.empty:

        return pd.DataFrame()

    hist[
        "mes"
    ] = hist[
        "datetime"
    ].dt.month

    hist[
        "dia"
    ] = hist[
        "datetime"
    ].dt.day


    resumen = (
        hist
        .groupby(
            [
                "mes",
                "dia",
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


    futuro = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    fechas,
                    errors="coerce",
                )
        }
    )


    futuro[
        "mes"
    ] = futuro[
        "datetime"
    ].dt.month

    futuro[
        "dia"
    ] = futuro[
        "datetime"
    ].dt.day


    futuro = futuro.merge(
        resumen,
        on=[
            "mes",
            "dia",
        ],
        how="left",
    )


    return futuro


# ============================================================
# POSICIÓN HISTÓRICA ACTUAL
# ============================================================

def calcular_posicion_historica(
    df_historico,
    fecha_actual,
    nivel_actual,
):

    resultado = {

        "min":
            None,

        "max":
            None,

        "prom":
            None,

        "posicion":
            None,

        "registros":
            0,
    }


    hist = nivel_diario(
        df_historico
    )


    if hist.empty:

        return resultado


    fecha_actual = pd.to_datetime(
        fecha_actual,
        errors="coerce",
    )


    if pd.isna(
        fecha_actual
    ):

        return resultado


    mismo_dia = hist[
        (
            hist[
                "datetime"
            ].dt.month
            == fecha_actual.month
        )
        &
        (
            hist[
                "datetime"
            ].dt.day
            == fecha_actual.day
        )
    ]


    if mismo_dia.empty:

        return resultado


    minimo = float(
        mismo_dia[
            "nivel"
        ].min()
    )

    maximo = float(
        mismo_dia[
            "nivel"
        ].max()
    )

    promedio = float(
        mismo_dia[
            "nivel"
        ].mean()
    )


    resultado[
        "min"
    ] = minimo

    resultado[
        "max"
    ] = maximo

    resultado[
        "prom"
    ] = promedio

    resultado[
        "registros"
    ] = len(
        mismo_dia
    )


    if maximo > minimo:

        posicion = (
            (
                nivel_actual
                - minimo
            )
            /
            (
                maximo
                - minimo
            )
            * 100
        )

        resultado[
            "posicion"
        ] = float(
            np.clip(
                posicion,
                0,
                100,
            )
        )


    return resultado


# ============================================================
# TENDENCIA DEL CAUDAL
# ============================================================

def calcular_tendencia_caudal(
    df_caudal,
):

    resultado = {

        "actual":
            None,

        "delta_3":
            None,

        "delta_7":
            None,

        "pct_7":
            None,

        "estado":
            "Sin datos",
    }


    if (
        df_caudal is None
        or not isinstance(
            df_caudal,
            pd.DataFrame,
        )
        or df_caudal.empty
        or "caudal_m3s"
        not in df_caudal.columns
    ):

        return resultado


    valores = (
        pd.to_numeric(
            df_caudal[
                "caudal_m3s"
            ],
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )


    if len(
        valores
    ) == 0:

        return resultado


    actual = float(
        valores[
            -1
        ]
    )


    resultado[
        "actual"
    ] = actual


    if len(
        valores
    ) >= 4:

        resultado[
            "delta_3"
        ] = (
            actual
            - float(
                valores[
                    -4
                ]
            )
        )


    if len(
        valores
    ) >= 8:

        base7 = float(
            valores[
                -8
            ]
        )

        delta7 = (
            actual
            - base7
        )


        resultado[
            "delta_7"
        ] = delta7


        if base7 != 0:

            resultado[
                "pct_7"
            ] = (
                delta7
                / base7
                * 100
            )


    recientes = valores[
        -min(
            7,
            len(
                valores
            ),
        ):
    ]


    if len(
        recientes
    ) >= 3:

        pendiente = float(
            np.polyfit(
                np.arange(
                    len(
                        recientes
                    )
                ),
                recientes,
                1,
            )[0]
        )

    else:

        pendiente = 0.0


    umbral = max(
        abs(
            actual
        )
        * 0.002,
        1.0,
    )


    if pendiente > umbral:

        resultado[
            "estado"
        ] = "Creciente ↑"

    elif pendiente < -umbral:

        resultado[
            "estado"
        ] = "Bajante ↓"

    else:

        resultado[
            "estado"
        ] = "Estable →"


    return resultado


# ============================================================
# TENDENCIA ESTACIÓN
# ============================================================

def tendencia_estacion(
    info,
):

    if not isinstance(
        info,
        dict,
    ):

        return "Sin datos"


    estado = info.get(
        "estado",
        "Sin datos",
    )


    if estado == "Creciente":

        return "↑"

    if estado == "Bajante":

        return "↓"

    if estado == "Estable":

        return "→"


    return "?"


# ============================================================
# VALIDACIÓN
# ============================================================

if desde > hasta:

    st.sidebar.error(
        "La fecha Desde no puede ser posterior a Hasta."
    )


# ============================================================
# ACTUALIZAR MODELO
# ============================================================

if actualizar:

    if desde > hasta:

        st.error(
            "El período seleccionado no es válido."
        )

    else:

        inicio_visual = desde.strftime(
            "%Y-%m-%d"
        )

        fin = hasta.strftime(
            "%Y-%m-%d"
        )


        # ====================================================
        # NIVEL ACTUAL
        # ====================================================

        with st.spinner(
            "Consultando nivel de San Nicolás..."
        ):

            df_ina, error_ina = observed(
                inicio_visual,
                fin,
            )


        if error_ina:

            st.error(
                error_ina
            )

        else:

            df_visual = preparar_datos(
                df_ina
            )


            if df_visual.empty:

                st.error(
                    "No se obtuvieron niveles válidos "
                    "para San Nicolás."
                )

            else:

                # ============================================
                # HISTÓRICO COMPLETO
                # ============================================

                with st.spinner(
                    "Consultando histórico de niveles..."
                ):

                    try:

                        (
                            df_hist_raw,
                            error_hist,
                        ) = observed(
                            FULL_HISTORY_START,
                            fin,
                        )


                        if (
                            error_hist
                            or df_hist_raw is None
                        ):

                            df_historico = (
                                df_visual.copy()
                            )

                        else:

                            df_historico = preparar_datos(
                                df_hist_raw
                            )


                            if df_historico.empty:

                                df_historico = (
                                    df_visual.copy()
                                )

                    except Exception:

                        df_historico = (
                            df_visual.copy()
                        )


                # ============================================
                # PERÍODO DE ENTRENAMIENTO
                # ============================================

                fecha_modelo_inicio = (
                    pd.Timestamp(
                        hasta
                    )
                    - pd.DateOffset(
                        years=MODEL_HISTORY_YEARS
                    )
                )


                inicio_modelo = (
                    fecha_modelo_inicio
                    .strftime(
                        "%Y-%m-%d"
                    )
                )


                # ============================================
                # NIVEL PARA MODELO
                # ============================================

                try:

                    (
                        df_model_raw,
                        error_model,
                    ) = observed(
                        inicio_modelo,
                        fin,
                    )


                    if error_model:

                        df_model = (
                            df_historico.copy()
                        )

                    else:

                        df_model = preparar_datos(
                            df_model_raw
                        )

                except Exception:

                    df_model = (
                        df_historico.copy()
                    )


                if df_model.empty:

                    df_model = (
                        df_visual.copy()
                    )


                # ============================================
                # LLUVIA + CAUDAL
                # ============================================

                with st.spinner(
                    "Consultando lluvia y caudal..."
                ):

                    try:

                        (
                            exog_history,
                            exog_future,
                            exog_meta,
                        ) = get_exogenous_data(

                            inicio_modelo,

                            fin,

                            forecast_days=
                                TREND_DAYS,
                        )

                    except Exception as exc:

                        exog_history = (
                            pd.DataFrame()
                        )

                        exog_future = (
                            pd.DataFrame()
                        )

                        exog_meta = {}

                        st.warning(
                            "Variables externas incompletas: "
                            f"{exc}"
                        )


                # ============================================
                # AGUAS ARRIBA
                # ============================================

                with st.spinner(
                    "Consultando estaciones aguas arriba..."
                ):

                    try:

                        fechas_futuras = None


                        if (
                            isinstance(
                                exog_future,
                                pd.DataFrame,
                            )
                            and not exog_future.empty
                            and "datetime"
                            in exog_future.columns
                        ):

                            fechas_futuras = (
                                exog_future[
                                    "datetime"
                                ]
                            )


                        (
                            upstream_history,
                            upstream_future,
                            upstream_meta,
                            upstream_projection_meta,
                        ) = get_upstream_data(

                            start=
                                inicio_modelo,

                            end=
                                fin,

                            forecast_days=
                                TREND_DAYS,

                            future_dates=
                                fechas_futuras,
                        )

                    except Exception as exc:

                        upstream_history = (
                            pd.DataFrame()
                        )

                        upstream_future = (
                            pd.DataFrame()
                        )

                        upstream_meta = {}

                        upstream_projection_meta = {}

                        st.warning(
                            "Estaciones aguas arriba incompletas: "
                            f"{exc}"
                        )


                # ============================================
                # ENTRENAMIENTO
                # ============================================

                with st.spinner(
                    "Entrenando modelo..."
                ):

                    try:

                        models, metrics = train(

                            df_model,

                            exog_history=
                                exog_history,

                            upstream_history=
                                upstream_history,
                        )

                    except Exception as exc:

                        models = {}

                        metrics = {}

                        st.error(
                            "No fue posible entrenar el modelo: "
                            f"{exc}"
                        )


                # ============================================
                # PRONÓSTICO
                # ============================================

                forecast30 = (
                    pd.DataFrame()
                )

                forecast15 = (
                    pd.DataFrame()
                )


                if models:

                    with st.spinner(
                        "Generando pronóstico de 30 días..."
                    ):

                        try:

                            forecast30 = predict(

                                df=
                                    df_model,

                                models=
                                    models,

                                days=
                                    TREND_DAYS,

                                exog_future=
                                    exog_future,

                                upstream_future=
                                    upstream_future,
                            )


                            if (
                                isinstance(
                                    forecast30,
                                    pd.DataFrame,
                                )
                                and not forecast30.empty
                            ):

                                forecast15 = (
                                    forecast30
                                    .head(
                                        FORECAST_DAYS
                                    )
                                    .copy()
                                )

                        except Exception as exc:

                            st.error(
                                "No fue posible generar "
                                f"el pronóstico: {exc}"
                            )


                # ============================================
                # SESIÓN
                # ============================================

                st.session_state[
                    "datos"
                ] = df_visual


                st.session_state[
                    "datos_modelo"
                ] = df_model


                st.session_state[
                    "datos_historicos"
                ] = df_historico


                st.session_state[
                    "forecast"
                ] = forecast15


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
                    "upstream_future"
                ] = upstream_future


                st.session_state[
                    "upstream_meta"
                ] = upstream_meta


                st.session_state[
                    "upstream_projection_meta"
                ] = upstream_projection_meta


                st.session_state[
                    "actualizado"
                ] = datetime.now()


                st.success(
                    "✅ Modelo actualizado."
                )


# ============================================================
# ESTADO INICIAL
# ============================================================

if "datos" not in st.session_state:

    st.info(
        "Seleccione el período y presione "
        "**Actualizar modelo**."
    )


# ============================================================
# RESULTADOS
# ============================================================

else:

    df = st.session_state[
        "datos"
    ]


    df_model = st.session_state.get(
        "datos_modelo",
        df,
    )


    df_historico = st.session_state.get(
        "datos_historicos",
        df,
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


    exog_meta = st.session_state.get(
        "exog_meta",
        {},
    )


    upstream_history = st.session_state.get(
        "upstream_history",
        pd.DataFrame(),
    )


    upstream_future = st.session_state.get(
        "upstream_future",
        pd.DataFrame(),
    )


    upstream_meta = st.session_state.get(
        "upstream_meta",
        {},
    )


    upstream_projection_meta = (
        st.session_state.get(
            "upstream_projection_meta",
            {},
        )
    )


    actualizado = st.session_state.get(
        "actualizado"
    )


    # ========================================================
    # NIVEL ACTUAL
    # ========================================================

    ultima_fecha = df[
        "datetime"
    ].iloc[
        -1
    ]


    nivel_actual = float(
        df[
            "nivel"
        ].iloc[
            -1
        ]
    )


    # ========================================================
    # VARIACIÓN 24 H
    # ========================================================

    df_daily = nivel_diario(
        df
    )


    delta_24h = None


    if len(
        df_daily
    ) >= 2:

        delta_24h = float(
            df_daily[
                "nivel"
            ].iloc[
                -1
            ]
            - df_daily[
                "nivel"
            ].iloc[
                -2
            ]
        )


    # ========================================================
    # PRONÓSTICOS CLAVE
    # ========================================================

    nivel_dia3 = None
    nivel_dia7 = None
    nivel_dia15 = None
    nivel_dia30 = None


    if (
        isinstance(
            forecast30,
            pd.DataFrame,
        )
        and not forecast30.empty
    ):

        if len(
            forecast30
        ) >= 3:

            nivel_dia3 = float(
                forecast30[
                    "prediction"
                ].iloc[
                    2
                ]
            )

        if len(
            forecast30
        ) >= 7:

            nivel_dia7 = float(
                forecast30[
                    "prediction"
                ].iloc[
                    6
                ]
            )

        if len(
            forecast30
        ) >= 15:

            nivel_dia15 = float(
                forecast30[
                    "prediction"
                ].iloc[
                    14
                ]
            )

        if len(
            forecast30
        ) >= 30:

            nivel_dia30 = float(
                forecast30[
                    "prediction"
                ].iloc[
                    29
                ]
            )


    # ========================================================
    # CAUDAL
    # ========================================================

    tq = calcular_tendencia_caudal(
        exog_history
    )


    # ========================================================
    # TENDENCIA GENERAL
    # ========================================================

    if (
        nivel_dia15
        is not None
    ):

        cambio15 = (
            nivel_dia15
            - nivel_actual
        )


        if cambio15 >= 0.15:

            tendencia_general = (
                "Creciente ↑"
            )

        elif cambio15 <= -0.15:

            tendencia_general = (
                "Bajante ↓"
            )

        else:

            tendencia_general = (
                "Estable →"
            )

    else:

        tendencia_general = (
            "Sin datos"
        )


    # ========================================================
    # FILA PRINCIPAL DE MÉTRICAS
    # ========================================================

    st.markdown(
        "### 📊 Resumen operativo"
    )


    m1, m2, m3, m4, m5, m6 = st.columns(
        6
    )


    m1.metric(
        "Nivel actual",
        f"{nivel_actual:.2f} m",
    )


    m2.metric(
        "Variación 24 h",
        (
            f"{delta_24h:+.2f} m"
            if delta_24h
            is not None
            else "Sin datos"
        ),
    )


    m3.metric(
        "Día 15",
        (
            f"{nivel_dia15:.2f} m"
            if nivel_dia15
            is not None
            else "Sin datos"
        ),
        (
            f"{nivel_dia15 - nivel_actual:+.2f} m"
            if nivel_dia15
            is not None
            else None
        ),
    )


    m4.metric(
        "Día 30",
        (
            f"{nivel_dia30:.2f} m"
            if nivel_dia30
            is not None
            else "Sin datos"
        ),
    )


    m5.metric(
        "Caudal actual",
        (
            f"{tq['actual']:,.0f} m³/s"
            if tq[
                "actual"
            ]
            is not None
            else "Sin datos"
        ),
    )


    m6.metric(
        "Tendencia",
        tendencia_general,
    )


    # ========================================================
    # ESTADO DEL SISTEMA
    # ========================================================

    estado_lluvia = (
        isinstance(
            exog_future,
            pd.DataFrame,
        )
        and not exog_future.empty
        and "precip_mm"
        in exog_future.columns
    )


    estado_caudal = (
        isinstance(
            exog_history,
            pd.DataFrame,
        )
        and not exog_history.empty
        and "caudal_m3s"
        in exog_history.columns
        and exog_history[
            "caudal_m3s"
        ].notna().any()
    )


    estaciones_disponibles = 0


    if isinstance(
        upstream_history,
        pd.DataFrame,
    ):

        estaciones_disponibles = len(
            [
                c
                for c
                in upstream_history.columns
                if (
                    c.startswith(
                        "nivel_"
                    )
                    and upstream_history[
                        c
                    ].notna().any()
                )
            ]
        )


    rmse = metrics.get(
        "RMSE"
    )


    st.markdown(
        """
        <div class="status-box">
        """,
        unsafe_allow_html=True,
    )


    status_cols = st.columns(
        6
    )


    status_cols[
        0
    ].caption(
        "🟢 **INA** · OK"
    )


    status_cols[
        1
    ].caption(
        (
            "🟢 **Lluvia** · OK"
            if estado_lluvia
            else "🟠 **Lluvia** · Sin datos"
        )
    )


    status_cols[
        2
    ].caption(
        (
            "🟢 **Caudal** · OK"
            if estado_caudal
            else "🟠 **Caudal** · Sin datos"
        )
    )


    status_cols[
        3
    ].caption(
        f"🟢 **Aguas arriba** · "
        f"{estaciones_disponibles}/6"
    )


    status_cols[
        4
    ].caption(
        (
            "🟢 **Modelo** · OK"
            if models
            else "🔴 **Modelo** · Error"
        )
    )


    status_cols[
        5
    ].caption(
        (
            f"**RMSE** · {float(rmse):.3f} m"
            if rmse
            is not None
            else "**RMSE** · --"
        )
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    # ========================================================
    # GRÁFICO PRINCIPAL + PANEL DERECHO
    # ========================================================

    grafico_col, resumen_col = st.columns(
        [
            3.3,
            1.15,
        ]
    )


    with grafico_col:

        st.markdown(
            "### 📈 Nivel observado y proyección"
        )


        fig_main = go.Figure()


        obs = df.tail(
            60
        )


        fig_main.add_trace(
            go.Scatter(

                x=obs[
                    "datetime"
                ],

                y=obs[
                    "nivel"
                ],

                mode="lines",

                name="Nivel observado",

                line=dict(
                    color="#bdbdbd",
                    width=2,
                ),
            )
        )


        if (
            isinstance(
                forecast30,
                pd.DataFrame,
            )
            and not forecast30.empty
        ):

            forecast_plot = (
                forecast30.copy()
            )


            env = calcular_envolvente_historica(
                df_historico,
                forecast_plot[
                    "datetime"
                ],
            )


            if not env.empty:

                forecast_plot = (
                    forecast_plot
                    .merge(
                        env[
                            [
                                "datetime",
                                "nivel_min_historico",
                                "nivel_max_historico",
                                "nivel_promedio_historico",
                            ]
                        ],
                        on="datetime",
                        how="left",
                    )
                )


            if (
                "nivel_max_historico"
                in forecast_plot.columns
            ):

                fig_main.add_trace(
                    go.Scatter(

                        x=forecast_plot[
                            "datetime"
                        ],

                        y=forecast_plot[
                            "nivel_max_historico"
                        ],

                        mode="lines",

                        name="Máximo histórico",

                        line=dict(
                            color="#ef553b",
                            width=2,
                        ),
                    )
                )


            if (
                "nivel_min_historico"
                in forecast_plot.columns
            ):

                fig_main.add_trace(
                    go.Scatter(

                        x=forecast_plot[
                            "datetime"
                        ],

                        y=forecast_plot[
                            "nivel_min_historico"
                        ],

                        mode="lines",

                        name="Mínimo histórico",

                        line=dict(
                            color="#00cc96",
                            width=2,
                        ),
                    )
                )


            pron15 = forecast_plot.head(
                FORECAST_DAYS
            )


            pron30ext = forecast_plot.iloc[
                FORECAST_DAYS:
            ]


            fig_main.add_trace(
                go.Scatter(

                    x=pron15[
                        "datetime"
                    ],

                    y=pron15[
                        "prediction"
                    ],

                    mode="lines+markers",

                    name="Pronóstico 1–15 días",

                    line=dict(
                        color="#1f77b4",
                        width=4,
                    ),

                    marker=dict(
                        size=6,
                    ),
                )
            )


            if not pron30ext.empty:

                fig_main.add_trace(
                    go.Scatter(

                        x=pron30ext[
                            "datetime"
                        ],

                        y=pron30ext[
                            "prediction"
                        ],

                        mode="lines+markers",

                        name="Extensión 16–30 días",

                        line=dict(
                            color="#636efa",
                            width=3,
                            dash="dot",
                        ),

                        marker=dict(
                            size=5,
                        ),
                    )
                )


            if (
                "upper"
                in pron15.columns
                and "lower"
                in pron15.columns
            ):

                fig_main.add_trace(
                    go.Scatter(

                        x=pron15[
                            "datetime"
                        ],

                        y=pron15[
                            "upper"
                        ],

                        mode="lines",

                        line=dict(
                            width=0,
                        ),

                        showlegend=False,

                        hoverinfo="skip",
                    )
                )


                fig_main.add_trace(
                    go.Scatter(

                        x=pron15[
                            "datetime"
                        ],

                        y=pron15[
                            "lower"
                        ],

                        mode="lines",

                        line=dict(
                            width=0,
                        ),

                        fill="tonexty",

                        name="Incertidumbre",

                        hoverinfo="skip",
                    )
                )


        fig_main.add_hline(

            y=nivel_actual,

            line_dash="dash",

            annotation_text=(
                f"Nivel actual "
                f"{nivel_actual:.2f} m"
            ),
        )


        fig_main.update_layout(

            height=500,

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10,
            ),

            hovermode="x unified",

            legend=dict(
                orientation="h",
                y=1.10,
            ),
        )


        fig_main.update_xaxes(
            tickformat="%d/%m",
            title_text="Fecha",
        )


        fig_main.update_yaxes(

            title_text="Nivel (m)",

            range=[
                Y_MIN,
                Y_MAX,
            ],

            dtick=
                Y_STEP,
        )


        st.plotly_chart(
            fig_main,
            use_container_width=True,
        )


    # ========================================================
    # PANEL RESUMEN DERECHO
    # ========================================================

    with resumen_col:

        st.markdown(
            "### 🎯 Proyección"
        )


        if (
            isinstance(
                forecast30,
                pd.DataFrame,
            )
            and not forecast30.empty
        ):

            max_idx = (
                forecast30[
                    "prediction"
                ]
                .idxmax()
            )


            max_nivel = float(
                forecast30.loc[
                    max_idx,
                    "prediction"
                ]
            )


            max_fecha = pd.to_datetime(
                forecast30.loc[
                    max_idx,
                    "datetime"
                ]
            )


            st.metric(
                "Día 3",
                (
                    f"{nivel_dia3:.2f} m"
                    if nivel_dia3
                    is not None
                    else "--"
                ),
            )


            st.metric(
                "Día 7",
                (
                    f"{nivel_dia7:.2f} m"
                    if nivel_dia7
                    is not None
                    else "--"
                ),
            )


            st.metric(
                "Día 15",
                (
                    f"{nivel_dia15:.2f} m"
                    if nivel_dia15
                    is not None
                    else "--"
                ),
            )


            st.metric(
                "Día 30",
                (
                    f"{nivel_dia30:.2f} m"
                    if nivel_dia30
                    is not None
                    else "--"
                ),
            )


            st.metric(
                "Máximo proyectado",
                f"{max_nivel:.2f} m",
                f"{max_nivel - nivel_actual:+.2f} m",
            )


            st.caption(
                "Fecha máximo: "
                f"**{max_fecha.strftime('%d/%m/%Y')}**"
            )


    # ========================================================
    # POSICIÓN HISTÓRICA
    # ========================================================

    posicion = calcular_posicion_historica(
        df_historico,
        ultima_fecha,
        nivel_actual,
    )


    st.markdown(
        "### 📚 Posición respecto del historial"
    )


    h1, h2, h3, h4, h5 = st.columns(
        5
    )


    h1.metric(
        "Nivel actual",
        f"{nivel_actual:.2f} m",
    )


    h2.metric(
        "Mínimo histórico fecha",
        (
            f"{posicion['min']:.2f} m"
            if posicion[
                "min"
            ]
            is not None
            else "--"
        ),
    )


    h3.metric(
        "Promedio histórico fecha",
        (
            f"{posicion['prom']:.2f} m"
            if posicion[
                "prom"
            ]
            is not None
            else "--"
        ),
    )


    h4.metric(
        "Máximo histórico fecha",
        (
            f"{posicion['max']:.2f} m"
            if posicion[
                "max"
            ]
            is not None
            else "--"
        ),
    )


    h5.metric(
        "Posición en rango",
        (
            f"{posicion['posicion']:.0f}%"
            if posicion[
                "posicion"
            ]
            is not None
            else "--"
        ),
    )


    # ========================================================
    # LLUVIA / CAUDAL / AGUAS ARRIBA
    # ========================================================

    lluvia_col, caudal_col, upstream_col = st.columns(
        3
    )


    # ========================================================
    # LLUVIA
    # ========================================================

    with lluvia_col:

        st.markdown(
            "### 🌧️ Lluvia"
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

            rain = exog_future.copy()


            rain[
                "precip_mm"
            ] = (
                pd.to_numeric(
                    rain[
                        "precip_mm"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0.0
                )
            )


            rain3 = rain.head(
                3
            )

            rain7 = rain.head(
                7
            )

            rain15 = rain.head(
                15
            )


            rc1, rc2, rc3 = st.columns(
                3
            )


            rc1.metric(
                "3 días",
                f"{rain3['precip_mm'].sum():.1f} mm",
            )


            rc2.metric(
                "7 días",
                f"{rain7['precip_mm'].sum():.1f} mm",
            )


            rc3.metric(
                "15 días",
                f"{rain15['precip_mm'].sum():.1f} mm",
            )


            rain_fig = go.Figure()


            rain_fig.add_trace(
                go.Bar(

                    x=rain.head(
                        15
                    )[
                        "datetime"
                    ],

                    y=rain.head(
                        15
                    )[
                        "precip_mm"
                    ],

                    name="Lluvia",
                )
            )


            rain_fig.update_layout(

                height=280,

                margin=dict(
                    l=5,
                    r=5,
                    t=5,
                    b=5,
                ),

                yaxis_title="mm/día",

                showlegend=False,
            )


            rain_fig.update_xaxes(
                tickformat="%d/%m",
            )


            st.plotly_chart(
                rain_fig,
                use_container_width=True,
            )


        else:

            st.info(
                "Sin lluvia disponible."
            )


    # ========================================================
    # CAUDAL
    # ========================================================

    with caudal_col:

        st.markdown(
            "### 💧 Caudal"
        )


        qc1, qc2, qc3 = st.columns(
            3
        )


        qc1.metric(
            "Actual",
            (
                f"{tq['actual']:,.0f}"
                if tq[
                    "actual"
                ]
                is not None
                else "--"
            ),
        )


        qc2.metric(
            "Δ 3 días",
            (
                f"{tq['delta_3']:+,.0f}"
                if tq[
                    "delta_3"
                ]
                is not None
                else "--"
            ),
        )


        qc3.metric(
            "Tendencia",
            tq[
                "estado"
            ],
        )


        if (
            isinstance(
                exog_future,
                pd.DataFrame,
            )
            and not exog_future.empty
            and "caudal_m3s"
            in exog_future.columns
        ):

            qfig = go.Figure()


            qfig.add_trace(
                go.Scatter(

                    x=exog_future[
                        "datetime"
                    ],

                    y=exog_future[
                        "caudal_m3s"
                    ],

                    mode="lines+markers",

                    name="Caudal proyectado",
                )
            )


            qfig.update_layout(

                height=280,

                margin=dict(
                    l=5,
                    r=5,
                    t=5,
                    b=5,
                ),

                yaxis_title="m³/s",

                showlegend=False,
            )


            qfig.update_xaxes(
                tickformat="%d/%m",
            )


            st.plotly_chart(
                qfig,
                use_container_width=True,
            )


    # ========================================================
    # AGUAS ARRIBA
    # ========================================================

    with upstream_col:

        st.markdown(
            "### 🌊 Aguas arriba"
        )


        station_rows = []


        nombres_map = {

            "nivel_corrientes":
                "Corrientes",

            "nivel_goya":
                "Goya",

            "nivel_la_paz":
                "La Paz",

            "nivel_parana":
                "Paraná",

            "nivel_rosario":
                "Rosario",

            "nivel_villa_constitucion":
                "Villa Constitución",
        }


        for col, nombre in nombres_map.items():

            info = (
                upstream_projection_meta
                .get(
                    col,
                    {},
                )
            )


            actual = info.get(
                "actual"
            )


            station_rows.append(
                {
                    "Estación":
                        nombre,

                    "Nivel":
                        (
                            f"{actual:.2f} m"
                            if actual
                            is not None
                            else "--"
                        ),

                    "Tendencia":
                        tendencia_estacion(
                            info
                        ),
                }
            )


        st.dataframe(

            pd.DataFrame(
                station_rows
            ),

            use_container_width=True,

            hide_index=True,

            height=255,
        )


    # ========================================================
    # TABLA DIARIA 15 DÍAS
    # ========================================================

    with st.expander(
        "🔎 Pronóstico diario detallado · 15 días"
    ):

        if (
            isinstance(
                forecast,
                pd.DataFrame,
            )
            and not forecast.empty
        ):

            tabla = forecast.copy()


            tabla[
                "Fecha"
            ] = pd.to_datetime(
                tabla[
                    "datetime"
                ]
            ).dt.strftime(
                "%d/%m/%Y"
            )


            tabla[
                "Nivel base"
            ] = tabla[
                "nivel_base"
            ].round(
                2
            )


            tabla[
                "Lluvia mm"
            ] = tabla[
                "precip_mm"
            ].round(
                1
            )


            tabla[
                "Caudal m³/s"
            ] = tabla[
                "caudal_m3s"
            ].round(
                0
            )


            tabla[
                "Δ nivel"
            ] = tabla[
                "variacion_dia"
            ].round(
                3
            )


            tabla[
                "Nivel previsto"
            ] = tabla[
                "prediction"
            ].round(
                2
            )


            st.dataframe(

                tabla[
                    [
                        "Fecha",
                        "Nivel base",
                        "Lluvia mm",
                        "Caudal m³/s",
                        "Δ nivel",
                        "Nivel previsto",
                    ]
                ],

                use_container_width=True,

                hide_index=True,
            )


    # ========================================================
    # ESCENARIO 60 DÍAS
    # ========================================================

    st.divider()


    render_stress_scenario(

        df=
            df_historico,

        models=
            models,

        exog_history=
            exog_history,

        upstream_history=
            upstream_history,
    )


    # ========================================================
    # IMPORTANCIA DE VARIABLES
    # ========================================================

    importance = models.get(
        "importance"
    )


    if (
        isinstance(
            importance,
            pd.DataFrame,
        )
        and not importance.empty
    ):

        with st.expander(
            "🧠 Importancia de variables del modelo"
        ):

            top_imp = importance.head(
                20
            )


            imp_fig = go.Figure()


            imp_fig.add_trace(
                go.Bar(

                    x=top_imp[
                        "importance"
                    ],

                    y=top_imp[
                        "feature"
                    ],

                    orientation="h",
                )
            )


            imp_fig.update_layout(
                height=550,
                xaxis_title="Importancia relativa",
            )


            imp_fig.update_yaxes(
                autorange="reversed"
            )


            st.plotly_chart(
                imp_fig,
                use_container_width=True,
            )


    # ========================================================
    # METODOLOGÍA
    # ========================================================

    with st.expander(
        "ℹ️ Metodología y alcance"
    ):

        st.markdown(
            """
            **15 días:** pronóstico recursivo diario.

            Cada nuevo día parte del nivel calculado el día anterior
            y utiliza lluvia, caudal y niveles aguas arriba.

            **30 días:** aplica el mismo procedimiento recursivo.
            El tramo sin pronóstico meteorológico directo utiliza
            climatología histórica como variable de lluvia.

            **60 días:** escenario de estrés histórico.
            No representa un pronóstico meteorológico convencional.

            El modelo se recalcula cada vez que se actualiza la
            plataforma y parte de la última medición real disponible
            de San Nicolás.
            """
        )


        if rmse is not None:

            st.write(
                "**RMSE histórico:** "
                f"{float(rmse):.3f} m"
            )


        st.warning(
            "Herramienta experimental. "
            "No reemplaza información oficial."
        )


    # ========================================================
    # ACTUALIZACIÓN
    # ========================================================

    if actualizado:

        st.caption(
            "Última actualización: "
            f"{actualizado.strftime('%d/%m/%Y %H:%M')}"
        )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.divider()


footer1, footer2 = st.columns(
    2
)


with footer1:

    st.caption(
        "Datos hidrométricos y caudal: "
        "**Instituto Nacional del Agua (INA)**"
    )


with footer2:

    st.caption(
        "Precipitación: **Open-Meteo** · "
        "Modelo: **experimental propio**"
    )


st.caption(
    f"Paraná · San Nicolás {APP_VERSION} | "
    "15 días · 30 días · escenario severo 60 días"
)
