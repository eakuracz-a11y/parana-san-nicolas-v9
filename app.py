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

APP_VERSION = "V11.3"


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
        padding-top: 0.65rem;
        padding-bottom: 2rem;
        max-width: 1780px;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.45rem;
    }

    h1 {
        margin-bottom: 0.05rem;
    }

    h2,
    h3 {
        margin-top: 0.35rem;
        margin-bottom: 0.30rem;
    }

    /* ======================================================
       METRIC GENERAL
    ====================================================== */

    [data-testid="stMetric"] {
        padding: 0.35rem 0.45rem;
        border: 1px solid rgba(130, 130, 130, 0.20);
        border-radius: 8px;
        min-height: 82px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.70rem !important;
        line-height: 0.95rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.12rem !important;
        line-height: 1.35rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.68rem !important;
        white-space: nowrap !important;
    }

    /* ======================================================
       TARJETAS COMPACTAS
    ====================================================== */

    .compact-card {
        border: 1px solid rgba(130, 130, 130, 0.24);
        border-radius: 9px;
        padding: 0.55rem 0.65rem;
        min-height: 82px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: rgba(255,255,255,0.01);
    }

    .compact-label {
        font-size: 0.72rem;
        line-height: 0.90rem;
        opacity: 0.78;
        margin-bottom: 0.25rem;
        white-space: normal;
    }

    .compact-value {
        font-size: 1.22rem;
        line-height: 1.35rem;
        font-weight: 650;
        white-space: nowrap;
    }

    .compact-unit {
        font-size: 0.67rem;
        line-height: 0.85rem;
        opacity: 0.65;
        margin-top: 0.15rem;
        white-space: nowrap;
    }

    .compact-trend {
        font-size: 1.05rem;
        line-height: 1.25rem;
        font-weight: 650;
        white-space: nowrap;
    }

    .status-title {
        font-size: 0.76rem;
        font-weight: 700;
        margin-top: 0.10rem;
        margin-bottom: 0.05rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FORMATOS
# ============================================================

def formato_numero(
    valor,
    decimales=0,
):

    if valor is None:
        return "--"

    try:

        if pd.isna(valor):
            return "--"

    except Exception:
        pass

    texto = (
        f"{float(valor):,.{decimales}f}"
    )

    texto = (
        texto
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    return texto


def formato_nivel(
    valor,
):

    if valor is None:
        return "--"

    try:

        if pd.isna(valor):
            return "--"
    except Exception:
        pass

    return (
        f"{float(valor):.2f} m"
    )


def formato_caudal(
    valor,
):

    return formato_numero(
        valor,
        0,
    )


# ============================================================
# TARJETA COMPACTA
# ============================================================

def tarjeta_compacta(
    label,
    value,
    unit="",
    trend=False,
):

    value_class = (
        "compact-trend"
        if trend
        else "compact-value"
    )

    unit_html = (
        f'<div class="compact-unit">{unit}</div>'
        if unit
        else ""
    )

    html = f"""
    <div class="compact-card">
        <div class="compact-label">{label}</div>
        <div class="{value_class}">{value}</div>
        {unit_html}
    </div>
    """

    st.markdown(
        html,
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
        f"{APP_VERSION} · "
        "Monitoreo y pronóstico hidrométrico experimental"
    )


with header_right:

    st.caption(
        "📍 San Nicolás de los Arroyos"
    )

    st.caption(
        f"📅 {date.today().strftime('%d/%m/%Y')}"
    )


st.caption(
    "Nivel INA · lluvia corredor Paraná · caudal · "
    "estaciones aguas arriba · pronóstico recursivo diario"
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
    "Random Forest · modelo experimental"
)


# ============================================================
# PREPARAR DATOS
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
# POSICIÓN HISTÓRICA
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
# TENDENCIA CAUDAL
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

        return "?"

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
# ACTUALIZAR
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
        # NIVEL
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
                # HISTÓRICO
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
                # PERÍODO MODELO
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
                # NIVEL MODELO
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
                # MODELO
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

                forecast30 = pd.DataFrame()
                forecast15 = pd.DataFrame()


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
                    "✅ Modelo actualizado correctamente."
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
    # NIVELES CLAVE
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

    if nivel_dia15 is not None:

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
    # RESUMEN OPERATIVO
    # ========================================================

    st.subheader(
        "📊 Resumen operativo"
    )


    m1, m2, m3, m4, m5, m6 = st.columns(
        [
            1,
            1,
            1,
            1,
            1.15,
            1.15,
        ]
    )


    m1.metric(
        "Nivel actual",
        formato_nivel(
            nivel_actual
        ),
    )


    m2.metric(
        "Variación 24 h",
        (
            f"{delta_24h:+.2f} m"
            if delta_24h
            is not None
            else "--"
        ),
    )


    m3.metric(
        "Día 15",
        formato_nivel(
            nivel_dia15
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
        formato_nivel(
            nivel_dia30
        ),
    )


    m5.metric(
        "Caudal actual",
        (
            f"{formato_caudal(tq['actual'])}"
            if tq[
                "actual"
            ]
            is not None
            else "--"
        ),
        help="m³/s",
    )


    m6.metric(
        "Tendencia",
        tendencia_general,
    )


    # ========================================================
    # ESTADO
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
        '<div class="status-title">Estado del sistema</div>',
        unsafe_allow_html=True,
    )


    status_cols = st.columns(
        [
            1,
            1,
            1,
            1.2,
            1,
            1,
        ]
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


    # ========================================================
    # GRÁFICO PRINCIPAL
    # ========================================================

    grafico_col, resumen_col = st.columns(
        [
            3.45,
            1.05,
        ]
    )


    with grafico_col:

        st.subheader(
            "📈 Nivel observado y proyección"
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

                name="Observado",

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
                f"Actual "
                f"{nivel_actual:.2f} m"
            ),
        )


        fig_main.update_layout(

            height=480,

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10,
            ),

            hovermode="x unified",

            legend=dict(
                orientation="h",
                y=1.11,
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
    # PROYECCIÓN
    # ========================================================

    with resumen_col:

        st.subheader(
            "🎯 Proyección"
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
                formato_nivel(
                    nivel_dia3
                ),
            )


            st.metric(
                "Día 7",
                formato_nivel(
                    nivel_dia7
                ),
            )


            st.metric(
                "Día 15",
                formato_nivel(
                    nivel_dia15
                ),
            )


            st.metric(
                "Día 30",
                formato_nivel(
                    nivel_dia30
                ),
            )


            st.metric(
                "Máximo 30 días",
                formato_nivel(
                    max_nivel
                ),
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


    st.subheader(
        "📚 Posición respecto del historial"
    )


    h1, h2, h3, h4, h5 = st.columns(
        [
            1,
            1.25,
            1.25,
            1.25,
            1,
        ]
    )


    h1.metric(
        "Nivel actual",
        formato_nivel(
            nivel_actual
        ),
    )


    h2.metric(
        "Mínimo histórico fecha",
        formato_nivel(
            posicion[
                "min"
            ]
        ),
    )


    h3.metric(
        "Promedio histórico fecha",
        formato_nivel(
            posicion[
                "prom"
            ]
        ),
    )


    h4.metric(
        "Máximo histórico fecha",
        formato_nivel(
            posicion[
                "max"
            ]
        ),
    )


    h5.metric(
        "Posición rango",
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
    # LLUVIA + CAUDAL
    # ========================================================

    lluvia_col, caudal_col = st.columns(
        [
            1,
            1,
        ]
    )


    # ========================================================
    # LLUVIA
    # ========================================================

    with lluvia_col:

        st.subheader(
            "🌧️ Lluvia"
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


            lluvia_cards = st.columns(
                [
                    1,
                    1,
                    1,
                    1,
                ]
            )


            with lluvia_cards[
                0
            ]:

                tarjeta_compacta(
                    "3 días",
                    formato_numero(
                        rain3[
                            "precip_mm"
                        ].sum(),
                        1,
                    ),
                    "mm acumulados",
                )


            with lluvia_cards[
                1
            ]:

                tarjeta_compacta(
                    "7 días",
                    formato_numero(
                        rain7[
                            "precip_mm"
                        ].sum(),
                        1,
                    ),
                    "mm acumulados",
                )


            with lluvia_cards[
                2
            ]:

                tarjeta_compacta(
                    "15 días",
                    formato_numero(
                        rain15[
                            "precip_mm"
                        ].sum(),
                        1,
                    ),
                    "mm acumulados",
                )


            with lluvia_cards[
                3
            ]:

                tarjeta_compacta(
                    "Máximo diario",
                    formato_numero(
                        rain15[
                            "precip_mm"
                        ].max(),
                        1,
                    ),
                    "mm/día",
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

                height=265,

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
                "Sin datos de lluvia."
            )


    # ========================================================
    # CAUDAL
    # ========================================================

    with caudal_col:

        st.subheader(
            "💧 Caudal"
        )


        caudal_cards = st.columns(
            [
                1.15,
                1,
                1,
                1.15,
            ]
        )


        with caudal_cards[
            0
        ]:

            tarjeta_compacta(
                "Actual",
                formato_caudal(
                    tq[
                        "actual"
                    ]
                ),
                "m³/s",
            )


        with caudal_cards[
            1
        ]:

            tarjeta_compacta(
                "Δ 3 días",
                (
                    formato_numero(
                        tq[
                            "delta_3"
                        ],
                        0,
                    )
                    if tq[
                        "delta_3"
                    ]
                    is not None
                    else "--"
                ),
                "m³/s",
            )


        with caudal_cards[
            2
        ]:

            tarjeta_compacta(
                "Δ 7 días",
                (
                    formato_numero(
                        tq[
                            "delta_7"
                        ],
                        0,
                    )
                    if tq[
                        "delta_7"
                    ]
                    is not None
                    else "--"
                ),
                "m³/s",
            )


        with caudal_cards[
            3
        ]:

            tarjeta_compacta(
                "Tendencia",
                tq[
                    "estado"
                ],
                "",
                trend=True,
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

                    name="Caudal",
                )
            )


            qfig.update_layout(

                height=265,

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
    # ESTACIONES AGUAS ARRIBA
    # ========================================================

    st.subheader(
        "🌊 Estaciones aguas arriba"
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


        estado = info.get(
            "estado",
            "Sin datos",
        )


        station_rows.append(
            {
                "Estación":
                    nombre,

                "Nivel actual":
                    (
                        formato_nivel(
                            actual
                        )
                        if actual
                        is not None
                        else "--"
                    ),

                "Tendencia":
                    estado,

                "Indicador":
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
    )


    # ========================================================
    # DETALLE 15 DÍAS
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
    # IMPORTANCIA
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

            **60 días:** escenario de estrés histórico.

            Cada actualización vuelve a utilizar la última medición
            real disponible de San Nicolás como punto de partida.
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
# PIE
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
