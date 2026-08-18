import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime
from zoneinfo import ZoneInfo


from src.ina import observed

from src.model import (
    train,
    predict,
)

from src.exogenous import (
    get_exogenous_data,
)

from src.upstream import (
    get_upstream_history,
)

from src.stress_ui import (
    render_stress_scenario,
)


# ============================================================
# PARANÁ · SAN NICOLÁS
# APP V11.8
# ============================================================

APP_VERSION = "V11.8"


# ============================================================
# HORIZONTES
# ============================================================

FORECAST_DAYS = 15
TREND_DAYS = 30
STRESS_DAYS = 60


# ============================================================
# HISTORIAL COMPLETO
# ============================================================

HISTORY_START = "1900-01-01"


# ============================================================
# ESCALA DE NIVEL
# SIEMPRE 0–7 m
# ============================================================

Y_MIN = 0.0
Y_MAX = 7.0
Y_STEP = 0.5


# ============================================================
# HORA ARGENTINA
# ============================================================

ARG_TZ = ZoneInfo(
    "America/Argentina/Buenos_Aires"
)


def ahora_argentina():

    return datetime.now(
        ARG_TZ
    )


def hoy_argentina():

    return (
        ahora_argentina()
        .date()
    )


# ============================================================
# PERÍODO AUTOMÁTICO
# 01/01 DEL AÑO ACTUAL → HOY
# ============================================================

HOY = hoy_argentina()

FECHA_INICIO_MODELO = HOY.replace(
    month=1,
    day=1,
)

FECHA_FIN_MODELO = HOY


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# OPTIMIZADO PARA PC + CELULAR
# ============================================================

st.markdown(
    """
    <style>

    /* ==============================================
       CONTENEDOR GENERAL
       ============================================== */

    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }


    /* ==============================================
       MÉTRICAS
       ============================================== */

    [data-testid="stMetric"] {
        padding: 0.35rem 0.45rem;
        border: 1px solid rgba(120,120,120,0.18);
        border-radius: 0.65rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.30rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.76rem;
    }


    /* ==============================================
       TÍTULOS
       ============================================== */

    h1 {
        font-size: 2rem;
        margin-bottom: 0.1rem;
    }

    h2 {
        font-size: 1.45rem;
        margin-top: 1.1rem;
        margin-bottom: 0.6rem;
    }

    h3 {
        font-size: 1.1rem;
    }


    /* ==============================================
       CAPTIONS
       ============================================== */

    [data-testid="stCaptionContainer"] {
        font-size: 0.82rem;
    }


    /* ==============================================
       BOTONES
       ============================================== */

    .stButton > button {
        min-height: 3rem;
        font-weight: 700;
        border-radius: 0.7rem;
    }


    /* ==============================================
       CELULAR
       ============================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-top: 0.6rem;
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }

        h1 {
            font-size: 1.55rem !important;
        }

        h2 {
            font-size: 1.2rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
        }

        [data-testid="stCaptionContainer"] {
            font-size: 0.76rem !important;
        }

        .stButton > button {
            width: 100%;
            min-height: 3.2rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "🌊 PARANÁ · SAN NICOLÁS"
)

st.caption(
    f"{APP_VERSION} · "
    "Monitoreo, pronóstico y análisis hidrométrico experimental"
)

st.markdown(
    """
    Seguimiento del río Paraná en **San Nicolás de los Arroyos**.

    **Nivel INA · lluvia · caudal · estaciones aguas arriba ·
    pronóstico 15/30 días · históricos · escenarios severos**
    """
)


# ============================================================
# PERÍODO AUTOMÁTICO EN PANTALLA
# ============================================================

p1, p2, p3 = st.columns(
    3
)

p1.metric(
    "Inicio automático",
    FECHA_INICIO_MODELO.strftime(
        "%d/%m/%Y"
    ),
)

p2.metric(
    "Fecha actual",
    FECHA_FIN_MODELO.strftime(
        "%d/%m/%Y"
    ),
)

p3.metric(
    "Período del modelo",
    (
        f"{(
            FECHA_FIN_MODELO
            - FECHA_INICIO_MODELO
        ).days + 1} días"
    ),
)


st.caption(
    "El período de entrenamiento se actualiza automáticamente "
    "desde el **1 de enero del año actual hasta hoy**."
)


# ============================================================
# BOTÓN PRINCIPAL
# ============================================================

actualizar = st.button(
    "🔄 Actualizar datos y pronóstico",
    use_container_width=True,
    type="primary",
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_fechas(
    df,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
    ):

        return pd.DataFrame()

    if df.empty:

        return df.copy()

    x = df.copy()

    if "datetime" not in x.columns:

        return x

    x[
        "datetime"
    ] = pd.to_datetime(
        x[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    x = x.dropna(
        subset=[
            "datetime"
        ]
    )

    try:

        x[
            "datetime"
        ] = (
            x[
                "datetime"
            ]
            .dt
            .tz_convert(
                "America/Argentina/Buenos_Aires"
            )
            .dt
            .tz_localize(
                None
            )
        )

    except Exception:

        try:

            x[
                "datetime"
            ] = (
                x[
                    "datetime"
                ]
                .dt
                .tz_localize(
                    None
                )
            )

        except Exception:

            pass

    return (
        x
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PREPARAR DATOS INA
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

    x[
        "datetime"
    ] = pd.to_datetime(
        x[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    x = (
        x
        .dropna(
            subset=[
                "datetime",
                "nivel",
            ]
        )
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
# FORMATO NUMÉRICO
# ============================================================

def formato_numero(
    value,
    decimals=0,
):

    try:

        if pd.isna(
            value
        ):

            return "--"

        text = (
            f"{float(value):,.{decimals}f}"
        )

        return (
            text
            .replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )

    except Exception:

        return "--"


# ============================================================
# ESCALA 0–7
# ============================================================

def aplicar_escala_nivel(
    fig,
):

    fig.update_yaxes(
        title_text=(
            "Nivel hidrométrico (m)"
        ),
        range=[
            Y_MIN,
            Y_MAX,
        ],
        tick0=0,
        dtick=Y_STEP,
        autorange=False,
        fixedrange=False,
    )

    return fig


# ============================================================
# EJE FECHA
# ============================================================

def aplicar_eje_fecha(
    fig,
    intervalo_dias=2,
    formato="%d/%m",
):

    fig.update_xaxes(
        title_text="Fecha",
        type="date",
        tickformat=formato,
        dtick=(
            intervalo_dias
            * 24
            * 60
            * 60
            * 1000
        ),
        tickangle=0,
    )

    return fig


# ============================================================
# INCERTIDUMBRE
# ============================================================

def obtener_margen_incertidumbre(
    forecast,
):

    if (
        forecast is None
        or not isinstance(
            forecast,
            pd.DataFrame,
        )
        or forecast.empty
    ):

        return pd.Series(
            dtype=float
        )

    if (
        "uncertainty_margin"
        in forecast.columns
    ):

        return pd.to_numeric(
            forecast[
                "uncertainty_margin"
            ],
            errors="coerce",
        )

    if (
        "upper"
        in forecast.columns
        and "lower"
        in forecast.columns
    ):

        upper = pd.to_numeric(
            forecast[
                "upper"
            ],
            errors="coerce",
        )

        lower = pd.to_numeric(
            forecast[
                "lower"
            ],
            errors="coerce",
        )

        return (
            upper
            - lower
        ) / 2.0

    return pd.Series(
        np.nan,
        index=forecast.index,
        dtype=float,
    )


# ============================================================
# BANDA 80 %
# ============================================================

def agregar_banda_incertidumbre(
    fig,
    forecast,
):

    if (
        forecast is None
        or not isinstance(
            forecast,
            pd.DataFrame,
        )
        or forecast.empty
        or "upper"
        not in forecast.columns
        or "lower"
        not in forecast.columns
    ):

        return

    f = normalizar_fechas(
        forecast
    )

    f[
        "upper"
    ] = pd.to_numeric(
        f[
            "upper"
        ],
        errors="coerce",
    )

    f[
        "lower"
    ] = pd.to_numeric(
        f[
            "lower"
        ],
        errors="coerce",
    )

    fig.add_trace(
        go.Scatter(
            x=f[
                "datetime"
            ],
            y=f[
                "upper"
            ],
            mode="lines",
            line=dict(
                width=1,
                color=(
                    "rgba(255,165,0,0.25)"
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=f[
                "datetime"
            ],
            y=f[
                "lower"
            ],
            mode="lines",
            line=dict(
                width=1,
                color=(
                    "rgba(255,165,0,0.25)"
                ),
            ),
            fill="tonexty",
            fillcolor=(
                "rgba(255,165,0,0.10)"
            ),
            name=(
                "Banda experimental 80%"
            ),
            hoverinfo="skip",
        )
    )


# ============================================================
# PRONÓSTICO
# ============================================================

def agregar_pronostico(
    fig,
    forecast,
    nombre,
    dash=None,
):

    if (
        forecast is None
        or not isinstance(
            forecast,
            pd.DataFrame,
        )
        or forecast.empty
    ):

        return

    f = normalizar_fechas(
        forecast
    )

    for col in [
        "prediction",
        "lower",
        "upper",
        "nivel_base",
        "variacion_dia",
        "precip_mm",
        "caudal_m3s",
    ]:

        if col not in f.columns:

            f[
                col
            ] = np.nan

        f[
            col
        ] = pd.to_numeric(
            f[
                col
            ],
            errors="coerce",
        )

    f[
        "uncertainty_margin"
    ] = (
        obtener_margen_incertidumbre(
            f
        )
    )

    customdata = np.column_stack(
        [
            f[
                "lower"
            ],

            f[
                "upper"
            ],

            f[
                "uncertainty_margin"
            ],

            f[
                "nivel_base"
            ],

            f[
                "variacion_dia"
            ],

            f[
                "precip_mm"
            ],

            f[
                "caudal_m3s"
            ],
        ]
    )

    line_config = {
        "width":
            3,
    }

    if dash:

        line_config[
            "dash"
        ] = dash

    fig.add_trace(
        go.Scatter(
            x=f[
                "datetime"
            ],
            y=f[
                "prediction"
            ],
            customdata=
                customdata,
            mode="lines+markers",
            name=nombre,
            line=line_config,
            marker=dict(
                size=6,
            ),
            hovertemplate=(
                "<b>%{x|%d/%m/%Y}</b>"
                "<br>Nivel: %{y:.2f} m"
                "<br>Inferior: %{customdata[0]:.2f} m"
                "<br>Superior: %{customdata[1]:.2f} m"
                "<br>Incertidumbre: ±%{customdata[2]:.2f} m"
                "<br>Nivel base: %{customdata[3]:.2f} m"
                "<br>Δ diario: %{customdata[4]:+.3f} m"
                "<br>Lluvia: %{customdata[5]:.1f} mm"
                "<br>Caudal: %{customdata[6]:,.0f} m³/s"
                "<extra></extra>"
            ),
        )
    )


# ============================================================
# TENDENCIA CAUDAL
# ============================================================

def calcular_tendencia_caudal(
    df_caudal,
):

    resultado = {
        "actual": None,
        "delta_3": None,
        "delta_7": None,
        "pct_7": None,
        "estado": "Sin datos",
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

    q = df_caudal.copy()

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

    if q.empty:

        return resultado

    valores = (
        q[
            "caudal_m3s"
        ]
        .to_numpy(
            dtype=float
        )
    )

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

        q7 = float(
            valores[
                -8
            ]
        )

        resultado[
            "delta_7"
        ] = (
            actual
            - q7
        )

        if q7 != 0:

            resultado[
                "pct_7"
            ] = (
                resultado[
                    "delta_7"
                ]
                / q7
                * 100.0
            )

    ultimos = valores[
        -min(
            7,
            len(
                valores
            ),
        ):
    ]

    pendiente = 0.0

    if len(
        ultimos
    ) >= 3:

        try:

            pendiente = float(
                np.polyfit(
                    np.arange(
                        len(
                            ultimos
                        )
                    ),
                    ultimos,
                    1,
                )[0]
            )

        except Exception:

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
        ] = "Creciente"

    elif pendiente < -umbral:

        resultado[
            "estado"
        ] = "Bajante"

    else:

        resultado[
            "estado"
        ] = "Estable"

    return resultado


# ============================================================
# TENDENCIA 30 DÍAS
# ============================================================

def calcular_tendencia_30_dias(
    df,
    forecast30,
):

    resultado = {
        "estado": "Sin datos",
        "nivel_actual": None,
        "nivel_dia_15": None,
        "nivel_dia_30": None,
        "cambio_30": None,
        "cambio_pct": None,
    }

    if (
        df is None
        or df.empty
        or "nivel"
        not in df.columns
    ):

        return resultado

    niveles = (
        pd.to_numeric(
            df[
                "nivel"
            ],
            errors="coerce",
        )
        .dropna()
    )

    if niveles.empty:

        return resultado

    nivel_actual = float(
        niveles.iloc[
            -1
        ]
    )

    resultado[
        "nivel_actual"
    ] = nivel_actual

    if (
        forecast30 is None
        or not isinstance(
            forecast30,
            pd.DataFrame,
        )
        or forecast30.empty
        or "prediction"
        not in forecast30.columns
    ):

        return resultado

    serie = forecast30.copy()

    serie[
        "prediction"
    ] = pd.to_numeric(
        serie[
            "prediction"
        ],
        errors="coerce",
    )

    serie = serie.dropna(
        subset=[
            "prediction"
        ]
    )

    if serie.empty:

        return resultado

    if len(
        serie
    ) >= 15:

        nivel15 = float(
            serie[
                "prediction"
            ]
            .iloc[
                14
            ]
        )

    else:

        nivel15 = float(
            serie[
                "prediction"
            ]
            .iloc[
                -1
            ]
        )

    nivel30 = float(
        serie[
            "prediction"
        ]
        .iloc[
            -1
        ]
    )

    cambio30 = (
        nivel30
        - nivel_actual
    )

    resultado[
        "nivel_dia_15"
    ] = nivel15

    resultado[
        "nivel_dia_30"
    ] = nivel30

    resultado[
        "cambio_30"
    ] = cambio30

    if nivel_actual != 0:

        resultado[
            "cambio_pct"
        ] = (
            cambio30
            / nivel_actual
            * 100.0
        )

    if cambio30 >= 0.30:

        resultado[
            "estado"
        ] = "Creciente"

    elif cambio30 <= -0.30:

        resultado[
            "estado"
        ] = "Bajante"

    else:

        resultado[
            "estado"
        ] = "Estable"

    return resultado


# ============================================================
# ENVOLVENTE HISTÓRICA
# ============================================================

def construir_envolvente_historica(
    df_historico,
    fechas_objetivo,
):

    if (
        df_historico is None
        or not isinstance(
            df_historico,
            pd.DataFrame,
        )
        or df_historico.empty
        or "datetime"
        not in df_historico.columns
        or "nivel"
        not in df_historico.columns
    ):

        return pd.DataFrame()

    hist = df_historico.copy()

    hist[
        "datetime"
    ] = pd.to_datetime(
        hist[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    hist[
        "nivel"
    ] = pd.to_numeric(
        hist[
            "nivel"
        ],
        errors="coerce",
    )

    hist = hist.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    if hist.empty:

        return pd.DataFrame()

    hist[
        "mes_dia"
    ] = (
        hist[
            "datetime"
        ]
        .dt
        .strftime(
            "%m-%d"
        )
    )

    resumen = (
        hist
        .groupby(
            "mes_dia"
        )[
            "nivel"
        ]
        .agg(
            nivel_min_historico="min",
            nivel_max_historico="max",
            nivel_promedio_historico="mean",
            registros="count",
        )
        .reset_index()
    )

    fechas = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    fechas_objetivo,
                    errors="coerce",
                    utc=True,
                )
        }
    )

    fechas[
        "mes_dia"
    ] = (
        fechas[
            "datetime"
        ]
        .dt
        .strftime(
            "%m-%d"
        )
    )

    resultado = fechas.merge(
        resumen,
        on="mes_dia",
        how="left",
    )

    return normalizar_fechas(
        resultado
    )


# ============================================================
# CARGA COMPLETA
# ============================================================

def actualizar_sistema():

    inicio = (
        FECHA_INICIO_MODELO
        .strftime(
            "%Y-%m-%d"
        )
    )

    fin = (
        FECHA_FIN_MODELO
        .strftime(
            "%Y-%m-%d"
        )
    )

    # ========================================================
    # NIVEL ACTUAL / AÑO
    # ========================================================

    with st.spinner(
        "Consultando nivel del INA..."
    ):

        df_ina, error_ina = observed(
            inicio,
            fin,
        )

    if error_ina:

        raise RuntimeError(
            error_ina
        )

    df = preparar_datos(
        df_ina
    )

    if df.empty:

        raise RuntimeError(
            "No se obtuvieron observaciones "
            "válidas del INA."
        )

    # ========================================================
    # HISTORIAL COMPLETO
    # PARA MÁXIMOS / MÍNIMOS / ESCENARIOS
    # ========================================================

    with st.spinner(
        "Consultando historial hidrométrico..."
    ):

        try:

            hist_raw, hist_error = (
                observed(
                    HISTORY_START,
                    fin,
                )
            )

            df_historico_total = (
                preparar_datos(
                    hist_raw
                )
            )

            if (
                hist_error
                or df_historico_total.empty
            ):

                df_historico_total = (
                    df.copy()
                )

        except Exception:

            df_historico_total = (
                df.copy()
            )

    # ========================================================
    # LLUVIA + CAUDAL
    # ========================================================

    with st.spinner(
        "Consultando lluvia y caudal..."
    ):

        try:

            (
                exog_history,
                exog_future,
                exog_meta,
            ) = get_exogenous_data(
                inicio,
                fin,
                TREND_DAYS,
            )

        except Exception as exc:

            exog_history = (
                pd.DataFrame()
            )

            exog_future = (
                pd.DataFrame()
            )

            exog_meta = {
                "error":
                    str(
                        exc
                    )
            }

    # ========================================================
    # AGUAS ARRIBA
    # ========================================================

    with st.spinner(
        "Consultando estaciones aguas arriba..."
    ):

        try:

            (
                upstream_history,
                upstream_meta,
            ) = get_upstream_history(
                inicio,
                fin,
            )

        except Exception as exc:

            upstream_history = (
                pd.DataFrame()
            )

            upstream_meta = {
                "error":
                    str(
                        exc
                    )
            }

    # ========================================================
    # MODELO
    # ========================================================

    with st.spinner(
        "Entrenando modelo y calculando pronóstico..."
    ):

        models, metrics = train(
            df,
            exog_history=
                exog_history,
            upstream_history=
                upstream_history,
        )

        forecast30 = predict(
            df=df,
            models=models,
            days=TREND_DAYS,
            exog_future=
                exog_future,
            upstream_future=
                None,
        )

    forecast30 = normalizar_fechas(
        forecast30
    )

    forecast = (
        forecast30
        .head(
            FORECAST_DAYS
        )
        .copy()
    )

    # ========================================================
    # SESSION
    # ========================================================

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
        "df_historico_total"
    ] = df_historico_total

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
        "actualizado"
    ] = ahora_argentina()


# ============================================================
# EJECUTAR ACTUALIZACIÓN
# ============================================================

if actualizar:

    try:

        actualizar_sistema()

        st.success(
            "✅ Datos y pronóstico actualizados correctamente."
        )

    except Exception as exc:

        st.error(
            "No fue posible actualizar el sistema: "
            f"{exc}"
        )


# ============================================================
# PRIMER INGRESO
# ============================================================

if (
    "datos"
    not in st.session_state
):

    st.info(
        "Presione **Actualizar datos y pronóstico** "
        "para cargar la información actual."
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

    df_historico_total = (
        st.session_state.get(
            "df_historico_total",
            df,
        )
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

    upstream_history = (
        st.session_state.get(
            "upstream_history",
            pd.DataFrame(),
        )
    )

    upstream_meta = (
        st.session_state.get(
            "upstream_meta",
            {},
        )
    )

    actualizado = st.session_state.get(
        "actualizado"
    )

    # ========================================================
    # NORMALIZACIÓN
    # ========================================================

    df_plot = normalizar_fechas(
        df
    )

    forecast = normalizar_fechas(
        forecast
    )

    forecast30 = normalizar_fechas(
        forecast30
    )

    exog_history_plot = (
        normalizar_fechas(
            exog_history
        )
    )

    exog_future_plot = (
        normalizar_fechas(
            exog_future
        )
    )

    # ========================================================
    # NIVEL ACTUAL
    # ========================================================

    nivel_actual = float(
        pd.to_numeric(
            df[
                "nivel"
            ],
            errors="coerce",
        )
        .dropna()
        .iloc[
            -1
        ]
    )

    ultima_fecha = (
        df_plot[
            "datetime"
        ]
        .iloc[
            -1
        ]
    )

    # ========================================================
    # SITUACIÓN ACTUAL
    # ========================================================

    st.subheader(
        "📊 Situación actual"
    )

    m1, m2, m3, m4 = st.columns(
        4
    )

    m1.metric(
        "Nivel",
        f"{nivel_actual:.2f} m",
    )

    m2.metric(
        "Fecha medición",
        ultima_fecha.strftime(
            "%d/%m/%Y"
        ),
    )

    m3.metric(
        "Mínimo año",
        f"{df['nivel'].min():.2f} m",
    )

    m4.metric(
        "Máximo año",
        f"{df['nivel'].max():.2f} m",
    )

    st.caption(
        f"Datos del modelo: "
        f"**{FECHA_INICIO_MODELO.strftime('%d/%m/%Y')} → "
        f"{FECHA_FIN_MODELO.strftime('%d/%m/%Y')}**"
    )

    # ========================================================
    # ESTADO COMPACTO
    # ========================================================

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
                    ]
                    .notna()
                    .any()
                )
            ]
        )

    e1, e2, e3, e4 = st.columns(
        4
    )

    e1.caption(
        "✅ **INA**"
    )

    e2.caption(
        (
            "✅ **Lluvia**"
            if (
                isinstance(
                    exog_future,
                    pd.DataFrame,
                )
                and not exog_future.empty
            )
            else
            "⚠️ **Lluvia**"
        )
    )

    e3.caption(
        (
            "✅ **Caudal**"
            if (
                isinstance(
                    exog_history,
                    pd.DataFrame,
                )
                and "caudal_m3s"
                in exog_history.columns
                and exog_history[
                    "caudal_m3s"
                ]
                .notna()
                .any()
            )
            else
            "⚠️ **Caudal**"
        )
    )

    e4.caption(
        f"✅ **Aguas arriba:** "
        f"{estaciones_disponibles}"
    )

    st.divider()

    # ========================================================
    # PRONÓSTICO 15 DÍAS
    # ========================================================

    st.subheader(
        "📈 Pronóstico · 15 días"
    )

    fig15 = go.Figure()

    obs = df_plot.tail(
        45
    )

    fig15.add_trace(
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
                width=2,
                color=(
                    "rgba(120,120,120,0.55)"
                ),
            ),
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>Observado: %{y:.2f} m"
                "<extra></extra>"
            ),
        )
    )

    fig15.add_hline(
        y=nivel_actual,
        line_dash="dash",
        line_width=2,
        line_color="black",
        annotation_text=(
            f"Actual {nivel_actual:.2f} m"
        ),
    )

    if not forecast.empty:

        agregar_banda_incertidumbre(
            fig15,
            forecast,
        )

        agregar_pronostico(
            fig15,
            forecast,
            "Pronóstico 1–15 días",
        )

    fig15.update_layout(
        height=520,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.08,
        ),
        margin=dict(
            l=5,
            r=5,
            t=30,
            b=5,
        ),
    )

    aplicar_escala_nivel(
        fig15
    )

    aplicar_eje_fecha(
        fig15,
        intervalo_dias=2,
    )

    st.plotly_chart(
        fig15,
        use_container_width=True,
    )

    # ========================================================
    # INCERTIDUMBRE
    # ========================================================

    if not forecast.empty:

        margen = (
            obtener_margen_incertidumbre(
                forecast
            )
            .dropna()
        )

        if not margen.empty:

            st.caption(
                "Banda experimental 80% · "
                f"Día 1: **±{margen.iloc[0]:.2f} m** · "
                f"Día 15: **±{margen.iloc[-1]:.2f} m** · "
                "máximo **±0,35 m**."
            )

    # ========================================================
    # TABLA DIARIA
    # ========================================================

    if not forecast.empty:

        with st.expander(
            "🔎 Ver pronóstico diario"
        ):

            tabla = forecast.copy()

            tabla[
                "Día"
            ] = np.arange(
                1,
                len(
                    tabla
                )
                + 1,
            )

            tabla[
                "Fecha"
            ] = (
                tabla[
                    "datetime"
                ]
                .dt
                .strftime(
                    "%d/%m/%Y"
                )
            )

            tabla[
                "Base"
            ] = (
                pd.to_numeric(
                    tabla[
                        "nivel_base"
                    ],
                    errors="coerce",
                )
                .round(2)
            )

            tabla[
                "Nivel"
            ] = (
                pd.to_numeric(
                    tabla[
                        "prediction"
                    ],
                    errors="coerce",
                )
                .round(2)
            )

            tabla[
                "Δ"
            ] = (
                pd.to_numeric(
                    tabla[
                        "variacion_dia"
                    ],
                    errors="coerce",
                )
                .round(3)
            )

            tabla[
                "Lluvia"
            ] = (
                pd.to_numeric(
                    tabla[
                        "precip_mm"
                    ],
                    errors="coerce",
                )
                .round(1)
            )

            tabla[
                "Caudal"
            ] = (
                pd.to_numeric(
                    tabla[
                        "caudal_m3s"
                    ],
                    errors="coerce",
                )
                .round(0)
            )

            tabla[
                "±"
            ] = (
                obtener_margen_incertidumbre(
                    tabla
                )
                .round(2)
            )

            st.dataframe(
                tabla[
                    [
                        "Día",
                        "Fecha",
                        "Base",
                        "Lluvia",
                        "Caudal",
                        "Δ",
                        "Nivel",
                        "±",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # TENDENCIA 30 DÍAS
    # ========================================================

    st.subheader(
        "🧭 Tendencia · 30 días"
    )

    tendencia30 = (
        calcular_tendencia_30_dias(
            df,
            forecast30,
        )
    )

    t1, t2, t3, t4 = st.columns(
        4
    )

    t1.metric(
        "Tendencia",
        tendencia30.get(
            "estado",
            "--",
        ),
    )

    nivel15 = tendencia30.get(
        "nivel_dia_15"
    )

    nivel30 = tendencia30.get(
        "nivel_dia_30"
    )

    cambio30 = tendencia30.get(
        "cambio_30"
    )

    pct30 = tendencia30.get(
        "cambio_pct"
    )

    t2.metric(
        "Día 15",
        (
            f"{nivel15:.2f} m"
            if nivel15
            is not None
            else "--"
        ),
    )

    t3.metric(
        "Día 30",
        (
            f"{nivel30:.2f} m"
            if nivel30
            is not None
            else "--"
        ),
    )

    if cambio30 is not None:

        texto_cambio = (
            f"{cambio30:+.2f} m"
        )

        if pct30 is not None:

            texto_cambio += (
                f" ({pct30:+.1f}%)"
            )

    else:

        texto_cambio = "--"

    t4.metric(
        "Cambio",
        texto_cambio,
    )

    if not forecast30.empty:

        fig30 = go.Figure()

        obs30 = df_plot.tail(
            30
        )

        fig30.add_trace(
            go.Scatter(
                x=obs30[
                    "datetime"
                ],
                y=obs30[
                    "nivel"
                ],
                mode="lines",
                name="Observado",
                line=dict(
                    color=(
                        "rgba(120,120,120,0.50)"
                    ),
                    width=2,
                ),
            )
        )

        fig30.add_hline(
            y=nivel_actual,
            line_dash="dash",
            line_width=2,
            line_color="black",
            annotation_text=(
                f"Actual {nivel_actual:.2f} m"
            ),
        )

        agregar_banda_incertidumbre(
            fig30,
            forecast30,
        )

        agregar_pronostico(
            fig30,
            forecast30.head(
                15
            ),
            "Pronóstico 1–15 días",
        )

        if len(
            forecast30
        ) > 15:

            agregar_pronostico(
                fig30,
                forecast30.iloc[
                    14:
                ],
                "Extensión 16–30 días",
                dash="dot",
            )

        fig30.update_layout(
            height=450,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.08,
            ),
            margin=dict(
                l=5,
                r=5,
                t=30,
                b=5,
            ),
        )

        aplicar_escala_nivel(
            fig30
        )

        aplicar_eje_fecha(
            fig30,
            intervalo_dias=3,
        )

        st.plotly_chart(
            fig30,
            use_container_width=True,
        )

    # ========================================================
    # EXTREMOS HISTÓRICOS
    # ========================================================

    st.subheader(
        "📏 Nivel vs. extremos históricos"
    )

    fechas_env = [
        ultima_fecha
    ]

    niveles_env = [
        nivel_actual
    ]

    if not forecast30.empty:

        fechas_env.extend(
            forecast30[
                "datetime"
            ].tolist()
        )

        niveles_env.extend(
            forecast30[
                "prediction"
            ].tolist()
        )

    envolvente = (
        construir_envolvente_historica(
            df_historico_total,
            fechas_env,
        )
    )

    if not envolvente.empty:

        cantidad = min(
            len(
                envolvente
            ),
            len(
                niveles_env
            ),
        )

        envolvente = (
            envolvente
            .head(
                cantidad
            )
            .copy()
        )

        envolvente[
            "nivel_dia"
        ] = niveles_env[
            :cantidad
        ]

        fig_hist = go.Figure()

        fig_hist.add_trace(
            go.Scatter(
                x=envolvente[
                    "datetime"
                ],
                y=envolvente[
                    "nivel_max_historico"
                ],
                mode="lines",
                name="Máximo histórico",
                line=dict(
                    color="crimson",
                    width=2,
                ),
            )
        )

        fig_hist.add_trace(
            go.Scatter(
                x=envolvente[
                    "datetime"
                ],
                y=envolvente[
                    "nivel_dia"
                ],
                mode="lines+markers",
                name="Nivel actual / proyectado",
                line=dict(
                    color="royalblue",
                    width=3,
                ),
            )
        )

        fig_hist.add_trace(
            go.Scatter(
                x=envolvente[
                    "datetime"
                ],
                y=envolvente[
                    "nivel_min_historico"
                ],
                mode="lines",
                name="Mínimo histórico",
                line=dict(
                    color="seagreen",
                    width=2,
                ),
            )
        )

        fig_hist.update_layout(
            height=430,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.08,
            ),
            margin=dict(
                l=5,
                r=5,
                t=30,
                b=5,
            ),
        )

        aplicar_escala_nivel(
            fig_hist
        )

        aplicar_eje_fecha(
            fig_hist,
            intervalo_dias=3,
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True,
        )

    # ========================================================
    # ESCENARIOS HISTÓRICOS 60 DÍAS
    # ========================================================

    render_stress_scenario(
        df=df,
        models=models,
        exog_history=
            exog_history,
        upstream_history=
            upstream_history,
    )

    # ========================================================
    # LLUVIA
    # ========================================================

    st.subheader(
        "🌧️ Lluvia prevista · 15 días"
    )

    if (
        not exog_future_plot.empty
        and "precip_mm"
        in exog_future_plot.columns
    ):

        rain = (
            exog_future_plot
            .head(
                15
            )
            .copy()
        )

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
            .clip(
                lower=0.0
            )
        )

        r1, r2, r3 = st.columns(
            3
        )

        r1.metric(
            "Acumulado",
            f"{rain['precip_mm'].sum():.1f} mm",
        )

        r2.metric(
            "Máximo diario",
            f"{rain['precip_mm'].max():.1f} mm",
        )

        r3.metric(
            "Días ≥ 1 mm",
            int(
                (
                    rain[
                        "precip_mm"
                    ]
                    >= 1.0
                )
                .sum()
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
                name="Lluvia",
            )
        )

        rain_fig.update_layout(
            height=280,
            yaxis_title=(
                "Precipitación (mm/día)"
            ),
            margin=dict(
                l=5,
                r=5,
                t=10,
                b=5,
            ),
        )

        aplicar_eje_fecha(
            rain_fig,
            intervalo_dias=2,
        )

        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )

    else:

        st.info(
            "Sin información de lluvia prevista."
        )

    # ========================================================
    # CAUDAL
    # ========================================================

    st.subheader(
        "💧 Caudal"
    )

    if (
        not exog_history_plot.empty
        and "caudal_m3s"
        in exog_history_plot.columns
        and exog_history_plot[
            "caudal_m3s"
        ]
        .notna()
        .any()
    ):

        q_hist = (
            exog_history_plot
            .dropna(
                subset=[
                    "caudal_m3s"
                ]
            )
            .copy()
        )

        tq = calcular_tendencia_caudal(
            q_hist
        )

        q1, q2, q3, q4 = st.columns(
            4
        )

        q1.metric(
            "Actual",
            (
                f"{formato_numero(tq['actual'], 0)} m³/s"
                if tq[
                    "actual"
                ]
                is not None
                else "--"
            ),
        )

        q2.metric(
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

        if tq[
            "delta_7"
        ] is not None:

            texto7 = (
                f"{tq['delta_7']:+,.0f}"
            )

            if tq[
                "pct_7"
            ] is not None:

                texto7 += (
                    f" ({tq['pct_7']:+.1f}%)"
                )

        else:

            texto7 = "--"

        q3.metric(
            "Δ 7 días",
            texto7,
        )

        q4.metric(
            "Tendencia",
            tq[
                "estado"
            ],
        )

        q_fig = go.Figure()

        q_fig.add_trace(
            go.Scatter(
                x=q_hist[
                    "datetime"
                ],
                y=q_hist[
                    "caudal_m3s"
                ],
                mode="lines",
                name="Caudal histórico",
            )
        )

        if (
            not exog_future_plot.empty
            and "caudal_m3s"
            in exog_future_plot.columns
            and exog_future_plot[
                "caudal_m3s"
            ]
            .notna()
            .any()
        ):

            q_fig.add_trace(
                go.Scatter(
                    x=exog_future_plot[
                        "datetime"
                    ],
                    y=exog_future_plot[
                        "caudal_m3s"
                    ],
                    mode="lines+markers",
                    line=dict(
                        dash="dash",
                    ),
                    name="Proyección",
                )
            )

        q_fig.update_layout(
            height=340,
            hovermode="x unified",
            yaxis_title="Caudal (m³/s)",
            legend=dict(
                orientation="h",
                y=1.05,
            ),
            margin=dict(
                l=5,
                r=5,
                t=20,
                b=5,
            ),
        )

        aplicar_eje_fecha(
            q_fig,
            intervalo_dias=7,
        )

        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )

    else:

        st.info(
            "Sin serie de caudal utilizable."
        )

    # ========================================================
    # IMPORTANCIA DEL MODELO
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
            "🧠 Variables más importantes"
        ):

            top_imp = (
                importance
                .head(
                    15
                )
                .copy()
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
                height=480,
                xaxis_title="Importancia relativa",
                margin=dict(
                    l=5,
                    r=5,
                    t=10,
                    b=5,
                ),
            )

            imp_fig.update_yaxes(
                autorange="reversed"
            )

            st.plotly_chart(
                imp_fig,
                use_container_width=True,
            )

    # ========================================================
    # DIAGNÓSTICO
    # ========================================================

    with st.expander(
        "🧪 Diagnóstico técnico"
    ):

        rmse = metrics.get(
            "RMSE"
        )

        limite = models.get(
            "daily_change_limit"
        )

        diagnostico = pd.DataFrame(
            [
                {
                    "Parámetro":
                        "App",

                    "Valor":
                        APP_VERSION,
                },

                {
                    "Parámetro":
                        "Modelo",

                    "Valor":
                        models.get(
                            "version",
                            "V11.7",
                        ),
                },

                {
                    "Parámetro":
                        "Inicio modelo",

                    "Valor":
                        FECHA_INICIO_MODELO.strftime(
                            "%d/%m/%Y"
                        ),
                },

                {
                    "Parámetro":
                        "Última fecha",

                    "Valor":
                        FECHA_FIN_MODELO.strftime(
                            "%d/%m/%Y"
                        ),
                },

                {
                    "Parámetro":
                        "Observaciones",

                    "Valor":
                        models.get(
                            "observations",
                            "--",
                        ),
                },

                {
                    "Parámetro":
                        "RMSE",

                    "Valor":
                        (
                            f"{float(rmse):.3f} m"
                            if rmse
                            is not None
                            else "--"
                        ),
                },

                {
                    "Parámetro":
                        "Límite Δ diario",

                    "Valor":
                        (
                            f"±{float(limite):.3f} m/día"
                            if limite
                            is not None
                            else "--"
                        ),
                },

                {
                    "Parámetro":
                        "Banda",

                    "Valor":
                        "80% · máximo ±0,35 m",
                },

                {
                    "Parámetro":
                        "Escala",

                    "Valor":
                        "0–7 m",
                },
            ]
        )

        st.dataframe(
            diagnostico,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # METODOLOGÍA
    # ========================================================

    with st.expander(
        "ℹ️ Cómo funciona"
    ):

        st.markdown(
            f"""
            **Período automático**

            El modelo utiliza automáticamente datos desde:

            **{FECHA_INICIO_MODELO.strftime('%d/%m/%Y')}**

            hasta:

            **{FECHA_FIN_MODELO.strftime('%d/%m/%Y')}**

            No es necesario seleccionar fechas manualmente.

            **15 días**

            El cálculo comienza en la última altura real disponible.
            Para cada nuevo día se consideran lluvia, caudal,
            evolución del nivel y señales aguas arriba.

            **30 días**

            Continúa la misma simulación. El nivel del día anterior
            es siempre la base del siguiente.

            **60 días**

            Los escenarios extremos utilizan el historial completo
            disponible y no solamente los datos del año actual.

            **Escala**

            Los gráficos de altura se mantienen siempre entre
            **0 y 7 metros**.
            """
        )

        st.warning(
            "La aplicación es experimental. "
            "No reemplaza pronósticos, alertas ni avisos oficiales."
        )

    # ========================================================
    # ACTUALIZACIÓN
    # ========================================================

    if actualizado:

        try:

            texto_actualizado = (
                actualizado
                .strftime(
                    "%d/%m/%Y %H:%M"
                )
            )

        except Exception:

            texto_actualizado = str(
                actualizado
            )

        st.caption(
            "Última actualización: "
            f"**{texto_actualizado}**"
        )


# ============================================================
# FUENTES
# ============================================================

st.divider()

st.markdown(
    """
    **Fuentes**

    Nivel hidrométrico y caudal: **Instituto Nacional del Agua (INA)**  
    Precipitación: **Open-Meteo**  
    Predicción y escenarios: **modelo experimental propio**
    """
)

st.warning(
    "Los resultados tienen carácter experimental e informativo. "
    "Ante situaciones de riesgo deben consultarse las "
    "comunicaciones oficiales."
)

st.caption(
    f"Paraná · San Nicolás {APP_VERSION} · "
    "15 días + 30 días + escenarios 60 días · "
    "Escala 0–7 m"
)
