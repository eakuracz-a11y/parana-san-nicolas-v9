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

APP_VERSION = "V11.0"


# ============================================================
# HORIZONTES
# ============================================================

FORECAST_DAYS = 15

TREND_DAYS = 30

STRESS_DAYS = 60


# ============================================================
# HISTÓRICO
# ============================================================

# Histórico extenso del nivel INA.
#
# Se intenta consultar desde una fecha muy antigua.
# INA devolverá únicamente los registros efectivamente
# disponibles para la serie.

FULL_HISTORY_START = "1900-01-01"


# Para entrenamiento + lluvia + caudal + estaciones aguas arriba
# utilizamos una ventana amplia pero operativamente razonable.

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
)


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.65rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.88rem;
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
    f"{APP_VERSION} · Plataforma pública experimental "
    "de monitoreo, pronóstico y análisis hidrométrico"
)

st.markdown(
    """
    Seguimiento del río Paraná en **San Nicolás de los Arroyos**.

    El sistema integra:

    **nivel real INA · lluvia del corredor aguas arriba · caudal ·
    estaciones aguas arriba · pronóstico recursivo diario**
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Consulta"
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


st.sidebar.subheader(
    "Horizontes"
)

st.sidebar.write(
    "Pronóstico principal: **15 días**"
)

st.sidebar.write(
    "Pronóstico extendido: **30 días**"
)

st.sidebar.write(
    "Escenario histórico severo: **60 días**"
)


st.sidebar.subheader(
    "Escala"
)

st.sidebar.write(
    "Nivel hidrométrico: **0–7 m**"
)


st.sidebar.divider()

st.sidebar.caption(
    "Nivel y caudal: INA"
)

st.sidebar.caption(
    "Precipitación: Open-Meteo"
)

st.sidebar.caption(
    "Modelo: experimental"
)


# ============================================================
# PREPARAR DATOS DE NIVEL
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
# NORMALIZAR NIVEL A FECHA DIARIA
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

        "pendiente":
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


    valores = q[
        "caudal_m3s"
    ].to_numpy(
        dtype=float
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

        delta7 = (
            actual
            - q7
        )

        resultado[
            "delta_7"
        ] = delta7

        if q7 != 0:

            resultado[
                "pct_7"
            ] = (
                delta7
                / q7
                * 100
            )


    ultimos = valores[
        -min(
            7,
            len(
                valores
            ),
        ):
    ]


    if len(
        ultimos
    ) >= 3:

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

    else:

        pendiente = 0.0


    resultado[
        "pendiente"
    ] = pendiente


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
        # NIVEL PARA VISUALIZACIÓN
        # ====================================================

        with st.spinner(
            "Consultando nivel actual de San Nicolás..."
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
                    "No se obtuvieron observaciones "
                    "válidas de San Nicolás."
                )

            else:

                # ============================================
                # HISTÓRICO COMPLETO DE NIVEL
                # ============================================

                with st.spinner(
                    "Consultando histórico hidrométrico..."
                ):

                    try:

                        df_hist_raw, error_hist = observed(
                            FULL_HISTORY_START,
                            fin,
                        )

                        if error_hist:

                            df_historico = (
                                df_visual.copy()
                            )

                        else:

                            df_historico = (
                                preparar_datos(
                                    df_hist_raw
                                )
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
                # INICIO PARA MODELO
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
                # NIVEL PARA ENTRENAMIENTO
                # ============================================

                try:

                    df_model_raw, error_model = observed(
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
                            "No fue posible obtener "
                            f"todas las variables externas: {exc}"
                        )


                # ============================================
                # ESTACIONES AGUAS ARRIBA
                # ============================================

                with st.spinner(
                    "Consultando y proyectando estaciones aguas arriba..."
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
                            "No fue posible obtener "
                            f"todas las estaciones aguas arriba: {exc}"
                        )


                # ============================================
                # ENTRENAMIENTO
                # ============================================

                with st.spinner(
                    "Entrenando modelo diario..."
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
                            "No fue posible entrenar "
                            f"el modelo: {exc}"
                        )


                # ============================================
                # PRONÓSTICO 30 DÍAS
                # ============================================

                forecast30 = pd.DataFrame()

                forecast15 = pd.DataFrame()


                if models:

                    with st.spinner(
                        "Generando pronóstico diario de 30 días..."
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
                # GUARDAR SESIÓN
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
                    "✅ Datos y modelo actualizados correctamente."
                )


# ============================================================
# ESTADO INICIAL
# ============================================================

if "datos" not in st.session_state:

    st.info(
        "Seleccione el período de visualización y presione "
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
    # ÚLTIMA OBSERVACIÓN REAL
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


    st.subheader(
        "📊 Situación observada"
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "Último nivel real",
        f"{nivel_actual:.2f} m",
    )


    c2.metric(
        "Mínimo período",
        f"{df['nivel'].min():.2f} m",
    )


    c3.metric(
        "Máximo período",
        f"{df['nivel'].max():.2f} m",
    )


    c4.metric(
        "Promedio período",
        f"{df['nivel'].mean():.2f} m",
    )


    try:

        if getattr(
            ultima_fecha,
            "tzinfo",
            None,
        ):

            fecha_obs = (
                ultima_fecha
                .tz_convert(
                    "America/Argentina/Buenos_Aires"
                )
                .strftime(
                    "%d/%m/%Y %H:%M"
                )
            )

        else:

            fecha_obs = (
                ultima_fecha
                .strftime(
                    "%d/%m/%Y"
                )
            )

    except Exception:

        fecha_obs = str(
            ultima_fecha
        )


    st.caption(
        f"Última medición real INA: **{fecha_obs}** · "
        f"Nivel utilizado como punto inicial del pronóstico: "
        f"**{nivel_actual:.2f} m**"
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


    st.caption(
        "🟢 **Estado del sistema**"
    )


    s1, s2, s3, s4 = st.columns(
        4
    )


    s1.caption(
        "**INA San Nicolás** · ✅ Disponible"
    )


    s2.caption(
        (
            "**Lluvia** · ✅ Disponible"
            if estado_lluvia
            else "**Lluvia** · ⚠️ Sin datos"
        )
    )


    s3.caption(
        (
            "**Caudal** · ✅ Disponible"
            if estado_caudal
            else "**Caudal** · ⚠️ Sin datos"
        )
    )


    s4.caption(
        "**Aguas arriba** · "
        f"✅ {estaciones_disponibles}/6"
    )


    st.divider()


    # ========================================================
    # PRONÓSTICO 15 DÍAS
    # ========================================================

    st.subheader(
        "📈 Pronóstico diario · 15 días"
    )


    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
    ):

        nivel15 = float(
            forecast[
                "prediction"
            ].iloc[
                -1
            ]
        )


        max15 = float(
            forecast[
                "prediction"
            ].max()
        )


        min15 = float(
            forecast[
                "prediction"
            ].min()
        )


        p1, p2, p3, p4 = st.columns(
            4
        )


        p1.metric(
            "Nivel actual",
            f"{nivel_actual:.2f} m",
        )


        p2.metric(
            "Nivel día 15",
            f"{nivel15:.2f} m",
            f"{nivel15 - nivel_actual:+.2f} m",
        )


        p3.metric(
            "Máximo previsto",
            f"{max15:.2f} m",
        )


        p4.metric(
            "Mínimo previsto",
            f"{min15:.2f} m",
        )


        fig15 = go.Figure()


        obs_recent = df.tail(
            45
        )


        fig15.add_trace(
            go.Scatter(
                x=obs_recent[
                    "datetime"
                ],
                y=obs_recent[
                    "nivel"
                ],
                mode="lines",
                name="Nivel observado",
                line=dict(
                    color="#444444",
                    width=2,
                ),
            )
        )


        fig15.add_trace(
            go.Scatter(
                x=forecast[
                    "datetime"
                ],
                y=forecast[
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


        fig15.add_trace(
            go.Scatter(
                x=forecast[
                    "datetime"
                ],
                y=forecast[
                    "lower"
                ],
                mode="lines",
                line=dict(
                    width=0,
                ),
                fill="tonexty",
                name="Intervalo experimental",
                hoverinfo="skip",
            )
        )


        fig15.add_trace(
            go.Scatter(
                x=forecast[
                    "datetime"
                ],
                y=forecast[
                    "prediction"
                ],
                mode="lines+markers",
                name="Pronóstico diario",
                line=dict(
                    color="#1f77b4",
                    width=3,
                ),
                marker=dict(
                    size=6,
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Nivel: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )


        fig15.update_layout(
            height=520,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.05,
            ),
        )


        fig15.update_xaxes(
            title_text="Fecha",
            tickformat="%d/%m/%Y",
        )


        fig15.update_yaxes(
            title_text="Nivel hidrométrico (m)",
            range=[
                Y_MIN,
                Y_MAX,
            ],
            dtick=Y_STEP,
        )


        st.plotly_chart(
            fig15,
            use_container_width=True,
        )


        # ====================================================
        # TABLA DE AUDITORÍA 15 DÍAS
        # ====================================================

        with st.expander(
            "🔎 Ver cálculo diario del pronóstico"
        ):

            tabla15 = forecast.copy()


            tabla15[
                "Fecha"
            ] = pd.to_datetime(
                tabla15[
                    "datetime"
                ]
            ).dt.strftime(
                "%d/%m/%Y"
            )


            tabla15[
                "Nivel base (m)"
            ] = tabla15[
                "nivel_base"
            ].round(
                2
            )


            tabla15[
                "Variación estimada (m)"
            ] = tabla15[
                "variacion_dia"
            ].round(
                3
            )


            tabla15[
                "Nivel resultante (m)"
            ] = tabla15[
                "prediction"
            ].round(
                2
            )


            tabla15[
                "Lluvia (mm)"
            ] = tabla15[
                "precip_mm"
            ].round(
                1
            )


            tabla15[
                "Caudal (m³/s)"
            ] = tabla15[
                "caudal_m3s"
            ].round(
                0
            )


            columnas = [

                "Fecha",
                "Nivel base (m)",
                "Lluvia (mm)",
                "Caudal (m³/s)",
                "Variación estimada (m)",
                "Nivel resultante (m)",
            ]


            st.dataframe(
                tabla15[
                    columnas
                ],
                use_container_width=True,
                hide_index=True,
            )


    else:

        st.warning(
            "No se encuentra disponible "
            "el pronóstico de 15 días."
        )


    # ========================================================
    # PRONÓSTICO 30 DÍAS
    # ========================================================

    st.subheader(
        "🧭 Pronóstico extendido · 30 días"
    )


    if (
        isinstance(
            forecast30,
            pd.DataFrame,
        )
        and not forecast30.empty
    ):

        nivel30 = float(
            forecast30[
                "prediction"
            ].iloc[
                -1
            ]
        )


        cambio30 = (
            nivel30
            - nivel_actual
        )


        if cambio30 >= 0.30:

            estado30 = "Creciente"

        elif cambio30 <= -0.30:

            estado30 = "Bajante"

        else:

            estado30 = "Estable"


        t1, t2, t3, t4 = st.columns(
            4
        )


        t1.metric(
            "Tendencia",
            estado30,
        )


        t2.metric(
            "Día 15",
            (
                f"{forecast30['prediction'].iloc[14]:.2f} m"
                if len(
                    forecast30
                ) >= 15
                else "Sin datos"
            ),
        )


        t3.metric(
            "Día 30",
            f"{nivel30:.2f} m",
        )


        t4.metric(
            "Cambio vs. actual",
            f"{cambio30:+.2f} m",
        )


        # ====================================================
        # ENVOLVENTE HISTÓRICA 30 DÍAS
        # ====================================================

        envelope30 = (
            calcular_envolvente_historica(
                df_historico,
                forecast30[
                    "datetime"
                ],
            )
        )


        forecast30_plot = (
            forecast30.copy()
        )


        if not envelope30.empty:

            forecast30_plot = (
                forecast30_plot
                .merge(
                    envelope30[
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


        fig30 = go.Figure()


        if (
            "nivel_max_historico"
            in forecast30_plot.columns
        ):

            fig30.add_trace(
                go.Scatter(
                    x=forecast30_plot[
                        "datetime"
                    ],
                    y=forecast30_plot[
                        "nivel_max_historico"
                    ],
                    mode="lines",
                    name="Máximo histórico del día",
                    line=dict(
                        color="#d62728",
                        width=2,
                    ),
                )
            )


        fig30.add_trace(
            go.Scatter(
                x=forecast30_plot[
                    "datetime"
                ],
                y=forecast30_plot[
                    "prediction"
                ],
                mode="lines+markers",
                name="Nivel proyectado",
                line=dict(
                    color="#1f77b4",
                    width=3,
                ),
                marker=dict(
                    size=5,
                ),
            )
        )


        if (
            "nivel_min_historico"
            in forecast30_plot.columns
        ):

            fig30.add_trace(
                go.Scatter(
                    x=forecast30_plot[
                        "datetime"
                    ],
                    y=forecast30_plot[
                        "nivel_min_historico"
                    ],
                    mode="lines",
                    name="Mínimo histórico del día",
                    line=dict(
                        color="#2ca02c",
                        width=2,
                    ),
                )
            )


        fig30.add_hline(
            y=nivel_actual,
            line_dash="dash",
            annotation_text=(
                f"Nivel real: "
                f"{nivel_actual:.2f} m"
            ),
        )


        fig30.update_layout(
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.06,
            ),
        )


        fig30.update_xaxes(
            title_text="Fecha",
            tickformat="%d/%m/%Y",
        )


        fig30.update_yaxes(
            title_text="Nivel hidrométrico (m)",
            range=[
                Y_MIN,
                Y_MAX,
            ],
            dtick=Y_STEP,
        )


        st.plotly_chart(
            fig30,
            use_container_width=True,
        )


        st.caption(
            "Azul: nivel proyectado. "
            "Rojo: máximo histórico registrado para ese día y mes. "
            "Verde: mínimo histórico registrado para ese día y mes."
        )


    else:

        st.info(
            "No se encuentra disponible "
            "el pronóstico de 30 días."
        )


    # ========================================================
    # LLUVIA FUTURA
    # ========================================================

    st.subheader(
        "🌧️ Precipitación utilizada por el modelo"
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


        rain15 = rain.head(
            FORECAST_DAYS
        )


        r1, r2, r3 = st.columns(
            3
        )


        r1.metric(
            "Acumulado 15 días",
            f"{rain15['precip_mm'].sum():.1f} mm",
        )


        r2.metric(
            "Máximo diario 15 días",
            f"{rain15['precip_mm'].max():.1f} mm",
        )


        r3.metric(
            "Días ≥ 1 mm",
            int(
                (
                    rain15[
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
                marker_color="#17becf",
            )
        )


        rain_fig.update_layout(
            height=320,
            yaxis_title="Precipitación (mm/día)",
        )


        rain_fig.update_xaxes(
            tickformat="%d/%m",
        )


        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )


        if "rain_source" in rain.columns:

            with st.expander(
                "Fuente de lluvia día por día"
            ):

                rain_table = rain[
                    [
                        "datetime",
                        "precip_mm",
                        "rain_source",
                    ]
                ].copy()


                rain_table[
                    "datetime"
                ] = pd.to_datetime(
                    rain_table[
                        "datetime"
                    ]
                ).dt.strftime(
                    "%d/%m/%Y"
                )


                rain_table.columns = [
                    "Fecha",
                    "Lluvia (mm)",
                    "Fuente",
                ]


                st.dataframe(
                    rain_table,
                    use_container_width=True,
                    hide_index=True,
                )


    else:

        st.info(
            "No se encuentra disponible "
            "la precipitación futura."
        )


    # ========================================================
    # CAUDAL
    # ========================================================

    st.subheader(
        "💧 Caudal utilizado por el modelo"
    )


    if (
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
    ):

        q_hist = (
            exog_history
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


        q2.metric(
            "Variación 3 días",
            (
                f"{tq['delta_3']:+,.0f} m³/s"
                if tq[
                    "delta_3"
                ]
                is not None
                else "Sin datos"
            ),
        )


        if tq[
            "delta_7"
        ] is not None:

            texto7 = (
                f"{tq['delta_7']:+,.0f} m³/s"
            )

            if tq[
                "pct_7"
            ] is not None:

                texto7 += (
                    f" ({tq['pct_7']:+.1f}%)"
                )

        else:

            texto7 = "Sin datos"


        q3.metric(
            "Variación 7 días",
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
                name="Caudal observado",
                line=dict(
                    color="#444444",
                ),
            )
        )


        if (
            isinstance(
                exog_future,
                pd.DataFrame,
            )
            and "caudal_m3s"
            in exog_future.columns
        ):

            q_fig.add_trace(
                go.Scatter(
                    x=exog_future[
                        "datetime"
                    ],
                    y=exog_future[
                        "caudal_m3s"
                    ],
                    mode="lines+markers",
                    name="Caudal proyectado",
                    line=dict(
                        color="#9467bd",
                        dash="dash",
                    ),
                )
            )


        q_fig.update_layout(
            height=360,
            hovermode="x unified",
            yaxis_title="Caudal (m³/s)",
        )


        q_fig.update_xaxes(
            tickformat="%d/%m",
        )


        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )


    else:

        st.info(
            "No se encuentra disponible "
            "una serie de caudal utilizable."
        )


    # ========================================================
    # ESTACIONES AGUAS ARRIBA
    # ========================================================

    st.subheader(
        "🌊 Estaciones aguas arriba"
    )


    if (
        isinstance(
            upstream_future,
            pd.DataFrame,
        )
        and not upstream_future.empty
    ):

        up_fig = go.Figure()


        level_cols = [
            c
            for c
            in upstream_future.columns
            if (
                c.startswith(
                    "nivel_"
                )
                and "_lag"
                not in c
                and "_diff"
                not in c
                and "_trend"
                not in c
                and "_mean"
                not in c
                and "_actual"
                not in c
                and "_next"
                not in c
            )
        ]


        for col in level_cols:

            nombre = (
                col
                .replace(
                    "nivel_",
                    ""
                )
                .replace(
                    "_",
                    " "
                )
                .title()
            )


            up_fig.add_trace(
                go.Scatter(
                    x=upstream_future[
                        "datetime"
                    ],
                    y=upstream_future[
                        col
                    ],
                    mode="lines",
                    name=nombre,
                )
            )


        up_fig.update_layout(
            height=420,
            hovermode="x unified",
            yaxis_title="Nivel hidrométrico (m)",
            legend=dict(
                orientation="h",
                y=1.08,
            ),
        )


        up_fig.update_xaxes(
            tickformat="%d/%m",
        )


        st.plotly_chart(
            up_fig,
            use_container_width=True,
        )


        st.caption(
            "Las trayectorias futuras de estaciones aguas arriba "
            "son proyecciones experimentales basadas en su "
            "comportamiento reciente y se utilizan como variables "
            "de entrada del modelo de San Nicolás."
        )


    else:

        st.info(
            "No se encuentran disponibles proyecciones "
            "de estaciones aguas arriba."
        )


    # ========================================================
    # ESCENARIO HISTÓRICO SEVERO 60 DÍAS
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
    # VARIABLES UTILIZADAS
    # ========================================================

    with st.expander(
        "🧠 Variables utilizadas por el modelo"
    ):

        rows = [

            {
                "Variable":
                    "Nivel diario San Nicolás",

                "Estado":
                    "✓ Utilizada",
            },

            {
                "Variable":
                    "Lluvia corredor Paraná",

                "Estado":
                    (
                        "✓ Utilizada"
                        if models.get(
                            "uses_rain",
                            False,
                        )
                        else "✗ No utilizada"
                    ),
            },

            {
                "Variable":
                    "Caudal",

                "Estado":
                    (
                        "✓ Utilizado"
                        if models.get(
                            "uses_caudal",
                            False,
                        )
                        else "✗ No utilizado"
                    ),
            },

            {
                "Variable":
                    "Niveles aguas arriba",

                "Estado":
                    (
                        "✓ Utilizados"
                        if models.get(
                            "uses_upstream",
                            False,
                        )
                        else "✗ No utilizados"
                    ),
            },
        ]


        for estacion, info in upstream_meta.items():

            rows.append(
                {
                    "Variable":
                        estacion,

                    "Estado":
                        (
                            "✓ Serie INA encontrada"
                            if info
                            is not None
                            else "✗ Sin serie"
                        ),
                }
            )


        st.dataframe(
            pd.DataFrame(
                rows
            ),
            use_container_width=True,
            hide_index=True,
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
                25
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
                    name="Importancia",
                )
            )


            imp_fig.update_layout(
                height=650,
                xaxis_title="Importancia relativa",
                yaxis_title="Variable",
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
        "ℹ️ Metodología, calidad y alcance"
    ):

        st.write(
            "**Pronóstico principal:** 15 días"
        )

        st.write(
            "**Pronóstico extendido experimental:** 30 días"
        )

        st.write(
            "**Escenario histórico severo:** 60 días"
        )

        st.write(
            "**Modelo:** Random Forest Regressor "
            "sobre variación diaria del nivel"
        )


        rmse = metrics.get(
            "RMSE"
        )


        if rmse is not None:

            st.write(
                "**RMSE de validación histórica:** "
                f"{float(rmse):.3f} m"
            )


        st.markdown(
            """
            ### Pronóstico de 15 días

            Parte de la **última medición real de San Nicolás**.

            Cada día se calcula utilizando:

            - nivel del día anterior,
            - lluvia prevista del corredor aguas arriba,
            - caudal disponible/proyectado,
            - niveles de estaciones aguas arriba,
            - variaciones y tendencias,
            - retardos históricos.

            El nivel calculado pasa a ser la base del día siguiente.

            ### Pronóstico de 30 días

            Utiliza la misma metodología recursiva.

            Cuando deja de existir pronóstico meteorológico directo,
            la lluvia se extiende mediante climatología histórica y
            debe interpretarse con mayor incertidumbre.

            ### Escenario de 60 días

            No es un pronóstico meteorológico.

            Es un escenario de estrés construido buscando condiciones
            históricamente severas de lluvia, caudal, niveles aguas arriba
            y respuesta de San Nicolás.
            """
        )


        st.warning(
            "Esta plataforma es experimental. "
            "No reemplaza información, avisos, alertas "
            "ni pronósticos emitidos por organismos oficiales."
        )


    # ========================================================
    # ACTUALIZACIÓN
    # ========================================================

    if actualizado:

        st.caption(
            "Última actualización de la plataforma: "
            f"{actualizado.strftime('%d/%m/%Y %H:%M')}"
        )


# ============================================================
# FUENTES
# ============================================================

st.divider()


st.markdown(
    """
    **Fuentes**

    Datos hidrométricos y caudal: **Instituto Nacional del Agua (INA)**  
    Precipitación: **Open-Meteo**  
    Predicción y escenarios: **modelo experimental propio**
    """
)


st.warning(
    "Los resultados de esta plataforma tienen carácter "
    "experimental e informativo. Ante situaciones de riesgo "
    "deben consultarse las comunicaciones oficiales de las "
    "autoridades y organismos competentes."
)


st.caption(
    f"Paraná · San Nicolás {APP_VERSION} | "
    "Pronóstico diario: 15 días | "
    "Pronóstico extendido: 30 días | "
    "Escenario histórico severo: 60 días"
)
