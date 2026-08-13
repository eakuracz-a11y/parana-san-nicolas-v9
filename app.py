import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from datetime import date, timedelta

from src.ina import observed

from src.model import (
    train,
    predict,
)

from src.exogenous import (
    get_exogenous_data,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
)


FORECAST_DAYS = 15

Y_MIN = 0.0
Y_MAX = 7.0
Y_STEP = 0.5


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "🌊 PARANÁ · SAN NICOLÁS"
)

st.caption(
    "V9 · Monitoreo hidrométrico y pronóstico experimental"
)

st.markdown(
    """
    Datos observados del **Instituto Nacional del Agua (INA)**.
    El pronóstico experimental incorpora evolución del nivel,
    precipitación prevista y, cuando está disponible,
    información de caudal del río Paraná.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Consulta online"
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
)


st.sidebar.divider()

st.sidebar.subheader(
    "Objetivo"
)

st.sidebar.write(
    "San Nicolás de los Arroyos"
)

st.sidebar.subheader(
    "Horizonte"
)

st.sidebar.write(
    "15 días"
)

st.sidebar.subheader(
    "Escala hidrométrica"
)

st.sidebar.write(
    "0–7 m"
)


# ============================================================
# PREPARAR DATOS INA
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

    df = df.copy()

    if "datetime" not in df.columns:

        return pd.DataFrame()

    df[
        "datetime"
    ] = pd.to_datetime(
        df[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    if "value" in df.columns:

        df[
            "nivel"
        ] = pd.to_numeric(
            df[
                "value"
            ],
            errors="coerce",
        )

    elif "nivel" in df.columns:

        df[
            "nivel"
        ] = pd.to_numeric(
            df[
                "nivel"
            ],
            errors="coerce",
        )

    else:

        return pd.DataFrame()

    df = df.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    df = (
        df
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# ACTUALIZACIÓN
# ============================================================

if actualizar:

    if desde > hasta:

        st.error(
            "La fecha Desde no puede "
            "ser posterior a Hasta."
        )

    else:

        inicio = desde.strftime(
            "%Y-%m-%d"
        )

        fin = hasta.strftime(
            "%Y-%m-%d"
        )

        # ====================================================
        # INA
        # ====================================================

        with st.spinner(
            "Consultando nivel del INA..."
        ):

            df_ina, error = observed(
                inicio,
                fin,
            )

        if error:

            st.error(
                error
            )

        else:

            df = preparar_datos(
                df_ina
            )

            if df.empty:

                st.error(
                    "No se obtuvieron datos "
                    "hidrométricos válidos."
                )

            else:

                # ============================================
                # VARIABLES EXTERNAS
                # ============================================

                with st.spinner(
                    "Consultando lluvia y caudal..."
                ):

                    (
                        exog_history,
                        exog_future,
                        exog_meta,
                    ) = get_exogenous_data(
                        inicio,
                        fin,
                        FORECAST_DAYS,
                    )

                # ============================================
                # MODELO
                # ============================================

                with st.spinner(
                    "Entrenando modelo "
                    "y generando 15 días..."
                ):

                    try:

                        models, metrics = train(
                            df,
                            exog_history=exog_history,
                        )

                        forecast = predict(
                            df,
                            models,
                            days=FORECAST_DAYS,
                            exog_future=exog_future,
                        )

                    except Exception as exc:

                        st.error(
                            f"Error del modelo: {exc}"
                        )

                        forecast = pd.DataFrame()

                        models = {}

                        metrics = {}

                # ============================================
                # GUARDAR
                # ============================================

                st.session_state[
                    "datos"
                ] = df

                st.session_state[
                    "forecast"
                ] = forecast

                st.session_state[
                    "metrics"
                ] = metrics

                st.session_state[
                    "models"
                ] = models

                st.session_state[
                    "exog_history"
                ] = exog_history

                st.session_state[
                    "exog_future"
                ] = exog_future

                st.session_state[
                    "exog_meta"
                ] = exog_meta

                st.success(
                    "✅ Datos y pronóstico "
                    "actualizados correctamente."
                )


# ============================================================
# ESTADO INICIAL
# ============================================================

if "datos" not in st.session_state:

    st.info(
        "Presione **Actualizar modelo** "
        "para consultar INA, lluvia, caudal "
        "y generar el pronóstico."
    )


# ============================================================
# RESULTADOS
# ============================================================

else:

    df = st.session_state[
        "datos"
    ]

    forecast = st.session_state.get(
        "forecast",
        pd.DataFrame(),
    )

    metrics = st.session_state.get(
        "metrics",
        {},
    )

    models = st.session_state.get(
        "models",
        {},
    )

    exog_future = st.session_state.get(
        "exog_future",
        pd.DataFrame(),
    )

    exog_history = st.session_state.get(
        "exog_history",
        pd.DataFrame(),
    )

    exog_meta = st.session_state.get(
        "exog_meta",
        {},
    )


    # ========================================================
    # SITUACIÓN
    # ========================================================

    nivel_actual = float(
        df[
            "nivel"
        ].iloc[-1]
    )


    st.subheader(
        "📊 Situación observada"
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "Último nivel",
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
        "Promedio",
        f"{df['nivel'].mean():.2f} m",
    )


    # ========================================================
    # GRÁFICO PRINCIPAL
    # ========================================================

    st.subheader(
        "📈 Nivel observado y pronóstico a 15 días"
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=df[
                "datetime"
            ],
            y=df[
                "nivel"
            ],
            mode="lines+markers",
            name="Nivel observado",
            line=dict(
                width=3,
            ),
            marker=dict(
                size=4,
            ),
        )
    )


    if not forecast.empty:

        f = forecast.copy()

        f[
            "datetime"
        ] = pd.to_datetime(
            f[
                "datetime"
            ],
            errors="coerce",
        )


        # Banda superior

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
                    width=0,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )


        # Banda inferior

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
                    width=0,
                ),
                fill="tonexty",
                name="Intervalo experimental",
            )
        )


        # Predicción

        fig.add_trace(
            go.Scatter(
                x=f[
                    "datetime"
                ],
                y=f[
                    "prediction"
                ],
                mode="lines+markers",
                name=(
                    "Pronóstico · nivel + lluvia + caudal"
                ),
                line=dict(
                    dash="dash",
                    width=3,
                ),
                marker=dict(
                    size=7,
                ),
            )
        )


    fig.update_layout(
        height=620,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.03,
        ),
    )


    fig.update_xaxes(
        title_text="Fecha",
        type="date",
        tickformat="%d/%m/%Y",
    )


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
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # LLUVIA PREVISTA
    # ========================================================

    st.subheader(
        "🌧️ Precipitación prevista · 15 días"
    )


    if (
        exog_future is not None
        and not exog_future.empty
        and "precip_mm"
        in exog_future.columns
    ):

        rain = exog_future.copy()

        total_rain = float(
            rain[
                "precip_mm"
            ].fillna(
                0
            ).sum()
        )

        max_rain = float(
            rain[
                "precip_mm"
            ].fillna(
                0
            ).max()
        )

        wet_days = int(
            (
                rain[
                    "precip_mm"
                ].fillna(
                    0
                )
                >= 1.0
            ).sum()
        )


        r1, r2, r3 = st.columns(
            3
        )


        r1.metric(
            "Acumulado previsto",
            f"{total_rain:.1f} mm",
        )


        r2.metric(
            "Máximo diario",
            f"{max_rain:.1f} mm",
        )


        r3.metric(
            "Días ≥ 1 mm",
            wet_days,
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
                name=(
                    "Precipitación media "
                    "del corredor"
                ),
            )
        )


        rain_fig.update_layout(
            height=350,
            xaxis_title="Fecha",
            yaxis_title="Precipitación (mm/día)",
        )


        rain_fig.update_xaxes(
            tickformat="%d/%m"
        )


        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )


        st.caption(
            "La precipitación representa el promedio "
            "de puntos desde Corrientes hasta San Nicolás, "
            "no un balance hidrológico completo de la cuenca."
        )

    else:

        st.info(
            "No se obtuvo pronóstico de precipitación."
        )


    # ========================================================
    # CAUDAL
    # ========================================================

    st.subheader(
        "💧 Caudal del Paraná"
    )


    if (
        exog_history is not None
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


        q_actual = float(
            q_hist[
                "caudal_m3s"
            ].iloc[-1]
        )


        if len(
            q_hist
        ) >= 7:

            q_7 = float(
                q_hist[
                    "caudal_m3s"
                ].iloc[-7]
            )

            delta_q = (
                q_actual
                - q_7
            )

        else:

            delta_q = 0.0


        q_future = exog_future[
            "caudal_m3s"
        ].dropna()


        if len(
            q_future
        ):

            q_15 = float(
                q_future.iloc[-1]
            )

        else:

            q_15 = q_actual


        q1, q2, q3 = st.columns(
            3
        )


        q1.metric(
            "Último caudal",
            f"{q_actual:,.0f} m³/s",
        )


        q2.metric(
            "Variación reciente",
            f"{delta_q:+,.0f} m³/s",
        )


        q3.metric(
            "Proyección día 15",
            f"{q_15:,.0f} m³/s",
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
            )
        )


        if (
            "caudal_m3s"
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
                    line=dict(
                        dash="dash"
                    ),
                    name=(
                        "Proyección experimental"
                    ),
                )
            )


        q_fig.update_layout(
            height=380,
            xaxis_title="Fecha",
            yaxis_title="Caudal (m³/s)",
        )


        q_fig.update_xaxes(
            tickformat="%d/%m"
        )


        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )


        info_q = exog_meta.get(
            "caudal_series"
        )


        if info_q:

            st.caption(
                "Serie de caudal utilizada: "
                f"{info_q.get('station')} · "
                f"seriesId {info_q.get('series_id')} · "
                f"{info_q.get('proc_name')}."
            )


        st.warning(
            "La evolución futura del caudal es una "
            "proyección experimental de la tendencia reciente; "
            "no es un pronóstico oficial del INA."
        )

    else:

        st.info(
            "No se encontró una serie reciente "
            "de caudal adecuada para incorporar al modelo."
        )


    # ========================================================
    # TABLA 15 DÍAS
    # ========================================================

    if not forecast.empty:

        st.subheader(
            "🔮 Detalle del pronóstico"
        )


        table = forecast.copy()


        table[
            "Fecha"
        ] = pd.to_datetime(
            table[
                "datetime"
            ]
        ).dt.strftime(
            "%d/%m/%Y"
        )


        table[
            "Nivel (m)"
        ] = table[
            "prediction"
        ].round(
            2
        )


        table[
            "Lluvia (mm)"
        ] = table[
            "precip_mm"
        ].round(
            1
        )


        table[
            "Caudal (m³/s)"
        ] = table[
            "caudal_m3s"
        ].round(
            0
        )


        st.dataframe(
            table[
                [
                    "Fecha",
                    "Nivel (m)",
                    "Lluvia (mm)",
                    "Caudal (m³/s)",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # MODELO
    # ========================================================

    with st.expander(
        "🧠 Información del modelo"
    ):

        st.write(
            "**Modelo:** Random Forest Regressor"
        )

        st.write(
            "**Horizonte:** 15 días"
        )

        st.write(
            "**Precipitación prevista:** incluida"
        )

        st.write(
            "**Caudal:** incluido cuando existe "
            "una serie INA reciente utilizable"
        )


        if metrics.get(
            "RMSE"
        ) is not None:

            st.write(
                "**RMSE:** "
                f"{metrics['RMSE']:.3f} m"
            )


        st.warning(
            "El resultado es experimental. "
            "No reemplaza los pronósticos ni alertas "
            "emitidos por organismos oficiales."
        )


# ============================================================
# PIE
# ============================================================

st.divider()


st.caption(
    "Paraná · San Nicolás V9 | "
    "Nivel y caudal: INA | "
    "Precipitación: Open-Meteo | "
    "Pronóstico experimental: 15 días"
)
