import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from datetime import date, timedelta

from src.ina import (
    observed,
    forecast_meta,
)

from src.model import (
    train,
    predict,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# CONSTANTES
# ============================================================

FORECAST_DAYS = 15


# ============================================================
# ENCABEZADO
# ============================================================

st.title("🌊 PARANÁ · SAN NICOLÁS")

st.caption(
    "V9 · Plataforma pública de monitoreo y predicción experimental"
)

st.markdown(
    """
    Esta plataforma utiliza datos hidrométricos observados del
    **Instituto Nacional del Agua (INA)** y un modelo experimental
    propio para estimar la evolución del nivel del río Paraná
    en **San Nicolás de los Arroyos**.
    """
)


# ============================================================
# BARRA LATERAL
# ============================================================

st.sidebar.header("Consulta online")

fecha_hasta = date.today()

# Se utilizan 90 días por defecto para disponer
# de más información para entrenar el modelo.
fecha_desde = fecha_hasta - timedelta(days=90)


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
    "🔄 Actualizar INA",
    use_container_width=True,
)


st.sidebar.divider()

st.sidebar.subheader("Objetivo")
st.sidebar.write("San Nicolás de los Arroyos")

st.sidebar.subheader("Fuente")
st.sidebar.write("Instituto Nacional del Agua (INA)")

st.sidebar.subheader("Pronóstico")
st.sidebar.write("Horizonte experimental: 15 días")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def preparar_datos(df):
    """
    Prepara los datos provenientes de src.ina.
    """

    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ========================================================
    # FECHA
    # ========================================================

    if "datetime" not in df.columns:
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
        utc=True,
    )

    # Convertir a horario argentino para visualización
    try:

        df["datetime"] = (
            df["datetime"]
            .dt.tz_convert(
                "America/Argentina/Buenos_Aires"
            )
        )

    except Exception:
        pass

    # ========================================================
    # NIVEL
    # ========================================================

    if "value" in df.columns:

        df["nivel"] = pd.to_numeric(
            df["value"],
            errors="coerce",
        )

    elif "nivel" in df.columns:

        df["nivel"] = pd.to_numeric(
            df["nivel"],
            errors="coerce",
        )

    else:

        return pd.DataFrame()

    # ========================================================
    # LIMPIEZA
    # ========================================================

    df = df.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    df = (
        df
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# ENTRENAR Y PRONOSTICAR
# ============================================================

def generar_pronostico(df):
    """
    Entrena el modelo y genera 15 días futuros.
    """

    models, metrics = train(
        df
    )

    forecast = predict(
        df,
        models,
        days=FORECAST_DAYS,
    )

    return (
        forecast,
        models,
        metrics,
    )


# ============================================================
# VALIDACIÓN DE FECHAS
# ============================================================

if desde > hasta:

    st.sidebar.error(
        "La fecha Desde no puede ser posterior a la fecha Hasta."
    )


# ============================================================
# CONSULTA INA
# ============================================================

if actualizar:

    if desde > hasta:

        st.error(
            "El período seleccionado no es válido."
        )

    else:

        with st.spinner(
            "Consultando datos del INA..."
        ):

            try:

                inicio = desde.strftime(
                    "%Y-%m-%d"
                )

                fin = hasta.strftime(
                    "%Y-%m-%d"
                )

                # =================================================
                # CONSULTAR INA
                # =================================================

                df_ina, error_ina = observed(
                    inicio,
                    fin,
                )

                # Eliminar resultados anteriores
                st.session_state.pop(
                    "datos_ina",
                    None,
                )

                st.session_state.pop(
                    "pronostico",
                    None,
                )

                st.session_state.pop(
                    "metricas_modelo",
                    None,
                )

                # =================================================
                # VALIDACIONES
                # =================================================

                if error_ina:

                    st.warning(
                        error_ina
                    )

                elif df_ina is None:

                    st.warning(
                        "El INA no devolvió información."
                    )

                elif not isinstance(
                    df_ina,
                    pd.DataFrame,
                ):

                    st.error(
                        "La respuesta del INA no tiene "
                        "el formato esperado."
                    )

                elif df_ina.empty:

                    st.warning(
                        "El INA respondió correctamente, "
                        "pero no se encontraron observaciones "
                        "para el período seleccionado."
                    )

                else:

                    # =============================================
                    # PREPARAR OBSERVACIONES
                    # =============================================

                    df = preparar_datos(
                        df_ina
                    )

                    if df.empty:

                        st.warning(
                            "El INA devolvió observaciones, "
                            "pero no pudieron procesarse."
                        )

                    else:

                        st.session_state[
                            "datos_ina"
                        ] = df

                        st.session_state[
                            "periodo_ina"
                        ] = (
                            desde,
                            hasta,
                        )

                        st.success(
                            "✅ Datos del INA actualizados correctamente."
                        )

                        # =========================================
                        # ENTRENAR MODELO
                        # =========================================

                        with st.spinner(
                            "Generando pronóstico experimental "
                            "de 15 días..."
                        ):

                            try:

                                (
                                    forecast,
                                    models,
                                    metrics,
                                ) = generar_pronostico(
                                    df
                                )

                                st.session_state[
                                    "pronostico"
                                ] = forecast

                                st.session_state[
                                    "metricas_modelo"
                                ] = metrics

                                st.session_state[
                                    "modelo_info"
                                ] = models

                            except Exception as exc:

                                st.session_state[
                                    "error_modelo"
                                ] = str(
                                    exc
                                )

            except Exception as exc:

                st.error(
                    f"Error durante la consulta al INA: {exc}"
                )


# ============================================================
# ESTADO INICIAL
# ============================================================

if "datos_ina" not in st.session_state:

    st.info(
        "Seleccione un período y presione "
        "**Actualizar INA** para consultar los datos observados "
        "y generar el pronóstico experimental."
    )


# ============================================================
# DATOS OBSERVADOS
# ============================================================

else:

    df = st.session_state[
        "datos_ina"
    ]

    valores = (
        df["nivel"]
        .dropna()
    )

    if len(valores) == 0:

        st.warning(
            "No existen valores hidrométricos válidos."
        )

    else:

        # ====================================================
        # ÚLTIMA OBSERVACIÓN
        # ====================================================

        nivel_actual = float(
            valores.iloc[-1]
        )

        minimo = float(
            valores.min()
        )

        maximo = float(
            valores.max()
        )

        promedio = float(
            valores.mean()
        )

        ultima_fecha = (
            df.loc[
                df["nivel"].notna(),
                "datetime",
            ]
            .iloc[-1]
        )

        # ====================================================
        # MÉTRICAS OBSERVADAS
        # ====================================================

        st.subheader(
            "📊 Situación observada"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Último nivel",
            f"{nivel_actual:.2f} m",
        )

        c2.metric(
            "Mínimo período",
            f"{minimo:.2f} m",
        )

        c3.metric(
            "Máximo período",
            f"{maximo:.2f} m",
        )

        c4.metric(
            "Promedio",
            f"{promedio:.2f} m",
        )

        try:

            fecha_texto = ultima_fecha.strftime(
                "%d/%m/%Y %H:%M"
            )

        except Exception:

            fecha_texto = str(
                ultima_fecha
            )

        st.caption(
            f"Última observación disponible: "
            f"**{fecha_texto}** · "
            f"Registros observados: **{len(df)}**"
        )


        # ====================================================
        # OBTENER PRONÓSTICO
        # ====================================================

        forecast = st.session_state.get(
            "pronostico"
        )

        metrics = st.session_state.get(
            "metricas_modelo"
        )

        modelo_info = st.session_state.get(
            "modelo_info"
        )


        # ====================================================
        # GRÁFICO OBSERVADO + PRONÓSTICO
        # ====================================================

        st.subheader(
            "📈 Nivel observado y pronóstico a 15 días"
        )

        fig = go.Figure()


        # ----------------------------------------------------
        # OBSERVADO
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["nivel"],
                mode="lines+markers",
                name="Nivel observado",
                line=dict(
                    width=3,
                ),
                marker=dict(
                    size=5,
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Observado: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )


        # ====================================================
        # PRONÓSTICO
        # ====================================================

        if (
            forecast is not None
            and isinstance(
                forecast,
                pd.DataFrame,
            )
            and not forecast.empty
        ):

            forecast = forecast.copy()

            forecast[
                "datetime"
            ] = pd.to_datetime(
                forecast[
                    "datetime"
                ],
                errors="coerce",
            )


            # ------------------------------------------------
            # CONEXIÓN ÚLTIMO OBSERVADO → PRIMER PRONÓSTICO
            # ------------------------------------------------

            conexion_x = [
                df["datetime"].iloc[-1],
                forecast["datetime"].iloc[0],
            ]

            conexion_y = [
                df["nivel"].iloc[-1],
                forecast["prediction"].iloc[0],
            ]

            fig.add_trace(
                go.Scatter(
                    x=conexion_x,
                    y=conexion_y,
                    mode="lines",
                    line=dict(
                        dash="dash",
                        width=2,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )


            # ------------------------------------------------
            # BANDA SUPERIOR
            # ------------------------------------------------

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
                        width=0,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )


            # ------------------------------------------------
            # BANDA INFERIOR
            # ------------------------------------------------

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
                        width=0,
                    ),
                    fill="tonexty",
                    name="Intervalo experimental",
                    hoverinfo="skip",
                )
            )


            # ------------------------------------------------
            # LÍNEA DE PRONÓSTICO
            # ------------------------------------------------

            fig.add_trace(
                go.Scatter(
                    x=forecast[
                        "datetime"
                    ],
                    y=forecast[
                        "prediction"
                    ],
                    mode="lines+markers",
                    name="Pronóstico experimental",
                    line=dict(
                        dash="dash",
                        width=3,
                    ),
                    marker=dict(
                        size=6,
                    ),
                    hovertemplate=(
                        "%{x|%d/%m/%Y}"
                        "<br>"
                        "Pronóstico: %{y:.2f} m"
                        "<extra></extra>"
                    ),
                )
            )


        # ====================================================
        # DISEÑO DEL GRÁFICO
        # ====================================================

        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Nivel hidrométrico (m)",
            hovermode="x unified",
            height=570,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20,
            ),
        )

        fig.update_xaxes(
            range=[0, 7],
    dtick=0.5,
    title="Nivel hidrométrico (m)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # ====================================================
        # RESULTADOS DEL PRONÓSTICO
        # ====================================================

        if (
            forecast is not None
            and isinstance(
                forecast,
                pd.DataFrame,
            )
            and not forecast.empty
        ):

            st.subheader(
                "🔮 Pronóstico experimental · 15 días"
            )


            # ------------------------------------------------
            # RESUMEN
            # ------------------------------------------------

            primer_pronostico = float(
                forecast[
                    "prediction"
                ].iloc[0]
            )

            ultimo_pronostico = float(
                forecast[
                    "prediction"
                ].iloc[-1]
            )

            max_pronostico = float(
                forecast[
                    "prediction"
                ].max()
            )

            min_pronostico = float(
                forecast[
                    "prediction"
                ].min()
            )


            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Pronóstico día 1",
                f"{primer_pronostico:.2f} m",
            )

            p2.metric(
                "Pronóstico día 15",
                f"{ultimo_pronostico:.2f} m",
            )

            p3.metric(
                "Máximo previsto",
                f"{max_pronostico:.2f} m",
            )

            p4.metric(
                "Mínimo previsto",
                f"{min_pronostico:.2f} m",
            )


            # =================================================
            # TABLA PRONÓSTICO
            # =================================================

            tabla_forecast = (
                forecast.copy()
            )

            tabla_forecast[
                "Fecha"
            ] = tabla_forecast[
                "datetime"
            ].dt.strftime(
                "%d/%m/%Y"
            )

            tabla_forecast[
                "Pronóstico (m)"
            ] = tabla_forecast[
                "prediction"
            ].round(2)

            tabla_forecast[
                "Inferior (m)"
            ] = tabla_forecast[
                "lower"
            ].round(2)

            tabla_forecast[
                "Superior (m)"
            ] = tabla_forecast[
                "upper"
            ].round(2)

            tabla_forecast = (
                tabla_forecast[
                    [
                        "Fecha",
                        "Pronóstico (m)",
                        "Inferior (m)",
                        "Superior (m)",
                    ]
                ]
            )

            with st.expander(
                "📋 Ver pronóstico diario"
            ):

                st.dataframe(
                    tabla_forecast,
                    use_container_width=True,
                    hide_index=True,
                )


            # =================================================
            # INFORMACIÓN DEL MODELO
            # =================================================

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
                    "**Variable objetivo:** "
                    "nivel hidrométrico de San Nicolás"
                )

                if modelo_info:

                    st.write(
                        "**Observaciones utilizadas:**",
                        modelo_info.get(
                            "observations",
                            len(df),
                        ),
                    )

                    st.write(
                        "**Filas efectivas de entrenamiento:**",
                        modelo_info.get(
                            "training_rows",
                            "-",
                        ),
                    )

                if metrics:

                    rmse = metrics.get(
                        "RMSE"
                    )

                    if rmse is not None:

                        st.write(
                            "**RMSE de validación:** "
                            f"{float(rmse):.3f} m"
                        )

                st.caption(
                    "El pronóstico y su intervalo de incertidumbre "
                    "son experimentales. No constituyen un pronóstico "
                    "oficial del Instituto Nacional del Agua."
                )


        # ====================================================
        # ERROR DE MODELO
        # ====================================================

        elif "error_modelo" in st.session_state:

            st.warning(
                "Los datos observados fueron obtenidos "
                "correctamente, pero el modelo no pudo "
                "generar el pronóstico."
            )

            st.code(
                st.session_state[
                    "error_modelo"
                ]
            )


        # ====================================================
        # OBSERVACIONES
        # ====================================================

        with st.expander(
            "🗂️ Ver datos observados"
        ):

            tabla = df[
                [
                    "datetime",
                    "nivel",
                ]
            ].copy()

            tabla[
                "Fecha"
            ] = tabla[
                "datetime"
            ].dt.strftime(
                "%d/%m/%Y"
            )

            tabla[
                "Nivel (m)"
            ] = tabla[
                "nivel"
            ].round(2)

            tabla = tabla[
                [
                    "Fecha",
                    "Nivel (m)",
                ]
            ]

            st.dataframe(
                tabla,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# METODOLOGÍA
# ============================================================

st.divider()

with st.expander(
    "ℹ️ Metodología y alcance"
):

    try:

        meta = forecast_meta()

        st.write(
            "**Fuente de observaciones:** "
            f"{meta.get('fuente', 'INA')}"
        )

        st.write(
            "**Estación:** "
            f"{meta.get('estacion', 'San Nicolás')}"
        )

        st.write(
            "**Serie INA:** "
            f"{meta.get('seriesId', 36)}"
        )

        st.write(
            "**Variable:** "
            f"{meta.get('variable', 'Altura hidrométrica')}"
        )

        st.write(
            "**Unidad:** "
            f"{meta.get('unidad', 'm')}"
        )

    except Exception:

        st.write(
            "Fuente de datos observados: "
            "Instituto Nacional del Agua (INA)."
        )

    st.markdown(
        """
        El modelo utiliza la evolución reciente del nivel observado
        en San Nicolás para construir retardos temporales, diferencias,
        tendencias y promedios móviles.

        A partir de esas variables se entrena un modelo
        **Random Forest Regressor** y se genera recursivamente
        un pronóstico diario de **15 días**.

        La banda mostrada alrededor del pronóstico representa una
        estimación experimental de incertidumbre basada en el error
        de validación del modelo. La incertidumbre aumenta con el
        horizonte temporal.
        """
    )

    st.warning(
        "Este pronóstico es experimental y no reemplaza "
        "la información, alertas ni pronósticos oficiales "
        "emitidos por organismos competentes."
    )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "Paraná · San Nicolás V9 | "
    "Observaciones: Instituto Nacional del Agua (INA) | "
    "Serie 36 · San Nicolás | "
    "Pronóstico experimental: 15 días"
)
