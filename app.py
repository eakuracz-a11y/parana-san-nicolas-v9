import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from datetime import date, timedelta

from src.ina import (
    observed,
    forecast_meta,
    STATIONS,
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
fecha_desde = fecha_hasta - timedelta(days=30)


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


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def preparar_datos(df):
    """
    Prepara los datos obtenidos desde src.ina.
    """

    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # --------------------------------------------------------
    # FECHA
    # --------------------------------------------------------

    if "datetime" in df.columns:

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
            utc=True,
        )

        # Convertir a hora Argentina
        try:

            df["datetime"] = (
                df["datetime"]
                .dt.tz_convert("America/Argentina/Buenos_Aires")
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # NIVEL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LIMPIEZA
    # --------------------------------------------------------

    columnas_necesarias = []

    if "datetime" in df.columns:
        columnas_necesarias.append("datetime")

    if "nivel" in df.columns:
        columnas_necesarias.append("nivel")

    if columnas_necesarias:

        df = df.dropna(
            subset=columnas_necesarias
        )

    if "datetime" in df.columns:

        df = (
            df
            .sort_values("datetime")
            .reset_index(drop=True)
        )

    return df


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

                # ------------------------------------------------
                # CONSULTA
                # ------------------------------------------------

                df_ina, error_ina = observed(
                    inicio,
                    fin,
                )

                # ------------------------------------------------
                # ELIMINAR DATOS ANTERIORES
                # ------------------------------------------------

                st.session_state.pop(
                    "datos_ina",
                    None,
                )

                # ------------------------------------------------
                # ERROR
                # ------------------------------------------------

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

                    # ============================================
                    # PREPARAR DATOS
                    # ============================================

                    df = preparar_datos(
                        df_ina
                    )

                    if df.empty:

                        st.warning(
                            "El INA devolvió observaciones, "
                            "pero no fue posible procesarlas."
                        )

                    elif "datetime" not in df.columns:

                        st.error(
                            "No se pudo identificar "
                            "la fecha de las observaciones."
                        )

                    elif "nivel" not in df.columns:

                        st.error(
                            "No se pudo identificar "
                            "el nivel hidrométrico."
                        )

                    else:

                        # ========================================
                        # GUARDAR EN SESIÓN
                        # ========================================

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
        "**Actualizar INA** para consultar los datos observados."
    )


# ============================================================
# MOSTRAR DATOS
# ============================================================

else:

    df = st.session_state[
        "datos_ina"
    ]

    # ========================================================
    # DATOS HIDROMÉTRICOS
    # ========================================================

    st.subheader(
        "📊 Datos hidrométricos observados"
    )

    valores = (
        df["nivel"]
        .dropna()
    )

    if len(valores) == 0:

        st.warning(
            "No existen valores hidrométricos válidos."
        )

    else:

        # ----------------------------------------------------
        # DATOS ACTUALES
        # ----------------------------------------------------

        nivel_actual = valores.iloc[-1]

        minimo = valores.min()
        maximo = valores.max()
        promedio = valores.mean()

        ultima_fecha = (
            df.loc[
                df["nivel"].notna(),
                "datetime",
            ]
            .iloc[-1]
        )

        # ----------------------------------------------------
        # MÉTRICAS PRINCIPALES
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Último nivel",
            f"{nivel_actual:.2f} m",
        )

        col2.metric(
            "Mínimo",
            f"{minimo:.2f} m",
        )

        col3.metric(
            "Máximo",
            f"{maximo:.2f} m",
        )

        col4.metric(
            "Promedio",
            f"{promedio:.2f} m",
        )

        # ----------------------------------------------------
        # INFORMACIÓN DEL ÚLTIMO DATO
        # ----------------------------------------------------

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
            f"Registros utilizados: **{len(df)}**"
        )

        # ====================================================
        # GRÁFICO
        # ====================================================

        st.subheader(
            "📈 Evolución del nivel del río"
        )

        fig = go.Figure()

        # ----------------------------------------------------
        # SERIE OBSERVADA
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["nivel"],
                mode="lines+markers",
                name="Nivel observado",
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Nivel: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

        # ----------------------------------------------------
        # ÚLTIMO VALOR
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=[
                    df["datetime"].iloc[-1]
                ],
                y=[
                    df["nivel"].iloc[-1]
                ],
                mode="markers",
                marker=dict(
                    size=11,
                ),
                name="Último dato",
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Último nivel: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Nivel hidrométrico (m)",
            hovermode="x unified",
            height=520,
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
                t=50,
                b=20,
            ),
        )

        fig.update_xaxes(
            tickformat="%d/%m/%Y"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ====================================================
        # RESUMEN DEL PERÍODO
        # ====================================================

        with st.expander(
            "📋 Resumen del período"
        ):

            periodo = st.session_state.get(
                "periodo_ina"
            )

            if periodo:

                fecha_inicio = periodo[
                    0
                ].strftime(
                    "%d/%m/%Y"
                )

                fecha_fin = periodo[
                    1
                ].strftime(
                    "%d/%m/%Y"
                )

                st.write(
                    f"**Período consultado:** "
                    f"{fecha_inicio} al {fecha_fin}"
                )

            st.write(
                f"**Cantidad de observaciones:** {len(df)}"
            )

            st.write(
                f"**Nivel mínimo:** {minimo:.2f} m"
            )

            st.write(
                f"**Nivel máximo:** {maximo:.2f} m"
            )

            st.write(
                f"**Nivel promedio:** {promedio:.2f} m"
            )

            st.write(
                f"**Último nivel:** {nivel_actual:.2f} m"
            )

        # ====================================================
        # TABLA
        # ====================================================

        with st.expander(
            "🗂️ Ver observaciones"
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
# PRONÓSTICO EXPERIMENTAL
# ============================================================

st.divider()

st.subheader(
    "🔮 Pronóstico experimental"
)

try:

    meta = forecast_meta()

    if isinstance(
        meta,
        dict,
    ):

        st.info(
            meta.get(
                "observacion",
                "Pronóstico experimental generado "
                "por el modelo propio.",
            )
        )

except Exception:

    st.info(
        "El módulo de pronóstico experimental "
        "se encuentra disponible para futuras versiones."
    )


# ============================================================
# ESTACIONES CONSIDERADAS
# ============================================================

with st.expander(
    "📍 Estaciones consideradas"
):

    try:

        for estacion in STATIONS:

            st.write(
                f"• {estacion}"
            )

    except Exception:

        st.write(
            "• San Nicolás de los Arroyos"
        )


# ============================================================
# FUENTE Y PIE
# ============================================================

st.divider()

st.caption(
    "Paraná · San Nicolás V9 | "
    "Datos observados: Instituto Nacional del Agua (INA) | "
    "Serie 36 · San Nicolás | "
    "Predicción: modelo experimental propio"
)
