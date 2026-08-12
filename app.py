import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

from src.ina import observed, forecast_meta, STATIONS


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 PARANÁ · SAN NICOLÁS")
st.caption("V9 · Plataforma pública de monitoreo y predicción experimental")


# ============================================================
# DESCRIPCIÓN
# ============================================================

st.markdown(
    """
    Esta plataforma utiliza datos hidrométricos observados del
    Instituto Nacional del Agua (INA) y un modelo experimental
    propio para estimar la evolución del nivel del río Paraná
    en San Nicolás.
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
    value=fecha_desde
)

hasta = st.sidebar.date_input(
    "Hasta",
    value=fecha_hasta
)

actualizar = st.sidebar.button(
    "🔄 Actualizar INA",
    use_container_width=True
)


st.sidebar.divider()

st.sidebar.subheader("Objetivo")
st.sidebar.write("San Nicolás de los Arroyos")

st.sidebar.subheader("Fuente")
st.sidebar.write("Instituto Nacional del Agua (INA)")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def encontrar_columna_fecha(df):
    """
    Busca automáticamente la columna temporal entregada por INA.
    """

    candidatos = [
        "datetime",
        "dateTime",
        "timestamp",
        "timeStamp",
        "fecha_hora",
        "fechaHora",
        "fecha",
        "date",
        "time",
        "observedAt",
        "observed_at",
        "timeStart",
        "time_start",
    ]

    columnas = list(df.columns)

    # Primero buscamos coincidencias exactas
    for candidato in candidatos:
        for columna in columnas:
            if str(columna).lower() == candidato.lower():
                return columna

    # Después buscamos coincidencias parciales
    for columna in columnas:
        nombre = str(columna).lower()

        if (
            "date" in nombre
            or "time" in nombre
            or "fecha" in nombre
            or "hora" in nombre
        ):
            return columna

    return None


def encontrar_columna_nivel(df):
    """
    Busca automáticamente la columna que contiene el nivel
    hidrométrico.
    """

    candidatos = [
        "value",
        "valor",
        "nivel",
        "level",
        "height",
        "altura",
        "dato",
        "observed",
        "observacion",
        "measurement",
        "measure",
    ]

    columnas = list(df.columns)

    for candidato in candidatos:
        for columna in columnas:
            if str(columna).lower() == candidato.lower():
                return columna

    # Búsqueda parcial
    for columna in columnas:
        nombre = str(columna).lower()

        if (
            "nivel" in nombre
            or "level" in nombre
            or "height" in nombre
            or "valor" in nombre
            or "value" in nombre
        ):
            return columna

    # Último recurso:
    # buscar una columna numérica
    for columna in columnas:
        try:
            valores = pd.to_numeric(df[columna], errors="coerce")

            if valores.notna().sum() > 3:
                return columna
        except Exception:
            pass

    return None


def preparar_datos(df):
    """
    Normaliza los datos recibidos desde INA.
    """

    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # --------------------------------------------------------
    # Buscar fecha
    # --------------------------------------------------------

    columna_fecha = encontrar_columna_fecha(df)

    if columna_fecha is None:

        # Algunos servicios pueden devolver la fecha como índice
        if isinstance(df.index, pd.DatetimeIndex):
            df["datetime"] = df.index
            columna_fecha = "datetime"

        else:
            return df

    else:
        df["datetime"] = pd.to_datetime(
            df[columna_fecha],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Buscar nivel
    # --------------------------------------------------------

    columna_nivel = encontrar_columna_nivel(df)

    if columna_nivel is not None:

        if columna_nivel != "nivel":
            df["nivel"] = pd.to_numeric(
                df[columna_nivel],
                errors="coerce"
            )

    # Ordenar
    if "datetime" in df.columns:

        df = df.sort_values("datetime")

        df = df.dropna(
            subset=["datetime"]
        )

    return df


# ============================================================
# CONSULTA INA
# ============================================================

if actualizar:

    with st.spinner("Consultando datos del INA..."):

        try:

            inicio = desde.strftime("%Y-%m-%d")
            fin = hasta.strftime("%Y-%m-%d")

            resultado = observed(
                inicio,
                fin
            )

            if resultado is None:
                st.error(
                    "El INA no devolvió información."
                )

            else:

                df = preparar_datos(resultado)

                if df.empty:

                    st.warning(
                        "El INA respondió, pero no se encontraron "
                        "datos para el período seleccionado."
                    )

                else:

                    st.success(
                        "✅ Datos del INA actualizados correctamente."
                    )

                    # Guardamos los datos
                    st.session_state["datos_ina"] = df

        except Exception as e:

            st.error(
                f"Error durante la consulta al INA: {e}"
            )


# ============================================================
# MOSTRAR DATOS
# ============================================================

if "datos_ina" not in st.session_state:

    st.info(
        "Presione **Actualizar INA** para iniciar la consulta online."
    )

else:

    df = st.session_state["datos_ina"]

    # --------------------------------------------------------
    # INFORMACIÓN GENERAL
    # --------------------------------------------------------

    st.subheader("📊 Datos hidrométricos")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Registros",
        len(df)
    )

    if "nivel" in df.columns:

        nivel_actual = df["nivel"].dropna()

        if len(nivel_actual) > 0:

            col2.metric(
                "Último nivel",
                f"{nivel_actual.iloc[-1]:.2f}"
            )

            col3.metric(
                "Máximo período",
                f"{nivel_actual.max():.2f}"
            )

    # --------------------------------------------------------
    # GRÁFICO
    # --------------------------------------------------------

    if (
        "datetime" in df.columns
        and "nivel" in df.columns
        and df["nivel"].notna().any()
    ):

        st.subheader("📈 Evolución del nivel del río")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["nivel"],
                mode="lines",
                name="Nivel observado"
            )
        )

        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Nivel",
            hovermode="x unified",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # ESTADÍSTICAS
        # ----------------------------------------------------

        st.subheader("📋 Estadísticas")

        valores = df["nivel"].dropna()

        if len(valores) > 0:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Mínimo",
                f"{valores.min():.2f}"
            )

            c2.metric(
                "Máximo",
                f"{valores.max():.2f}"
            )

            c3.metric(
                "Promedio",
                f"{valores.mean():.2f}"
            )

            c4.metric(
                "Último",
                f"{valores.iloc[-1]:.2f}"
            )

    else:

        st.warning(
            "Los datos recibidos del INA no contienen "
            "una columna temporal o de nivel reconocible."
        )

        st.write(
            "Columnas recibidas desde INA:"
        )

        st.code(
            ", ".join(str(c) for c in df.columns)
        )

        st.dataframe(
            df,
            use_container_width=True
        )


# ============================================================
# INFORMACIÓN DEL PRONÓSTICO
# ============================================================

st.divider()

st.subheader("🔮 Pronóstico experimental")

try:

    meta = forecast_meta()

    if isinstance(meta, dict):

        st.info(
            meta.get(
                "observacion",
                "Pronóstico experimental generado por el modelo propio."
            )
        )

except Exception:

    st.info(
        "El módulo de pronóstico experimental está disponible "
        "para futuras versiones."
    )


# ============================================================
# ESTACIONES
# ============================================================

with st.expander("📍 Estaciones consideradas"):

    try:

        for estacion in STATIONS:
            st.write(f"• {estacion}")

    except Exception:

        st.write(
            "San Nicolás de los Arroyos"
        )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "Paraná · San Nicolás V9 | Datos observados: INA | "
    "Predicción: modelo experimental propio"
)
