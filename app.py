import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import date, timedelta

from src.ina import (
    observed,
    forecast_meta,
    STATIONS,
    get_last_diagnostic,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# TÍTULO
# ============================================================

st.title("🌊 PARANÁ · SAN NICOLÁS")

st.caption(
    "V9 · Plataforma pública de monitoreo y predicción experimental"
)


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

    # Coincidencias exactas
    for candidato in candidatos:

        for columna in columnas:

            if str(columna).lower() == candidato.lower():
                return columna

    # Coincidencias parciales
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
    Busca automáticamente la columna que contiene
    el nivel hidrométrico.
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

    # Coincidencias exactas
    for candidato in candidatos:

        for columna in columnas:

            if str(columna).lower() == candidato.lower():
                return columna

    # Coincidencias parciales
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

            valores = pd.to_numeric(
                df[columna],
                errors="coerce",
            )

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

    # ========================================================
    # FECHA
    # ========================================================

    columna_fecha = encontrar_columna_fecha(df)

    if columna_fecha is None:

        if isinstance(df.index, pd.DatetimeIndex):

            df["datetime"] = df.index
            columna_fecha = "datetime"

        else:

            return df

    if columna_fecha != "datetime":

        df["datetime"] = pd.to_datetime(
            df[columna_fecha],
            errors="coerce",
        )

    else:

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )

    # ========================================================
    # NIVEL
    # ========================================================

    columna_nivel = encontrar_columna_nivel(df)

    if columna_nivel is not None:

        df["nivel"] = pd.to_numeric(
            df[columna_nivel],
            errors="coerce",
        )

    # ========================================================
    # ORDENAR
    # ========================================================

    if "datetime" in df.columns:

        df = df.dropna(
            subset=["datetime"]
        )

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
            "El período seleccionado no es válido. "
            "La fecha Desde debe ser anterior o igual a Hasta."
        )

    else:

        with st.spinner("Consultando datos del INA..."):

            try:

                inicio = desde.strftime("%Y-%m-%d")
                fin = hasta.strftime("%Y-%m-%d")

                # ====================================================
                # CONSULTA AL MÓDULO INA
                # ====================================================

                df_ina, error_ina = observed(
                    inicio,
                    fin,
                )

                # ====================================================
                # OBTENER DIAGNÓSTICO
                # ====================================================

                diagnostico_ina = get_last_diagnostic()

                # ----------------------------------------------------
                # Borrar datos anteriores
                # ----------------------------------------------------

                st.session_state.pop(
                    "datos_ina",
                    None,
                )

                # ====================================================
                # MOSTRAR RESULTADO DE LA CONSULTA
                # ====================================================

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

                    st.write(
                        "Tipo recibido:",
                        type(df_ina).__name__,
                    )

                elif df_ina.empty:

                    st.warning(
                        "El INA respondió, pero no se encontraron "
                        "datos para el período seleccionado."
                    )

                else:

                    df = preparar_datos(
                        df_ina
                    )

                    if df.empty:

                        st.warning(
                            "El INA devolvió registros, pero "
                            "no fue posible procesarlos."
                        )

                    elif "datetime" not in df.columns:

                        st.warning(
                            "El INA devolvió datos, pero no "
                            "se pudo identificar la columna de fecha."
                        )

                        st.write(
                            "Columnas recibidas desde INA:"
                        )

                        st.code(
                            ", ".join(
                                str(c)
                                for c in df.columns
                            )
                        )

                        st.dataframe(
                            df.head(20),
                            use_container_width=True,
                        )

                    elif "nivel" not in df.columns:

                        st.warning(
                            "El INA devolvió datos, pero no "
                            "se pudo identificar la columna "
                            "de nivel hidrométrico."
                        )

                        st.write(
                            "Columnas recibidas desde INA:"
                        )

                        st.code(
                            ", ".join(
                                str(c)
                                for c in df.columns
                            )
                        )

                        st.dataframe(
                            df.head(20),
                            use_container_width=True,
                        )

                    else:

                        df["datetime"] = pd.to_datetime(
                            df["datetime"],
                            errors="coerce",
                        )

                        df["nivel"] = pd.to_numeric(
                            df["nivel"],
                            errors="coerce",
                        )

                        df = df.dropna(
                            subset=[
                                "datetime",
                                "nivel",
                            ]
                        )

                        df = (
                            df
                            .sort_values("datetime")
                            .reset_index(drop=True)
                        )

                        if df.empty:

                            st.warning(
                                "El INA devolvió observaciones, "
                                "pero no quedaron valores válidos."
                            )

                        else:

                            st.session_state[
                                "datos_ina"
                            ] = df

                            st.success(
                                "✅ Datos del INA actualizados "
                                "correctamente."
                            )

                            st.caption(
                                f"Período consultado: "
                                f"{desde.strftime('%d/%m/%Y')} → "
                                f"{hasta.strftime('%d/%m/%Y')} | "
                                f"Registros: {len(df)}"
                            )

                # ====================================================
                # DIAGNÓSTICO TÉCNICO
                # ====================================================

                with st.expander(
                    "🔧 Diagnóstico técnico INA",
                    expanded=True,
                ):

                    if diagnostico_ina:

                        # --------------------------------------------
                        # CONFIGURACIÓN
                        # --------------------------------------------

                        st.write(
                            "### Configuración"
                        )

                        configuracion = diagnostico_ina.get(
                            "configuracion",
                            {},
                        )

                        if configuracion:

                            st.json(
                                configuracion
                            )

                        else:

                            st.info(
                                "No se registró configuración."
                            )

                        # --------------------------------------------
                        # CONSULTA DE SERIES
                        # --------------------------------------------

                        st.write(
                            "### Consulta de series"
                        )

                        consulta_series = diagnostico_ina.get(
                            "consulta_series",
                            {},
                        )

                        if consulta_series:

                            st.write(
                                "HTTP:",
                                consulta_series.get(
                                    "status_series",
                                    "Sin información",
                                ),
                            )

                            st.write(
                                "Cantidad de series:",
                                consulta_series.get(
                                    "cantidad_series",
                                    0,
                                ),
                            )

                            st.write(
                                "URL consultada:"
                            )

                            st.code(
                                consulta_series.get(
                                    "url_series",
                                    "",
                                )
                            )

                            st.write(
                                "Respuesta del INA:"
                            )

                            texto_series = consulta_series.get(
                                "texto_series",
                                "",
                            )

                            st.code(
                                texto_series[:3000]
                            )

                        else:

                            st.warning(
                                "No se obtuvo información "
                                "de la consulta de series."
                            )

                        # --------------------------------------------
                        # SERIES DETECTADAS
                        # --------------------------------------------

                        st.write(
                            "### Series detectadas"
                        )

                        series_detectadas = diagnostico_ina.get(
                            "series_detectadas",
                            [],
                        )

                        if series_detectadas:

                            for serie in series_detectadas:

                                st.json(
                                    serie
                                )

                        else:

                            st.warning(
                                "No se detectaron series."
                            )

                        # --------------------------------------------
                        # INTENTOS POR SERIES ID
                        # --------------------------------------------

                        intentos = diagnostico_ina.get(
                            "intentos_datos",
                            [],
                        )

                        if intentos:

                            st.write(
                                "### Consultas por seriesId"
                            )

                            for intento in intentos:

                                st.json(
                                    intento
                                )

                        # --------------------------------------------
                        # CONSULTA DIRECTA
                        # --------------------------------------------

                        st.write(
                            "### Consulta directa siteCode + varId"
                        )

                        directo = diagnostico_ina.get(
                            "consulta_directa",
                            {},
                        )

                        if directo:

                            st.write(
                                "HTTP:",
                                directo.get(
                                    "status_datos",
                                    "Sin información",
                                ),
                            )

                            st.write(
                                "Cantidad de registros:",
                                directo.get(
                                    "cantidad_registros",
                                    0,
                                ),
                            )

                            st.write(
                                "URL consultada:"
                            )

                            st.code(
                                directo.get(
                                    "url_datos",
                                    "",
                                )
                            )

                            st.write(
                                "Respuesta original:"
                            )

                            texto_datos = directo.get(
                                "texto_datos",
                                "",
                            )

                            st.code(
                                texto_datos[:3000]
                            )

                        else:

                            st.warning(
                                "No se obtuvo diagnóstico "
                                "de la consulta directa."
                            )

                        # --------------------------------------------
                        # ERRORES
                        # --------------------------------------------

                        if "error_series" in diagnostico_ina:

                            st.error(
                                "Error buscando series: "
                                + str(
                                    diagnostico_ina[
                                        "error_series"
                                    ]
                                )
                            )

                        if (
                            "error_consulta_directa"
                            in diagnostico_ina
                        ):

                            st.error(
                                "Error en consulta directa: "
                                + str(
                                    diagnostico_ina[
                                        "error_consulta_directa"
                                    ]
                                )
                            )

                        if "error_general" in diagnostico_ina:

                            st.error(
                                "Error general: "
                                + str(
                                    diagnostico_ina[
                                        "error_general"
                                    ]
                                )
                            )

                    else:

                        st.warning(
                            "No se generó información "
                            "de diagnóstico."
                        )

            except Exception as e:

                st.error(
                    f"Error durante la consulta al INA: {e}"
                )


# ============================================================
# MOSTRAR DATOS
# ============================================================

if "datos_ina" not in st.session_state:

    st.info(
        "Presione **Actualizar INA** "
        "para iniciar la consulta online."
    )

else:

    df = st.session_state[
        "datos_ina"
    ]

    # ========================================================
    # INFORMACIÓN GENERAL
    # ========================================================

    st.subheader(
        "📊 Datos hidrométricos"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Registros",
        len(df),
    )

    if "nivel" in df.columns:

        nivel_actual = (
            df["nivel"]
            .dropna()
        )

        if len(nivel_actual) > 0:

            col2.metric(
                "Último nivel",
                f"{nivel_actual.iloc[-1]:.2f} m",
            )

            col3.metric(
                "Máximo período",
                f"{nivel_actual.max():.2f} m",
            )

    # ========================================================
    # GRÁFICO
    # ========================================================

    if (
        "datetime" in df.columns
        and "nivel" in df.columns
        and df["nivel"].notna().any()
    ):

        st.subheader(
            "📈 Evolución del nivel del río"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["nivel"],
                mode="lines",
                name="Nivel observado",
            )
        )

        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Nivel (m)",
            hovermode="x unified",
            height=500,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ====================================================
        # ESTADÍSTICAS
        # ====================================================

        st.subheader(
            "📋 Estadísticas"
        )

        valores = (
            df["nivel"]
            .dropna()
        )

        if len(valores) > 0:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Mínimo",
                f"{valores.min():.2f} m",
            )

            c2.metric(
                "Máximo",
                f"{valores.max():.2f} m",
            )

            c3.metric(
                "Promedio",
                f"{valores.mean():.2f} m",
            )

            c4.metric(
                "Último",
                f"{valores.iloc[-1]:.2f} m",
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
            ", ".join(
                str(c)
                for c in df.columns
            )
        )

        st.dataframe(
            df,
            use_container_width=True,
        )


# ============================================================
# INFORMACIÓN DEL PRONÓSTICO
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
        "está disponible para futuras versiones."
    )


# ============================================================
# ESTACIONES
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
            "San Nicolás de los Arroyos"
        )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "Paraná · San Nicolás V9 | "
    "Datos observados: INA | "
    "Predicción: modelo experimental propio"
)
