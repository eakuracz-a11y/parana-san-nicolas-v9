import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from datetime import date, timedelta
import json

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

    for candidato in candidatos:

        for columna in columnas:

            if str(columna).lower() == candidato.lower():
                return columna

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

    columna_nivel = encontrar_columna_nivel(df)

    if columna_nivel is not None:

        df["nivel"] = pd.to_numeric(
            df[columna_nivel],
            errors="coerce",
        )

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
# FUNCIÓN PARA BUSCAR SAN NICOLÁS EN SERIES
# ============================================================

def buscar_series_san_nicolas(series_detectadas):

    coincidencias = []

    for item in series_detectadas:

        registro = item.get(
            "registro",
            {}
        )

        texto = json.dumps(
            registro,
            ensure_ascii=False,
        ).lower()

        coincide_nombre = (
            "san nicolás" in texto
            or "san nicolas" in texto
        )

        coincide_site = False

        for clave in [
            "siteCode",
            "site_code",
            "sitecode",
            "codigoSitio",
            "codigo_sitio",
        ]:

            if clave in registro:

                valor = registro.get(
                    clave
                )

                try:

                    if int(valor) == 36:
                        coincide_site = True

                except Exception:
                    pass

        if coincide_nombre or coincide_site:

            coincidencias.append(
                item
            )

    return coincidencias


# ============================================================
# GENERAR TEXTO COMPACTO DE DIAGNÓSTICO
# ============================================================

def generar_texto_diagnostico(
    diagnostico
):

    lineas = []

    lineas.append(
        "=== DIAGNOSTICO INA ==="
    )

    lineas.append("")

    # --------------------------------------------------------
    # CONFIGURACIÓN
    # --------------------------------------------------------

    configuracion = diagnostico.get(
        "configuracion",
        {},
    )

    lineas.append(
        "CONFIGURACION:"
    )

    lineas.append(
        json.dumps(
            configuracion,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    lineas.append("")

    # --------------------------------------------------------
    # CONSULTA DE SERIES
    # --------------------------------------------------------

    consulta_series = diagnostico.get(
        "consulta_series",
        {},
    )

    lineas.append(
        "CONSULTA DE SERIES:"
    )

    lineas.append(
        f"HTTP: {consulta_series.get('status_series')}"
    )

    lineas.append(
        f"CANTIDAD TOTAL: "
        f"{consulta_series.get('cantidad_series', 0)}"
    )

    lineas.append(
        f"URL: "
        f"{consulta_series.get('url_series', '')}"
    )

    lineas.append("")

    # --------------------------------------------------------
    # SERIES DETECTADAS
    # --------------------------------------------------------

    series_detectadas = diagnostico.get(
        "series_detectadas",
        [],
    )

    lineas.append(
        "PRIMERAS 5 SERIES DETECTADAS:"
    )

    for i, serie in enumerate(
        series_detectadas[:5],
        start=1,
    ):

        lineas.append(
            f"\n--- SERIE {i} ---"
        )

        lineas.append(
            json.dumps(
                serie,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    lineas.append("")

    # --------------------------------------------------------
    # SAN NICOLÁS
    # --------------------------------------------------------

    coincidencias = buscar_series_san_nicolas(
        series_detectadas
    )

    lineas.append(
        "SERIES QUE COINCIDEN CON SAN NICOLAS / SITECODE 36:"
    )

    lineas.append(
        f"CANTIDAD: {len(coincidencias)}"
    )

    for i, serie in enumerate(
        coincidencias[:20],
        start=1,
    ):

        lineas.append(
            f"\n--- COINCIDENCIA {i} ---"
        )

        lineas.append(
            json.dumps(
                serie,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    lineas.append("")

    # --------------------------------------------------------
    # INTENTOS POR SERIES ID
    # --------------------------------------------------------

    intentos = diagnostico.get(
        "intentos_datos",
        [],
    )

    lineas.append(
        "INTENTOS POR SERIES ID:"
    )

    lineas.append(
        f"CANTIDAD DE INTENTOS: {len(intentos)}"
    )

    for i, intento in enumerate(
        intentos[:10],
        start=1,
    ):

        lineas.append(
            f"\n--- INTENTO {i} ---"
        )

        lineas.append(
            json.dumps(
                intento,
                ensure_ascii=False,
                indent=2,
                default=str,
            )[:3000]
        )

    lineas.append("")

    # --------------------------------------------------------
    # CONSULTA DIRECTA
    # --------------------------------------------------------

    directo = diagnostico.get(
        "consulta_directa",
        {},
    )

    lineas.append(
        "CONSULTA DIRECTA SITECODE + VARID:"
    )

    if directo:

        lineas.append(
            f"HTTP: "
            f"{directo.get('status_datos')}"
        )

        lineas.append(
            f"REGISTROS: "
            f"{directo.get('cantidad_registros', 0)}"
        )

        lineas.append(
            f"URL: "
            f"{directo.get('url_datos', '')}"
        )

        lineas.append(
            "RESPUESTA ORIGINAL:"
        )

        lineas.append(
            str(
                directo.get(
                    "texto_datos",
                    "",
                )
            )[:5000]
        )

    else:

        lineas.append(
            "Sin información."
        )

    # --------------------------------------------------------
    # ERRORES
    # --------------------------------------------------------

    if "error_series" in diagnostico:

        lineas.append("")

        lineas.append(
            "ERROR SERIES:"
        )

        lineas.append(
            str(
                diagnostico[
                    "error_series"
                ]
            )
        )

    if (
        "error_consulta_directa"
        in diagnostico
    ):

        lineas.append("")

        lineas.append(
            "ERROR CONSULTA DIRECTA:"
        )

        lineas.append(
            str(
                diagnostico[
                    "error_consulta_directa"
                ]
            )
        )

    if "error_general" in diagnostico:

        lineas.append("")

        lineas.append(
            "ERROR GENERAL:"
        )

        lineas.append(
            str(
                diagnostico[
                    "error_general"
                ]
            )
        )

    return "\n".join(
        lineas
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

                # ------------------------------------------------
                # CONSULTAR INA
                # ------------------------------------------------

                df_ina, error_ina = observed(
                    inicio,
                    fin,
                )

                diagnostico_ina = (
                    get_last_diagnostic()
                )

                # ------------------------------------------------
                # Borrar consulta anterior
                # ------------------------------------------------

                st.session_state.pop(
                    "datos_ina",
                    None,
                )

                # ------------------------------------------------
                # RESULTADO
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
                        "La respuesta del INA no "
                        "tiene el formato esperado."
                    )

                elif df_ina.empty:

                    st.warning(
                        "El INA respondió, pero no "
                        "se encontraron datos."
                    )

                else:

                    df = preparar_datos(
                        df_ina
                    )

                    if (
                        not df.empty
                        and "datetime" in df.columns
                        and "nivel" in df.columns
                    ):

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

                        if not df.empty:

                            st.session_state[
                                "datos_ina"
                            ] = df

                            st.success(
                                "✅ Datos del INA "
                                "actualizados correctamente."
                            )

                # =================================================
                # DIAGNÓSTICO COMPACTO
                # =================================================

                with st.expander(
                    "🔧 Diagnóstico técnico INA",
                    expanded=True,
                ):

                    if diagnostico_ina:

                        texto_diagnostico = (
                            generar_texto_diagnostico(
                                diagnostico_ina
                            )
                        )

                        st.write(
                            "### Diagnóstico para copiar"
                        )

                        st.caption(
                            "Haz clic dentro del cuadro, "
                            "Ctrl+A, Ctrl+C y pégalo en ChatGPT."
                        )

                        st.text_area(
                            "Contenido",
                            value=texto_diagnostico,
                            height=650,
                            key="diagnostico_copiable",
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

    st.subheader(
        "📊 Datos hidrométricos"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Registros",
        len(df),
    )

    if "nivel" in df.columns:

        valores = df[
            "nivel"
        ].dropna()

        if len(valores) > 0:

            col2.metric(
                "Último nivel",
                f"{valores.iloc[-1]:.2f} m",
            )

            col3.metric(
                "Máximo período",
                f"{valores.max():.2f} m",
            )

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

        st.subheader(
            "📋 Estadísticas"
        )

        valores = df[
            "nivel"
        ].dropna()

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


# ============================================================
# PRONÓSTICO
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
