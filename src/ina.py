import requests
import pandas as pd


# ============================================================
# CONFIGURACIÓN INA
# ============================================================

INA_DATOS_URL = "https://alerta.ina.gob.ar/pub/datos/datos"

# Serie observada confirmada por diagnóstico:
#
# Estación: San Nicolás
# sitecode: 36
# variable: Altura hidrométrica
# varid: 2
# procedimiento: medición directa
#
SAN_NICOLAS_SERIES_ID = 36


STATIONS = [
    "Corrientes",
    "Goya",
    "La Paz",
    "Paraná",
    "Diamante",
    "Rosario",
    "Villa Constitución",
    "San Nicolás",
]


# ============================================================
# DIAGNÓSTICO
# ============================================================

_LAST_DIAGNOSTIC = {}


def get_last_diagnostic():
    return _LAST_DIAGNOSTIC


# ============================================================
# CONSULTA INA
# ============================================================

def consultar_ina(start, end):

    global _LAST_DIAGNOSTIC

    # --------------------------------------------------------
    # IMPORTANTE:
    # enviamos seriesId confirmado = 36
    # --------------------------------------------------------

    params = {
        "seriesId": SAN_NICOLAS_SERIES_ID,
        "timeStart": str(start),
        "timeEnd": str(end),
    }

    headers = {
        "User-Agent": "Parana-San-Nicolas-V9/1.0",
        "Accept": "application/json",
    }

    response = requests.get(
        INA_DATOS_URL,
        params=params,
        headers=headers,
        timeout=30,
    )

    # --------------------------------------------------------
    # DIAGNÓSTICO
    # --------------------------------------------------------

    _LAST_DIAGNOSTIC = {
        "configuracion": {
            "seriesId": SAN_NICOLAS_SERIES_ID,
            "timeStart": str(start),
            "timeEnd": str(end),
        },
        "consulta_directa": {
            "status_datos": response.status_code,
            "url_datos": response.url,
            "texto_datos": response.text[:5000],
        },
    }

    response.raise_for_status()

    # --------------------------------------------------------
    # LEER JSON
    # --------------------------------------------------------

    try:

        data = response.json()

    except ValueError:

        raise RuntimeError(
            "El INA respondió, pero la respuesta no es JSON."
        )

    _LAST_DIAGNOSTIC[
        "consulta_directa"
    ]["json_datos"] = data

    return data


# ============================================================
# EXTRAER REGISTROS
# ============================================================

def extraer_registros(data):

    if data is None:
        return []

    # --------------------------------------------------------
    # RESPUESTA DIRECTAMENTE COMO LISTA
    # --------------------------------------------------------

    if isinstance(data, list):
        return data

    # --------------------------------------------------------
    # RESPUESTA COMO DICCIONARIO
    # --------------------------------------------------------

    if isinstance(data, dict):

        posibles_claves = [
            "data",
            "datos",
            "results",
            "result",
            "observaciones",
            "items",
        ]

        for clave in posibles_claves:

            if clave in data:

                valor = data[clave]

                if isinstance(valor, list):
                    return valor

        # ----------------------------------------------------
        # Buscar automáticamente una lista
        # ----------------------------------------------------

        for valor in data.values():

            if isinstance(valor, list):
                return valor

    return []


# ============================================================
# NORMALIZAR DATOS
# ============================================================

def normalizar_dataframe(registros):

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(
        registros
    )

    if df.empty:
        return df

    rename_map = {}

    # ========================================================
    # IDENTIFICAR COLUMNAS
    # ========================================================

    for columna in df.columns:

        nombre = (
            str(columna)
            .strip()
            .lower()
        )

        # ----------------------------------------------------
        # FECHA
        # ----------------------------------------------------

        if nombre in [
            "timestart",
            "time_start",
            "datetime",
            "date",
            "fecha",
            "timestamp",
            "time",
        ]:

            rename_map[
                columna
            ] = "datetime"

        # ----------------------------------------------------
        # VALOR
        # ----------------------------------------------------

        elif nombre in [
            "valor",
            "value",
            "nivel",
            "altura",
            "level",
            "height",
        ]:

            rename_map[
                columna
            ] = "value"

    df = df.rename(
        columns=rename_map
    )

    # ========================================================
    # BÚSQUEDA ADICIONAL DE FECHA
    # ========================================================

    if "datetime" not in df.columns:

        for columna in df.columns:

            nombre = str(
                columna
            ).lower()

            if (
                "time" in nombre
                or "date" in nombre
                or "fecha" in nombre
            ):

                df = df.rename(
                    columns={
                        columna: "datetime"
                    }
                )

                break

    # ========================================================
    # BÚSQUEDA ADICIONAL DE VALOR
    # ========================================================

    if "value" not in df.columns:

        for columna in df.columns:

            nombre = str(
                columna
            ).lower()

            if (
                "valor" in nombre
                or "value" in nombre
                or "nivel" in nombre
                or "altura" in nombre
            ):

                df = df.rename(
                    columns={
                        columna: "value"
                    }
                )

                break

    # ========================================================
    # CONVERTIR TIPOS
    # ========================================================

    if "datetime" in df.columns:

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )

    if "value" in df.columns:

        df["value"] = pd.to_numeric(
            df["value"],
            errors="coerce",
        )

    return df


# ============================================================
# FUNCIÓN PRINCIPAL USADA POR APP.PY
# ============================================================

def observed(start, end):

    global _LAST_DIAGNOSTIC

    try:

        # ====================================================
        # VALIDAR FECHAS
        # ====================================================

        start_dt = pd.to_datetime(
            start,
            errors="coerce",
        )

        end_dt = pd.to_datetime(
            end,
            errors="coerce",
        )

        if pd.isna(start_dt):

            return (
                pd.DataFrame(),
                "La fecha inicial no es válida.",
            )

        if pd.isna(end_dt):

            return (
                pd.DataFrame(),
                "La fecha final no es válida.",
            )

        today = (
            pd.Timestamp
            .today()
            .normalize()
        )

        # ----------------------------------------------------
        # No consultar futuro
        # ----------------------------------------------------

        if end_dt > today:
            end_dt = today

        if start_dt > end_dt:

            return (
                pd.DataFrame(),
                "La fecha Desde no puede ser posterior a Hasta.",
            )

        # ====================================================
        # FORMATO PARA INA
        # ====================================================

        inicio = start_dt.strftime(
            "%Y-%m-%d"
        )

        fin = end_dt.strftime(
            "%Y-%m-%d"
        )

        # ====================================================
        # CONSULTAR
        # ====================================================

        data = consultar_ina(
            inicio,
            fin,
        )

        registros = extraer_registros(
            data
        )

        _LAST_DIAGNOSTIC[
            "consulta_directa"
        ][
            "cantidad_registros"
        ] = len(registros)

        # ====================================================
        # SIN REGISTROS
        # ====================================================

        if not registros:

            mensaje_api = None

            if isinstance(
                data,
                dict,
            ):

                mensaje_api = (
                    data.get("mensaje")
                    or data.get("message")
                    or data.get("error")
                )

            if mensaje_api:

                return (
                    pd.DataFrame(),
                    f"Respuesta del INA: {mensaje_api}",
                )

            return (
                pd.DataFrame(),
                (
                    "El INA respondió correctamente, "
                    "pero no devolvió observaciones "
                    "para la serie 36."
                ),
            )

        # ====================================================
        # NORMALIZAR
        # ====================================================

        df = normalizar_dataframe(
            registros
        )

        if df.empty:

            return (
                pd.DataFrame(),
                "INA devolvió registros, pero no pudieron procesarse.",
            )

        # ====================================================
        # VALIDAR COLUMNAS
        # ====================================================

        if "datetime" not in df.columns:

            return (
                pd.DataFrame(),
                (
                    "INA devolvió datos, pero no se identificó "
                    "la columna de fecha. "
                    f"Columnas: {list(df.columns)}"
                ),
            )

        if "value" not in df.columns:

            return (
                pd.DataFrame(),
                (
                    "INA devolvió datos, pero no se identificó "
                    "la columna de nivel. "
                    f"Columnas: {list(df.columns)}"
                ),
            )

        # ====================================================
        # LIMPIAR
        # ====================================================

        df = df.dropna(
            subset=[
                "datetime",
                "value",
            ]
        )

        df = (
            df
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        if df.empty:

            return (
                pd.DataFrame(),
                (
                    "INA devolvió observaciones, "
                    "pero no quedaron valores válidos."
                ),
            )

        return (
            df,
            None,
        )

    # ========================================================
    # ERRORES HTTP
    # ========================================================

    except requests.exceptions.Timeout:

        return (
            pd.DataFrame(),
            "La consulta al INA excedió el tiempo de espera.",
        )

    except requests.exceptions.HTTPError as exc:

        return (
            pd.DataFrame(),
            f"Error HTTP del INA: {exc}",
        )

    except requests.exceptions.ConnectionError:

        return (
            pd.DataFrame(),
            "No fue posible conectar con el servidor del INA.",
        )

    except requests.exceptions.RequestException as exc:

        return (
            pd.DataFrame(),
            f"Error de comunicación con INA: {exc}",
        )

    except Exception as exc:

        _LAST_DIAGNOSTIC[
            "error_general"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

        return (
            pd.DataFrame(),
            (
                "Error procesando INA: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# METADATOS
# ============================================================

def forecast_meta():

    return {
        "fuente": "Instituto Nacional del Agua (INA)",
        "estacion": "San Nicolás",
        "siteCode": 36,
        "seriesId": SAN_NICOLAS_SERIES_ID,
        "variable": "Altura hidrométrica",
        "varId": 2,
        "procedimiento": "medición directa",
        "unidad": "m",
        "observacion": (
            "Pronóstico experimental generado "
            "por el modelo propio."
        ),
    }
