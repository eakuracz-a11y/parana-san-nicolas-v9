import requests
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

INA_A5_URL = "https://alerta.ina.gob.ar/a5/getObservaciones"

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
# EXTRAER OBSERVACIONES
# ============================================================

def extraer_observaciones(data):
    """
    Intenta obtener la lista de observaciones desde las
    estructuras JSON posibles de la API A5 del INA.
    """

    if data is None:
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        claves_posibles = [
            "observaciones",
            "observations",
            "data",
            "datos",
            "results",
            "items",
        ]

        for clave in claves_posibles:

            valor = data.get(clave)

            if isinstance(valor, list):
                return valor

            if isinstance(valor, dict):

                resultado = extraer_observaciones(valor)

                if resultado:
                    return resultado

        # Buscar recursivamente cualquier lista
        for valor in data.values():

            if isinstance(valor, list):
                return valor

            if isinstance(valor, dict):

                resultado = extraer_observaciones(valor)

                if resultado:
                    return resultado

    return []


# ============================================================
# NORMALIZAR DATAFRAME
# ============================================================

def normalizar_dataframe(registros):

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(registros)

    if df.empty:
        return df

    rename_map = {}

    for columna in df.columns:

        nombre = str(columna).strip().lower()

        # ----------------------------------------------------
        # FECHA
        # ----------------------------------------------------

        if nombre in [
            "timestart",
            "time_start",
            "datetime",
            "timestamp",
            "fecha",
            "date",
            "time",
            "observedat",
            "observed_at",
        ]:

            rename_map[columna] = "datetime"

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

            rename_map[columna] = "value"

    df = df.rename(
        columns=rename_map
    )

    # ========================================================
    # BÚSQUEDA SECUNDARIA DE FECHA
    # ========================================================

    if "datetime" not in df.columns:

        for columna in df.columns:

            nombre = str(columna).lower()

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
    # BÚSQUEDA SECUNDARIA DE VALOR
    # ========================================================

    if "value" not in df.columns:

        for columna in df.columns:

            nombre = str(columna).lower()

            if (
                "valor" in nombre
                or "value" in nombre
                or "nivel" in nombre
                or "altura" in nombre
                or "level" in nombre
                or "height" in nombre
            ):

                df = df.rename(
                    columns={
                        columna: "value"
                    }
                )

                break

    # ========================================================
    # CONVERSIÓN DE TIPOS
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
# CONSULTA API A5 INA
# ============================================================

def consultar_ina_a5(start, end):

    global _LAST_DIAGNOSTIC

    params = {
        "tipo": "puntual",
        "series_id": SAN_NICOLAS_SERIES_ID,
        "timestart": str(start),
        "timeend": str(end),
    }

    headers = {
        "User-Agent": "Parana-San-Nicolas-V9/1.0",
        "Accept": "application/json,text/plain,*/*",
    }

    response = requests.get(
        INA_A5_URL,
        params=params,
        headers=headers,
        timeout=30,
    )

    _LAST_DIAGNOSTIC = {
        "configuracion": {
            "api": "A5",
            "tipo": "puntual",
            "seriesId": SAN_NICOLAS_SERIES_ID,
            "timeStart": str(start),
            "timeEnd": str(end),
        },
        "consulta_directa": {
            "status_datos": response.status_code,
            "url_datos": response.url,
            "texto_datos": response.text[:8000],
        },
    }

    response.raise_for_status()

    try:

        data = response.json()

    except ValueError:

        raise RuntimeError(
            "El INA respondió, pero la respuesta "
            "de la API A5 no es JSON."
        )

    _LAST_DIAGNOSTIC[
        "consulta_directa"
    ]["json_datos"] = data

    return data


# ============================================================
# FUNCIÓN PRINCIPAL UTILIZADA POR APP.PY
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

        today = pd.Timestamp.today().normalize()

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
        # FORMATO DE FECHA PARA API
        # ====================================================

        inicio = start_dt.strftime(
            "%Y-%m-%d"
        )

        fin = end_dt.strftime(
            "%Y-%m-%d"
        )

        # ====================================================
        # CONSULTAR INA
        # ====================================================

        data = consultar_ina_a5(
            inicio,
            fin,
        )

        registros = extraer_observaciones(
            data
        )

        _LAST_DIAGNOSTIC[
            "consulta_directa"
        ]["cantidad_registros"] = len(registros)

        # ====================================================
        # SIN REGISTROS
        # ====================================================

        if not registros:

            mensaje = None

            if isinstance(data, dict):

                mensaje = (
                    data.get("mensaje")
                    or data.get("message")
                    or data.get("error")
                    or data.get("detail")
                )

            if mensaje:

                return (
                    pd.DataFrame(),
                    f"Respuesta del INA: {mensaje}",
                )

            return (
                pd.DataFrame(),
                (
                    "La API A5 del INA respondió, "
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
                (
                    "El INA devolvió registros, "
                    "pero no pudieron convertirse "
                    "a una tabla válida."
                ),
            )

        # ====================================================
        # VALIDAR COLUMNAS
        # ====================================================

        if "datetime" not in df.columns:

            return (
                pd.DataFrame(),
                (
                    "INA devolvió observaciones, pero "
                    "no se identificó la fecha. "
                    f"Columnas recibidas: {list(df.columns)}"
                ),
            )

        if "value" not in df.columns:

            return (
                pd.DataFrame(),
                (
                    "INA devolvió observaciones, pero "
                    "no se identificó el nivel. "
                    f"Columnas recibidas: {list(df.columns)}"
                ),
            )

        # ====================================================
        # LIMPIEZA
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
                    "INA devolvió observaciones, pero "
                    "no quedaron valores válidos."
                ),
            )

        return (
            df,
            None,
        )

    # ========================================================
    # ERRORES
    # ========================================================

    except requests.exceptions.Timeout:

        return (
            pd.DataFrame(),
            "La consulta al INA excedió el tiempo máximo de espera.",
        )

    except requests.exceptions.HTTPError as exc:

        return (
            pd.DataFrame(),
            f"La API A5 del INA devolvió un error HTTP: {exc}",
        )

    except requests.exceptions.ConnectionError:

        return (
            pd.DataFrame(),
            "No fue posible conectar con la API del INA.",
        )

    except requests.exceptions.RequestException as exc:

        return (
            pd.DataFrame(),
            f"Error de comunicación con el INA: {exc}",
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
                "Error procesando datos del INA: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# METADATOS
# ============================================================

def forecast_meta():

    return {
        "fuente": "Instituto Nacional del Agua (INA)",
        "api": "A5",
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
