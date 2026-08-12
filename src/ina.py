import requests
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

INA_DATOS_URL = "https://alerta.ina.gob.ar/pub/datos/datos"

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
# EXTRAER REGISTROS
# ============================================================

def extraer_registros(data):

    if data is None:
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for clave in [
            "data",
            "datos",
            "results",
            "result",
            "observaciones",
            "items",
        ]:

            valor = data.get(clave)

            if isinstance(valor, list):
                return valor

        for valor in data.values():

            if isinstance(valor, list):
                return valor

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

        if nombre in [
            "timestart",
            "time_start",
            "datetime",
            "date",
            "fecha",
            "timestamp",
            "time",
        ]:

            rename_map[columna] = "datetime"

        elif nombre in [
            "valor",
            "value",
            "nivel",
            "altura",
            "level",
            "height",
        ]:

            rename_map[columna] = "value"

    df = df.rename(columns=rename_map)

    # --------------------------------------------------------
    # Buscar fecha
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Buscar valor
    # --------------------------------------------------------

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
# CONSULTA INA
# ============================================================

def consultar_ina(start, end):

    global _LAST_DIAGNOSTIC

    parametros = {
        "timeStart": str(start),
        "timeEnd": str(end),
        "seriesId": str(SAN_NICOLAS_SERIES_ID),
        "format": "json",
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
    }

    diagnostico = {
        "configuracion": {
            "seriesId": SAN_NICOLAS_SERIES_ID,
            "timeStart": str(start),
            "timeEnd": str(end),
        },
        "intentos": [],
    }

    # ========================================================
    # INTENTO 1 - POST
    # ========================================================

    try:

        response = requests.post(
            INA_DATOS_URL,
            data=parametros,
            headers=headers,
            timeout=30,
        )

        intento_post = {
            "metodo": "POST",
            "http": response.status_code,
            "url": response.url,
            "texto": response.text[:5000],
        }

        diagnostico["intentos"].append(
            intento_post
        )

        response.raise_for_status()

        try:
            data = response.json()

        except ValueError:
            data = None

        registros = extraer_registros(
            data
        )

        intento_post[
            "cantidad_registros"
        ] = len(registros)

        if registros:

            _LAST_DIAGNOSTIC = diagnostico

            return data

    except Exception as exc:

        diagnostico[
            "error_post"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

    # ========================================================
    # INTENTO 2 - GET
    # ========================================================

    try:

        response = requests.get(
            INA_DATOS_URL,
            params=parametros,
            headers=headers,
            timeout=30,
        )

        intento_get = {
            "metodo": "GET",
            "http": response.status_code,
            "url": response.url,
            "texto": response.text[:5000],
        }

        diagnostico["intentos"].append(
            intento_get
        )

        response.raise_for_status()

        try:
            data = response.json()

        except ValueError:
            data = None

        registros = extraer_registros(
            data
        )

        intento_get[
            "cantidad_registros"
        ] = len(registros)

        _LAST_DIAGNOSTIC = diagnostico

        return data

    except Exception as exc:

        diagnostico[
            "error_get"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

        _LAST_DIAGNOSTIC = diagnostico

        return None


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def observed(start, end):

    global _LAST_DIAGNOSTIC

    try:

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

        if end_dt > today:
            end_dt = today

        if start_dt > end_dt:

            return (
                pd.DataFrame(),
                "La fecha Desde no puede ser posterior a Hasta.",
            )

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

        if data is None:

            return (
                pd.DataFrame(),
                "El INA no devolvió una respuesta válida.",
            )

        registros = extraer_registros(
            data
        )

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
                )

            if mensaje:

                return (
                    pd.DataFrame(),
                    f"Respuesta del INA: {mensaje}",
                )

            return (
                pd.DataFrame(),
                "INA respondió, pero no devolvió observaciones.",
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

        if "datetime" not in df.columns:

            return (
                pd.DataFrame(),
                (
                    "INA devolvió datos, pero no se identificó "
                    "la fecha. "
                    f"Columnas: {list(df.columns)}"
                ),
            )

        if "value" not in df.columns:

            return (
                pd.DataFrame(),
                (
                    "INA devolvió datos, pero no se identificó "
                    "el nivel. "
                    f"Columnas: {list(df.columns)}"
                ),
            )

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
                "No quedaron valores válidos después del procesamiento.",
            )

        return (
            df,
            None,
        )

    except requests.exceptions.Timeout:

        return (
            pd.DataFrame(),
            "La consulta al INA excedió el tiempo máximo de espera.",
        )

    except requests.exceptions.ConnectionError:

        return (
            pd.DataFrame(),
            "No fue posible conectar con el servidor del INA.",
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
