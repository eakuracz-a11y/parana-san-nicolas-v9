import requests
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

INA_BASE_URL = "https://alerta.ina.gob.ar/pub/datos"

INA_SERIES_URL = f"{INA_BASE_URL}/series"
INA_DATOS_URL = f"{INA_BASE_URL}/datos"

# San Nicolás
SAN_NICOLAS_SITE_CODE = 36

# Altura hidrométrica
NIVEL_VAR_ID = 2


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
# SESIÓN HTTP
# ============================================================

def crear_sesion():

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(compatible; Parana-San-Nicolas-V9/1.0)"
            ),
            "Accept": "application/json,text/plain,*/*",
        }
    )

    return session


# ============================================================
# NORMALIZAR JSON
# ============================================================

def extraer_lista(data):
    """
    Intenta obtener una lista desde distintas estructuras JSON
    posibles devueltas por la API del INA.
    """

    if data is None:
        return []

    # Respuesta directa como lista
    if isinstance(data, list):
        return data

    # Respuesta como diccionario
    if isinstance(data, dict):

        claves_posibles = [
            "data",
            "datos",
            "results",
            "result",
            "series",
            "observaciones",
            "items",
        ]

        for clave in claves_posibles:

            if clave in data:

                valor = data[clave]

                if isinstance(valor, list):
                    return valor

                if isinstance(valor, dict):
                    resultado = extraer_lista(valor)

                    if resultado:
                        return resultado

        # Buscar cualquier lista dentro del diccionario
        for valor in data.values():

            if isinstance(valor, list):
                return valor

            if isinstance(valor, dict):

                resultado = extraer_lista(valor)

                if resultado:
                    return resultado

    return []


# ============================================================
# BUSCAR ID DE SERIE
# ============================================================

def obtener_series():
    """
    Consulta las series observadas disponibles para
    San Nicolás y la variable altura hidrométrica.
    """

    params = {
        "siteCode": SAN_NICOLAS_SITE_CODE,
        "varId": NIVEL_VAR_ID,
    }

    session = crear_sesion()

    response = session.get(
        INA_SERIES_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    try:
        data = response.json()

    except ValueError:
        raise RuntimeError(
            "El INA respondió a la consulta de series, "
            "pero la respuesta no es JSON."
        )

    registros = extraer_lista(data)

    return registros


def encontrar_series_id():
    """
    Obtiene automáticamente un seriesId observado
    correspondiente a San Nicolás.
    """

    series = obtener_series()

    if not series:
        return None

    ids_encontrados = []

    for registro in series:

        if not isinstance(registro, dict):
            continue

        # ----------------------------------------------------
        # Posibles nombres del ID
        # ----------------------------------------------------

        posibles_claves = [
            "id",
            "seriesId",
            "series_id",
            "serieId",
            "serie_id",
        ]

        for clave in posibles_claves:

            if clave in registro:

                valor = registro.get(clave)

                if valor is not None:

                    try:
                        ids_encontrados.append(int(valor))

                    except (TypeError, ValueError):
                        pass

                    break

    if not ids_encontrados:
        return None

    # Usar primera serie observada disponible
    return ids_encontrados[0]


# ============================================================
# NORMALIZAR DATOS OBSERVADOS
# ============================================================

def normalizar_dataframe(registros):

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(registros)

    if df.empty:
        return df

    rename_map = {}

    # ========================================================
    # DETECTAR COLUMNAS
    # ========================================================

    for col in df.columns:

        nombre = str(col).strip().lower()

        # ----------------------------------------------------
        # FECHA
        # ----------------------------------------------------

        if nombre in [
            "fecha",
            "datetime",
            "date",
            "timestamp",
            "timestart",
            "time_start",
            "time",
            "observedat",
            "observed_at",
        ]:

            rename_map[col] = "datetime"

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
            "dato",
            "measurement",
        ]:

            rename_map[col] = "value"

    df = df.rename(
        columns=rename_map
    )

    # ========================================================
    # BÚSQUEDA SECUNDARIA DE FECHA
    # ========================================================

    if "datetime" not in df.columns:

        for col in df.columns:

            nombre = str(col).lower()

            if (
                "fecha" in nombre
                or "date" in nombre
                or "time" in nombre
            ):

                df = df.rename(
                    columns={
                        col: "datetime"
                    }
                )

                break

    # ========================================================
    # BÚSQUEDA SECUNDARIA DE NIVEL
    # ========================================================

    if "value" not in df.columns:

        for col in df.columns:

            nombre = str(col).lower()

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
                        col: "value"
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
# CONSULTAR DATOS POR SERIES ID
# ============================================================

def consultar_por_series_id(
    series_id,
    start,
    end,
):

    params = {
        "timeStart": str(start),
        "timeEnd": str(end),
        "seriesId": int(series_id),
    }

    session = crear_sesion()

    response = session.get(
        INA_DATOS_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    try:

        data = response.json()

    except ValueError:

        raise RuntimeError(
            "El INA respondió a la consulta de datos, "
            "pero la respuesta no es JSON."
        )

    registros = extraer_lista(data)

    return normalizar_dataframe(
        registros
    )


# ============================================================
# CONSULTA DIRECTA siteCode + varId
# ============================================================

def consultar_por_estacion(
    start,
    end,
):

    params = {
        "timeStart": str(start),
        "timeEnd": str(end),
        "siteCode": SAN_NICOLAS_SITE_CODE,
        "varId": NIVEL_VAR_ID,
    }

    session = crear_sesion()

    response = session.get(
        INA_DATOS_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    try:

        data = response.json()

    except ValueError:

        raise RuntimeError(
            "El INA respondió a la consulta directa, "
            "pero la respuesta no es JSON."
        )

    registros = extraer_lista(
        data
    )

    return normalizar_dataframe(
        registros
    )


# ============================================================
# FUNCIÓN PRINCIPAL DE CONSULTA
# ============================================================

def get_series(
    start,
    end,
):

    # --------------------------------------------------------
    # MÉTODO 1
    # Buscar automáticamente el seriesId observado
    # --------------------------------------------------------

    series_id = None

    try:

        series_id = encontrar_series_id()

    except Exception:
        # Si falla la búsqueda de series,
        # todavía intentamos consulta directa.
        series_id = None

    # --------------------------------------------------------
    # CONSULTAR POR SERIES ID
    # --------------------------------------------------------

    if series_id is not None:

        try:

            df = consultar_por_series_id(
                series_id=series_id,
                start=start,
                end=end,
            )

            if not df.empty:

                df["series_id"] = series_id

                return df

        except Exception:
            pass

    # --------------------------------------------------------
    # MÉTODO 2
    # Consulta directa por estación y variable
    # --------------------------------------------------------

    df = consultar_por_estacion(
        start=start,
        end=end,
    )

    return df


# ============================================================
# FUNCIÓN UTILIZADA POR app.py
# ============================================================

def observed(
    start,
    end,
):

    try:

        # ====================================================
        # FECHAS
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

        # ----------------------------------------------------
        # Fecha actual
        # ----------------------------------------------------

        today = pd.Timestamp.today().normalize()

        # ----------------------------------------------------
        # No consultar fechas futuras
        # ----------------------------------------------------

        if end_dt > today:

            end_dt = today

        # ----------------------------------------------------
        # Rango inválido
        # ----------------------------------------------------

        if start_dt > end_dt:

            return (
                pd.DataFrame(),
                "La fecha Desde no puede ser posterior "
                "a la fecha disponible más reciente.",
            )

        # Si son iguales, ampliamos 1 día hacia atrás.
        if start_dt == end_dt:

            start_dt = (
                end_dt
                - pd.Timedelta(days=1)
            )

        # ====================================================
        # CONSULTAR API
        # ====================================================

        df = get_series(
            start=start_dt.strftime(
                "%Y-%m-%d"
            ),
            end=end_dt.strftime(
                "%Y-%m-%d"
            ),
        )

        # ====================================================
        # VALIDAR RESPUESTA
        # ====================================================

        if df is None:

            return (
                pd.DataFrame(),
                "El INA no devolvió una respuesta válida.",
            )

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            return (
                pd.DataFrame(),
                "La respuesta del INA no tiene "
                "el formato esperado.",
            )

        if df.empty:

            return (
                pd.DataFrame(),
                (
                    "El INA respondió, pero no devolvió "
                    "observaciones para San Nicolás "
                    "en el período seleccionado."
                ),
            )

        # ====================================================
        # VALIDAR FECHA
        # ====================================================

        if "datetime" not in df.columns:

            return (
                pd.DataFrame(),
                (
                    "El INA devolvió registros, pero no fue "
                    "posible identificar la fecha. "
                    f"Columnas recibidas: {list(df.columns)}"
                ),
            )

        # ====================================================
        # VALIDAR NIVEL
        # ====================================================

        if "value" not in df.columns:

            return (
                pd.DataFrame(),
                (
                    "El INA devolvió registros, pero no fue "
                    "posible identificar el nivel. "
                    f"Columnas recibidas: {list(df.columns)}"
                ),
            )

        # ====================================================
        # LIMPIAR
        # ====================================================

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )

        df["value"] = pd.to_numeric(
            df["value"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "datetime",
                "value",
            ]
        )

        if df.empty:

            return (
                pd.DataFrame(),
                (
                    "El INA devolvió registros, pero "
                    "ninguno contiene fecha y nivel válidos."
                ),
            )

        # ----------------------------------------------------
        # Orden cronológico
        # ----------------------------------------------------

        df = (
            df
            .sort_values("datetime")
            .reset_index(drop=True)
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
            (
                "La consulta al INA excedió "
                "el tiempo máximo de espera."
            ),
        )

    except requests.exceptions.HTTPError as exc:

        return (
            pd.DataFrame(),
            (
                "El servidor del INA devolvió "
                f"un error HTTP: {exc}"
            ),
        )

    except requests.exceptions.ConnectionError:

        return (
            pd.DataFrame(),
            (
                "No fue posible establecer conexión "
                "con el servidor del INA."
            ),
        )

    except requests.exceptions.RequestException as exc:

        return (
            pd.DataFrame(),
            (
                "Error de comunicación con el INA: "
                f"{exc}"
            ),
        )

    # ========================================================
    # OTROS ERRORES
    # ========================================================

    except Exception as exc:

        return (
            pd.DataFrame(),
            (
                "Error procesando los datos del INA: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# METADATOS
# ============================================================

def forecast_meta():

    series_id = None

    try:

        series_id = encontrar_series_id()

    except Exception:
        pass

    return {
        "fuente": "Instituto Nacional del Agua (INA)",
        "estacion": "San Nicolás",
        "siteCode": SAN_NICOLAS_SITE_CODE,
        "variable": "Altura hidrométrica",
        "varId": NIVEL_VAR_ID,
        "seriesId": series_id,
        "unidad": "m",
        "observacion": (
            "Pronóstico experimental generado "
            "por el modelo propio."
        ),
    }
