import requests
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

INA_BASE_URL = "https://alerta.ina.gob.ar/pub/datos"

INA_SERIES_URL = f"{INA_BASE_URL}/series"
INA_DATOS_URL = f"{INA_BASE_URL}/datos"

SAN_NICOLAS_SITE_CODE = 36
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
            "User-Agent": "Parana-San-Nicolas-V9/1.0",
            "Accept": "application/json,text/plain,*/*",
        }
    )

    return session


# ============================================================
# AUXILIAR PARA RESPUESTAS JSON
# ============================================================

def extraer_lista(data):
    """
    Busca una lista dentro de distintas estructuras JSON
    posibles devueltas por INA.
    """

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

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

                lista = extraer_lista(valor)

                if lista:
                    return lista

    for valor in data.values():

        if isinstance(valor, list):
            return valor

        if isinstance(valor, dict):

            lista = extraer_lista(valor)

            if lista:
                return lista

    return []


# ============================================================
# CONSULTA DE SERIES
# ============================================================

def consultar_series_diagnostico():

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

    diagnostico = {
        "url_series": response.url,
        "status_series": response.status_code,
        "content_type_series": response.headers.get(
            "Content-Type",
            "",
        ),
        "texto_series": response.text[:5000],
    }

    response.raise_for_status()

    try:

        json_data = response.json()

        diagnostico["json_series"] = json_data

    except ValueError:

        diagnostico["json_series"] = None

        return [], diagnostico

    registros = extraer_lista(
        json_data
    )

    diagnostico[
        "cantidad_series"
    ] = len(registros)

    return registros, diagnostico


# ============================================================
# IDENTIFICAR SERIES ID
# ============================================================

def extraer_series_ids(series):

    encontrados = []

    for registro in series:

        if not isinstance(
            registro,
            dict,
        ):
            continue

        # Guardamos información completa
        info = {
            "registro": registro,
            "seriesId": None,
        }

        for clave in [
            "id",
            "seriesId",
            "series_id",
            "serieId",
            "serie_id",
        ]:

            if clave in registro:

                valor = registro.get(
                    clave
                )

                if valor is not None:

                    try:

                        info[
                            "seriesId"
                        ] = int(valor)

                    except Exception:

                        info[
                            "seriesId"
                        ] = valor

                break

        encontrados.append(
            info
        )

    return encontrados


# ============================================================
# CONSULTAR DATOS POR SERIES ID
# ============================================================

def consultar_datos_series(
    series_id,
    start,
    end,
):

    params = {
        "timeStart": str(start),
        "timeEnd": str(end),
        "seriesId": series_id,
    }

    session = crear_sesion()

    response = session.get(
        INA_DATOS_URL,
        params=params,
        timeout=30,
    )

    diagnostico = {
        "metodo": "seriesId",
        "seriesId": series_id,
        "url_datos": response.url,
        "status_datos": response.status_code,
        "content_type_datos": response.headers.get(
            "Content-Type",
            "",
        ),
        "texto_datos": response.text[:5000],
    }

    response.raise_for_status()

    try:

        json_data = response.json()

        diagnostico[
            "json_datos"
        ] = json_data

    except ValueError:

        diagnostico[
            "json_datos"
        ] = None

        return [], diagnostico

    registros = extraer_lista(
        json_data
    )

    diagnostico[
        "cantidad_registros"
    ] = len(registros)

    return registros, diagnostico


# ============================================================
# CONSULTA DIRECTA SITE + VARIABLE
# ============================================================

def consultar_datos_directo(
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

    diagnostico = {
        "metodo": "siteCode+varId",
        "siteCode": SAN_NICOLAS_SITE_CODE,
        "varId": NIVEL_VAR_ID,
        "url_datos": response.url,
        "status_datos": response.status_code,
        "content_type_datos": response.headers.get(
            "Content-Type",
            "",
        ),
        "texto_datos": response.text[:5000],
    }

    response.raise_for_status()

    try:

        json_data = response.json()

        diagnostico[
            "json_datos"
        ] = json_data

    except ValueError:

        diagnostico[
            "json_datos"
        ] = None

        return [], diagnostico

    registros = extraer_lista(
        json_data
    )

    diagnostico[
        "cantidad_registros"
    ] = len(registros)

    return registros, diagnostico


# ============================================================
# NORMALIZAR DATAFRAME
# ============================================================

def normalizar_dataframe(
    registros
):

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(
        registros
    )

    if df.empty:
        return df

    rename_map = {}

    for col in df.columns:

        nombre = (
            str(col)
            .strip()
            .lower()
        )

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

            rename_map[
                col
            ] = "datetime"

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

            rename_map[
                col
            ] = "value"

    df = df.rename(
        columns=rename_map
    )

    # --------------------------------------------------------
    # Buscar fecha si no quedó normalizada
    # --------------------------------------------------------

    if "datetime" not in df.columns:

        for col in df.columns:

            nombre = str(
                col
            ).lower()

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

    # --------------------------------------------------------
    # Buscar nivel si no quedó normalizado
    # --------------------------------------------------------

    if "value" not in df.columns:

        for col in df.columns:

            nombre = str(
                col
            ).lower()

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

    if "datetime" in df.columns:

        df[
            "datetime"
        ] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )

    if "value" in df.columns:

        df[
            "value"
        ] = pd.to_numeric(
            df["value"],
            errors="coerce",
        )

    return df


# ============================================================
# CONSULTA PRINCIPAL
# ============================================================

def get_series(
    start,
    end,
):

    diagnostico_general = {
        "configuracion": {
            "siteCode": SAN_NICOLAS_SITE_CODE,
            "varId": NIVEL_VAR_ID,
            "timeStart": str(start),
            "timeEnd": str(end),
        }
    }

    # ========================================================
    # PASO 1: BUSCAR SERIES
    # ========================================================

    try:

        series, diag_series = (
            consultar_series_diagnostico()
        )

        diagnostico_general[
            "consulta_series"
        ] = diag_series

    except Exception as exc:

        series = []

        diagnostico_general[
            "error_series"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

    # ========================================================
    # MOSTRAR IDS DETECTADOS
    # ========================================================

    series_info = extraer_series_ids(
        series
    )

    diagnostico_general[
        "series_detectadas"
    ] = series_info

    # ========================================================
    # PASO 2: PROBAR TODAS LAS SERIES ENCONTRADAS
    # ========================================================

    intentos = []

    for info in series_info:

        series_id = info.get(
            "seriesId"
        )

        if series_id is None:
            continue

        try:

            registros, diag = (
                consultar_datos_series(
                    series_id=series_id,
                    start=start,
                    end=end,
                )
            )

            intentos.append(
                diag
            )

            if registros:

                diagnostico_general[
                    "intentos_datos"
                ] = intentos

                diagnostico_general[
                    "serie_utilizada"
                ] = series_id

                df = normalizar_dataframe(
                    registros
                )

                return (
                    df,
                    diagnostico_general,
                )

        except Exception as exc:

            intentos.append(
                {
                    "seriesId": series_id,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

    diagnostico_general[
        "intentos_datos"
    ] = intentos

    # ========================================================
    # PASO 3: CONSULTA DIRECTA
    # ========================================================

    try:

        registros, diag_directo = (
            consultar_datos_directo(
                start=start,
                end=end,
            )
        )

        diagnostico_general[
            "consulta_directa"
        ] = diag_directo

        if registros:

            df = normalizar_dataframe(
                registros
            )

            return (
                df,
                diagnostico_general,
            )

    except Exception as exc:

        diagnostico_general[
            "error_consulta_directa"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

    return (
        pd.DataFrame(),
        diagnostico_general,
    )


# ============================================================
# FUNCIÓN UTILIZADA POR APP.PY
# ============================================================

def observed(
    start,
    end,
):

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
                "Fecha inicial inválida.",
            )

        if pd.isna(end_dt):

            return (
                pd.DataFrame(),
                "Fecha final inválida.",
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
                (
                    "La fecha Desde no puede ser "
                    "posterior a la fecha disponible."
                ),
            )

        if start_dt == end_dt:

            start_dt = (
                end_dt
                - pd.Timedelta(days=1)
            )

        # ====================================================
        # CONSULTA
        # ====================================================

        df, diagnostico = get_series(
            start=start_dt.strftime(
                "%Y-%m-%d"
            ),
            end=end_dt.strftime(
                "%Y-%m-%d"
            ),
        )

        # Guardamos diagnóstico en atributo
        # para que app.py pueda mostrarlo.
        observed.last_diagnostic = (
            diagnostico
        )

        # ====================================================
        # SIN DATOS
        # ====================================================

        if df.empty:

            return (
                pd.DataFrame(),
                (
                    "El INA respondió, pero no devolvió "
                    "observaciones para San Nicolás en "
                    "el período seleccionado."
                ),
            )

        # ====================================================
        # VALIDAR COLUMNAS
        # ====================================================

        if "datetime" not in df.columns:

            return (
                pd.DataFrame(),
                (
                    "INA devolvió registros, pero no se "
                    "identificó la columna de fecha. "
                    f"Columnas: {list(df.columns)}"
                ),
            )

        if "value" not in df.columns:

            return (
                pd.DataFrame(),
                (
                    "INA devolvió registros, pero no se "
                    "identificó la columna de nivel. "
                    f"Columnas: {list(df.columns)}"
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

        df = (
            df
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        if df.empty:

            return (
                pd.DataFrame(),
                (
                    "INA devolvió registros, pero "
                    "ninguno contiene fecha y nivel válidos."
                ),
            )

        return (
            df,
            None,
        )

    except Exception as exc:

        observed.last_diagnostic = {
            "error_general": (
                f"{type(exc).__name__}: {exc}"
            )
        }

        return (
            pd.DataFrame(),
            (
                "Error procesando INA: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# ATRIBUTO DE DIAGNÓSTICO
# ============================================================

observed.last_diagnostic = {}


# ============================================================
# OBTENER DIAGNÓSTICO DESDE APP.PY
# ============================================================

def get_last_diagnostic():

    return getattr(
        observed,
        "last_diagnostic",
        {},
    )


# ============================================================
# METADATOS
# ============================================================

def forecast_meta():

    return {
        "fuente": (
            "Instituto Nacional del Agua (INA)"
        ),
        "estacion": "San Nicolás",
        "siteCode": SAN_NICOLAS_SITE_CODE,
        "variable": "Altura hidrométrica",
        "varId": NIVEL_VAR_ID,
        "unidad": "m",
        "observacion": (
            "Pronóstico experimental generado "
            "por el modelo propio."
        ),
    }
