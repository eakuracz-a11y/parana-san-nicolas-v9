import requests
import pandas as pd


# ============================================================
# PARANÁ - SAN NICOLÁS V9
# Conexión con API pública del Instituto Nacional del Agua
# ============================================================

INA_BASE_URL = "https://alerta.ina.gob.ar/pub/datos"


# ============================================================
# ESTACIONES DISPONIBLES EN LA PLATAFORMA
# ============================================================

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
# CONFIGURACIÓN DE ESTACIONES
# ============================================================

STATION_CODES = {
    "Corrientes": 19,
    "Goya": 23,
    "La Paz": 26,
    "Paraná": 29,
    "Diamante": 31,
    "Rosario": 34,
    "Villa Constitución": 35,
    "San Nicolás": 36,
}


# Variable 2 = nivel hidrométrico
LEVEL_VARIABLE_ID = 2

# Serie oficial de San Nicolás
SAN_NICOLAS_SERIES_ID = 36


# ============================================================
# CONSULTA GENERAL AL INA
# ============================================================

def get_series(
    start,
    end,
    site_code=None,
    var_id=None,
    series_id=None
):
    """
    Consulta datos observados del INA.

    Parámetros:
        start: fecha inicial YYYY-MM-DD
        end: fecha final YYYY-MM-DD
        site_code: código de estación
        var_id: variable hidrológica
        series_id: identificador de serie

    Devuelve:
        DataFrame con los datos recibidos.
    """

    params = {
        "timeStart": str(start),
        "timeEnd": str(end),
        "format": "json",
    }

    # El INA permite consultar directamente por seriesId.
    if series_id is not None:
        params["seriesId"] = series_id

    else:
        if site_code is not None:
            params["siteCode"] = site_code

        if var_id is not None:
            params["varId"] = var_id

    url = f"{INA_BASE_URL}/datos"

    try:

        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:

        raise RuntimeError(
            f"No fue posible conectarse con el INA: {exc}"
        ) from exc

    except ValueError as exc:

        raise RuntimeError(
            "El INA no devolvió una respuesta JSON válida."
        ) from exc


    # ========================================================
    # RESPUESTAS DE ERROR DEL INA
    # ========================================================

    if isinstance(data, dict):

        # Error típico del servicio INA
        if "mensaje" in data:

            raise RuntimeError(
                f"INA: {data.get('mensaje')}"
            )

        if "message" in data:

            raise RuntimeError(
                f"INA: {data.get('message')}"
            )


        # Algunas respuestas vienen dentro de "data"
        if "data" in data:

            data = data["data"]


        elif "results" in data:

            data = data["results"]


        elif "observaciones" in data:

            data = data["observaciones"]


    # ========================================================
    # NORMALIZAR RESPUESTA
    # ========================================================

    if data is None:

        return pd.DataFrame()


    if isinstance(data, dict):

        data = [data]


    if not isinstance(data, list):

        return pd.DataFrame()


    df = pd.DataFrame(data)


    if df.empty:

        return df


    # ========================================================
    # NORMALIZACIÓN DE COLUMNAS
    # ========================================================

    original_columns = list(df.columns)

    rename_map = {}


    for column in original_columns:

        name = str(column).lower().strip()


        # Fecha / hora
        if name in [
            "fecha",
            "datetime",
            "date",
            "timestamp",
            "time",
            "timestart",
            "timeend"
        ]:

            rename_map[column] = "datetime"


        # Nivel / valor
        elif name in [
            "valor",
            "value",
            "nivel",
            "altura",
            "altura_m",
            "level"
        ]:

            rename_map[column] = "value"


    df = df.rename(columns=rename_map)


    # ========================================================
    # BUSCAR FECHA AUTOMÁTICAMENTE
    # ========================================================

    if "datetime" not in df.columns:

        possible_datetime_columns = [
            "fecha",
            "date",
            "timestamp",
            "time",
            "timestart"
        ]


        for column in possible_datetime_columns:

            if column in df.columns:

                df["datetime"] = df[column]

                break


    # ========================================================
    # BUSCAR NIVEL AUTOMÁTICAMENTE
    # ========================================================

    if "value" not in df.columns:

        possible_value_columns = [
            "valor",
            "value",
            "nivel",
            "altura",
            "level"
        ]


        for column in possible_value_columns:

            if column in df.columns:

                df["value"] = df[column]

                break


    # ========================================================
    # CONVERTIR FECHA
    # ========================================================

    if "datetime" in df.columns:

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
            utc=True
        )


    # ========================================================
    # CONVERTIR NIVEL A NÚMERO
    # ========================================================

    if "value" in df.columns:

        df["value"] = pd.to_numeric(
            df["value"],
            errors="coerce"
        )


    return df


# ============================================================
# FUNCIÓN PRINCIPAL UTILIZADA POR APP.PY
# ============================================================

def observed(start, end):
    """
    Obtiene los datos observados de nivel del río Paraná
    en la estación San Nicolás.

    IMPORTANTE:
    Esta función devuelve DOS valores porque app.py
    espera:

        df, error = observed(start, end)
    """

    try:

        df = get_series(
            start=start,
            end=end,
            series_id=SAN_NICOLAS_SERIES_ID
        )


        # ====================================================
        # SIN DATOS
        # ====================================================

        if df.empty:

            return (
                pd.DataFrame(),
                "El INA no devolvió datos para el período seleccionado."
            )


        # ====================================================
        # VERIFICAR FECHA
        # ====================================================

        if "datetime" not in df.columns:

            return (
                df,
                "Los datos recibidos del INA no contienen una columna temporal reconocible."
            )


        # ====================================================
        # VERIFICAR NIVEL
        # ====================================================

        if "value" not in df.columns:

            return (
                df,
                "Los datos recibidos del INA no contienen una columna de nivel reconocible."
            )


        # ====================================================
        # LIMPIAR REGISTROS INVÁLIDOS
        # ====================================================

        df = df.dropna(
            subset=["datetime", "value"]
        ).copy()


        if df.empty:

            return (
                df,
                "El INA respondió correctamente, pero no existen valores válidos de nivel."
            )


        # ====================================================
        # ORDEN CRONOLÓGICO
        # ====================================================

        df = df.sort_values(
            "datetime"
        ).reset_index(drop=True)


        # ====================================================
        # INFORMACIÓN ADICIONAL
        # ====================================================

        df["station"] = "San Nicolás"

        df["station_code"] = STATION_CODES["San Nicolás"]

        df["variable"] = "Nivel hidrométrico"

        df["unit"] = "m"


        return (
            df,
            None
        )


    except Exception as exc:

        return (
            pd.DataFrame(),
            str(exc)
        )


# ============================================================
# INFORMACIÓN DEL PRONÓSTICO
# ============================================================

def forecast_meta():
    """
    Información sobre el modelo experimental.
    """

    return {

        "fuente": "Instituto Nacional del Agua (INA)",

        "servicio": "Sistema de Información Hidrológica",

        "estacion": "San Nicolás",

        "serie": SAN_NICOLAS_SERIES_ID,

        "variable": "Nivel hidrométrico",

        "unidad": "metros",

        "estado": "Consulta disponible",

        "observacion": (
            "Los datos observados son obtenidos del servicio "
            "público del Instituto Nacional del Agua (INA). "
            "La predicción experimental de la evolución del "
            "nivel del río Paraná en San Nicolás corresponde "
            "al modelo propio de esta plataforma."
        ),
    }
