# ============================================================
# PARANÁ · SAN NICOLÁS
# src/ina.py
#
# Lectura robusta INA A5 - Serie 36
# ============================================================

from datetime import date, datetime

import pandas as pd
import requests


# ============================================================
# CONFIGURACIÓN
# ============================================================

INA_A5_URL = "https://alerta.ina.gob.ar/a5/getObservaciones"

REQUEST_TIMEOUT = 60

SAN_NICOLAS_SERIES_ID = 36


# ============================================================
# ESTACIONES
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


# ============================================================
# FECHAS
# ============================================================

def _format_date(value):
    """
    Convierte cualquier fecha aceptable a YYYY-MM-DD.
    """

    if value is None:
        raise ValueError("La fecha no puede estar vacía.")

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")

    parsed = pd.to_datetime(
        value,
        errors="raise",
    )

    return parsed.strftime("%Y-%m-%d")


# ============================================================
# REQUEST INA
# ============================================================

def _request_ina(start, end):
    """
    Consulta INA A5 usando la serie 36.
    """

    start_str = _format_date(start)
    end_str = _format_date(end)

    if pd.to_datetime(start_str) > pd.to_datetime(end_str):
        raise ValueError(
            "La fecha Desde no puede ser posterior a la fecha Hasta."
        )

    params = {
        "tipo": "puntual",
        "series_id": SAN_NICOLAS_SERIES_ID,
        "timestart": start_str,
        "timeend": end_str,
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "Parana-San-Nicolas-App/5.0",
    }

    try:

        response = requests.get(
            INA_A5_URL,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.Timeout as exc:

        raise RuntimeError(
            "El INA demoró demasiado en responder."
        ) from exc

    except requests.ConnectionError as exc:

        raise RuntimeError(
            "No fue posible establecer conexión con el INA."
        ) from exc

    except requests.RequestException as exc:

        raise RuntimeError(
            f"Error de comunicación con INA: {exc}"
        ) from exc

    if response.status_code != 200:

        raise RuntimeError(
            f"INA respondió HTTP {response.status_code}."
        )

    if not response.text.strip():

        raise RuntimeError(
            "INA respondió correctamente pero sin contenido."
        )

    try:

        data = response.json()

    except ValueError as exc:

        preview = response.text[:500]

        raise RuntimeError(
            "INA no devolvió JSON válido. "
            f"Respuesta inicial: {preview}"
        ) from exc

    return response, data, params


# ============================================================
# DETECTAR SI UN DICT PARECE OBSERVACIÓN
# ============================================================

def _looks_like_observation(obj):
    """
    Determina si un diccionario parece contener
    una fecha y un valor hidrométrico.
    """

    if not isinstance(obj, dict):
        return False

    keys = {
        str(key).strip().lower()
        for key in obj.keys()
    }

    date_keys = [
        "fecha",
        "datetime",
        "date",
        "timestamp",
        "time",
        "timestart",
        "fecha_hora",
        "fechahora",
    ]

    value_keys = [
        "valor",
        "value",
        "obsvalue",
        "obs_value",
        "nivel",
        "altura",
        "level",
    ]

    has_date = any(
        key in keys
        for key in date_keys
    )

    has_value = any(
        key in keys
        for key in value_keys
    )

    return has_date and has_value


# ============================================================
# EXTRAER REGISTROS RECURSIVAMENTE
# ============================================================

def _extract_records(data):
    """
    Recorre todo el JSON del INA buscando observaciones.
    """

    records = []

    def walk(obj):

        if isinstance(obj, list):

            for item in obj:

                if isinstance(item, dict) and _looks_like_observation(item):
                    records.append(item)
                else:
                    walk(item)

        elif isinstance(obj, dict):

            if _looks_like_observation(obj):
                records.append(obj)
                return

            for value in obj.values():
                walk(value)

    walk(data)

    return records


# ============================================================
# BUSCAR COLUMNA
# ============================================================

def _find_column(df, candidates):
    """
    Busca columnas ignorando mayúsculas/minúsculas.
    """

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in lookup:
            return lookup[key]

    return None


# ============================================================
# NORMALIZAR OBSERVACIONES
# ============================================================

def _normalize_records(records):
    """
    Convierte observaciones INA al formato estándar de app.py.
    """

    if not records:
        return pd.DataFrame()

    try:

        df = pd.json_normalize(records)

    except Exception:

        df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # FECHA
    # --------------------------------------------------------

    datetime_column = _find_column(
        df,
        [
            "timestart",
            "timeStart",
            "datetime",
            "fecha",
            "date",
            "timestamp",
            "time",
            "fecha_hora",
            "fechaHora",
        ],
    )

    # --------------------------------------------------------
    # VALOR
    # --------------------------------------------------------

    value_column = _find_column(
        df,
        [
            "valor",
            "value",
            "obsvalue",
            "obsValue",
            "obs_value",
            "nivel",
            "altura",
            "level",
        ],
    )

    # --------------------------------------------------------
    # BÚSQUEDA FLEXIBLE DE FECHA
    # --------------------------------------------------------

    if datetime_column is None:

        for column in df.columns:

            name = str(column).lower()

            if (
                "fecha" in name
                or "date" in name
                or "time" in name
            ):
                datetime_column = column
                break

    # --------------------------------------------------------
    # BÚSQUEDA FLEXIBLE DE VALOR
    # --------------------------------------------------------

    if value_column is None:

        for column in df.columns:

            name = str(column).lower()

            if (
                "valor" in name
                or "value" in name
                or "nivel" in name
                or "altura" in name
                or "obs" in name
            ):
                value_column = column
                break

    if datetime_column is None:

        raise RuntimeError(
            "No se encontró una columna temporal. "
            f"Columnas recibidas: {list(df.columns)}"
        )

    if value_column is None:

        raise RuntimeError(
            "No se encontró una columna de nivel. "
            f"Columnas recibidas: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # CONSTRUIR RESULTADO
    # --------------------------------------------------------

    result = pd.DataFrame()

    result["datetime"] = pd.to_datetime(
        df[datetime_column],
        errors="coerce",
        utc=True,
    )

    raw_values = (
        df[value_column]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    result["value"] = pd.to_numeric(
        raw_values,
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "datetime",
            "value",
        ]
    ).copy()

    if result.empty:

        raise RuntimeError(
            "El INA devolvió registros, pero no quedaron "
            "fechas y niveles válidos después de convertirlos."
        )

    # --------------------------------------------------------
    # FILTRO DE SEGURIDAD
    # --------------------------------------------------------

    result = result[
        result["value"].between(
            -5,
            20,
            inclusive="both",
        )
    ].copy()

    if result.empty:

        raise RuntimeError(
            "Los valores recibidos quedaron fuera del "
            "rango hidrométrico esperado."
        )

    # --------------------------------------------------------
    # ORDENAR
    # --------------------------------------------------------

    result = (
        result
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # METADATOS
    # --------------------------------------------------------

    result["station"] = "San Nicolás"

    result["station_code"] = 36

    result["series_id"] = SAN_NICOLAS_SERIES_ID

    result["variable"] = "Nivel hidrométrico"

    result["unit"] = "m"

    return result


# ============================================================
# GET SERIES
# ============================================================

def get_series(start, end):
    """
    Obtiene y normaliza la serie hidrométrica de San Nicolás.
    """

    response, data, params = _request_ina(
        start=start,
        end=end,
    )

    records = _extract_records(data)

    if not records:

        raise RuntimeError(
            "INA respondió HTTP 200, pero el parser "
            "no encontró observaciones dentro del JSON."
        )

    return _normalize_records(records)


# ============================================================
# FUNCIÓN PRINCIPAL APP.PY
# ============================================================

def observed(start, end):
    """
    app.py espera exactamente:

        df, error = observed(start, end)
    """

    try:

        df = get_series(
            start=start,
            end=end,
        )

        return (
            df,
            None,
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            str(exc),
        )


# ============================================================
# DIAGNÓSTICO
# ============================================================

def diagnostic(start, end):
    """
    Devuelve información detallada de la respuesta INA.
    """

    info = {
        "endpoint": INA_A5_URL,
        "series_id": SAN_NICOLAS_SERIES_ID,
        "tipo": "puntual",
        "desde": _format_date(start),
        "hasta": _format_date(end),

        "http_status": None,
        "registros": 0,
        "error": None,

        "json_tipo": None,
        "json_claves": None,
        "columnas_detectadas": None,
        "primer_registro": None,
        "respuesta_preview": None,
    }

    try:

        response, data, params = _request_ina(
            start=start,
            end=end,
        )

        info["http_status"] = response.status_code

        info["json_tipo"] = type(data).__name__

        # ----------------------------------------------------
        # CLAVES PRINCIPALES
        # ----------------------------------------------------

        if isinstance(data, dict):

            info["json_claves"] = list(
                data.keys()
            )

        elif isinstance(data, list):

            info["json_claves"] = [
                "respuesta_raiz_lista"
            ]

        else:

            info["json_claves"] = [
                type(data).__name__
            ]

        # ----------------------------------------------------
        # PREVIEW
        # ----------------------------------------------------

        info["respuesta_preview"] = str(
            data
        )[:2000]

        # ----------------------------------------------------
        # EXTRAER
        # ----------------------------------------------------

        records = _extract_records(data)

        info["registros"] = len(records)

        # ----------------------------------------------------
        # PRIMER REGISTRO
        # ----------------------------------------------------

        if records:

            info["primer_registro"] = str(
                records[0]
            )[:1500]

            try:

                temp = pd.json_normalize(
                    records
                )

                info["columnas_detectadas"] = list(
                    temp.columns
                )

            except Exception as exc:

                info["columnas_detectadas"] = [
                    f"Error al detectar columnas: {exc}"
                ]

        else:

            info["columnas_detectadas"] = []

        return info

    except Exception as exc:

        info["error"] = str(exc)

        return info


# ============================================================
# INFORMACIÓN DEL PRONÓSTICO
# ============================================================

def forecast_meta():
    """
    Información utilizada por app.py.
    """

    return {
        "fuente": "Instituto Nacional del Agua (INA)",
        "servicio": "INA A5 - getObservaciones",
        "estacion": "San Nicolás",
        "serie": SAN_NICOLAS_SERIES_ID,
        "variable": "Nivel hidrométrico",
        "unidad": "metros",
        "estado": "Consulta online",
        "observacion": (
            "Los datos observados son obtenidos "
            "del servicio público del Instituto "
            "Nacional del Agua (INA). "
            "La predicción corresponde al modelo "
            "experimental propio de la plataforma."
        ),
    }
