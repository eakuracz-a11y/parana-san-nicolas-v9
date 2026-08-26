# ============================================================
# PARANÁ · SAN NICOLÁS
# src/ina.py
#
# Consulta de datos públicos del Instituto Nacional del Agua
# Endpoint A5 / getObservaciones
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
# ESTACIONES UTILIZADAS POR LA APP
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
    Convierte una fecha a formato YYYY-MM-DD.
    """

    if value is None:
        raise ValueError("La fecha no puede estar vacía.")

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")

    parsed = pd.to_datetime(value, errors="raise")

    return parsed.strftime("%Y-%m-%d")


# ============================================================
# EXTRAER REGISTROS DE LA RESPUESTA JSON
# ============================================================

def _extract_records(data):
    """
    Intenta encontrar las observaciones dentro de la respuesta
    JSON del INA.
    """

    if data is None:
        return []

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    possible_keys = [
        "observaciones",
        "data",
        "datos",
        "results",
        "result",
        "items",
    ]

    for key in possible_keys:

        value = data.get(key)

        if isinstance(value, list):
            return value

        if isinstance(value, dict):

            nested = _extract_records(value)

            if nested:
                return nested

    # En algunos casos puede venir una sola observación
    # directamente como diccionario.

    lower_keys = {
        str(key).lower()
        for key in data.keys()
    }

    date_keys = {
        "fecha",
        "datetime",
        "date",
        "timestamp",
        "timestart",
        "time",
    }

    value_keys = {
        "valor",
        "value",
        "obsvalue",
        "nivel",
        "altura",
        "level",
    }

    if (
        lower_keys.intersection(date_keys)
        and lower_keys.intersection(value_keys)
    ):
        return [data]

    return []


# ============================================================
# BUSCAR COLUMNA
# ============================================================

def _find_column(df, candidates):
    """
    Busca una columna ignorando mayúsculas/minúsculas.
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
# NORMALIZAR DATOS
# ============================================================

def _normalize_observations(records):
    """
    Convierte la respuesta del INA al formato esperado por app.py.
    """

    if not records:
        return pd.DataFrame()

    try:
        df = pd.json_normalize(records)
    except Exception:
        df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame()

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
            "fechaHora",
            "fecha_hora",
        ],
    )

    value_column = _find_column(
        df,
        [
            "valor",
            "value",
            "obsvalue",
            "obsValue",
            "nivel",
            "altura",
            "level",
        ],
    )

    # --------------------------------------------------------
    # BUSQUEDA ALTERNATIVA DE FECHA
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
    # BUSQUEDA ALTERNATIVA DE VALOR
    # --------------------------------------------------------

    if value_column is None:

        for column in df.columns:

            name = str(column).lower()

            if (
                "valor" in name
                or "value" in name
                or "nivel" in name
                or "altura" in name
            ):
                value_column = column
                break

    if datetime_column is None:
        return pd.DataFrame()

    if value_column is None:
        return pd.DataFrame()

    result = pd.DataFrame()

    result["datetime"] = pd.to_datetime(
        df[datetime_column],
        errors="coerce",
        utc=True,
    )

    values = (
        df[value_column]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    result["value"] = pd.to_numeric(
        values,
        errors="coerce",
    )

    result = result.dropna(
        subset=["datetime", "value"]
    ).copy()

    if result.empty:
        return pd.DataFrame()

    # Rango amplio para evitar eliminar crecidas históricas
    result = result[
        result["value"].between(-5, 20)
    ].copy()

    if result.empty:
        return pd.DataFrame()

    result = (
        result
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    result["station"] = "San Nicolás"
    result["station_code"] = 36
    result["series_id"] = 36
    result["variable"] = "Nivel hidrométrico"
    result["unit"] = "m"

    return result


# ============================================================
# CONSULTA INA
# ============================================================

def get_series(
    start,
    end,
    series_id=SAN_NICOLAS_SERIES_ID,
):
    """
    Consulta observaciones del INA.

    Endpoint:
        https://alerta.ina.gob.ar/a5/getObservaciones

    Parámetros:
        tipo = puntual
        series_id = 36
        timestart = YYYY-MM-DD
        timeend = YYYY-MM-DD
    """

    start_str = _format_date(start)
    end_str = _format_date(end)

    start_date = pd.to_datetime(start_str)
    end_date = pd.to_datetime(end_str)

    if start_date > end_date:
        raise ValueError(
            "La fecha Desde no puede ser posterior a la fecha Hasta."
        )

    params = {
        "tipo": "puntual",
        "series_id": int(series_id),
        "timestart": start_str,
        "timeend": end_str,
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "Parana-San-Nicolas-App/1.0",
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
            f"Error de comunicación con el INA: {exc}"
        ) from exc

    # --------------------------------------------------------
    # HTTP
    # --------------------------------------------------------

    if response.status_code != 200:

        raise RuntimeError(
            f"INA respondió HTTP {response.status_code}."
        )

    if not response.text.strip():

        raise RuntimeError(
            "INA respondió correctamente pero sin contenido."
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:

        data = response.json()

    except ValueError as exc:

        preview = response.text[:300]

        raise RuntimeError(
            "INA no devolvió JSON válido. "
            f"Inicio de respuesta: {preview}"
        ) from exc

    # --------------------------------------------------------
    # POSIBLES MENSAJES DE ERROR
    # --------------------------------------------------------

    if isinstance(data, dict):

        for key in [
            "error",
            "mensaje",
            "message",
        ]:

            value = data.get(key)

            if value:

                records_test = _extract_records(data)

                if not records_test:

                    raise RuntimeError(
                        f"INA: {value}"
                    )

    records = _extract_records(data)

    if not records:
        return pd.DataFrame()

    return _normalize_observations(records)


# ============================================================
# FUNCIÓN PRINCIPAL PARA APP.PY
# ============================================================

def observed(start, end):
    """
    app.py espera:

        df, error = observed(start, end)

    Esta función SIEMPRE devuelve dos valores.
    """

    try:

        df = get_series(
            start=start,
            end=end,
            series_id=SAN_NICOLAS_SERIES_ID,
        )

        if df.empty:

            return (
                pd.DataFrame(),
                (
                    "El INA respondió correctamente, "
                    "pero no devolvió observaciones válidas "
                    "para San Nicolás, serie 36, "
                    "en el período seleccionado."
                ),
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
# DIAGNÓSTICO INA
# ============================================================

def diagnostic(start, end):
    """
    Diagnóstico para comprobar:
    - endpoint
    - serie
    - HTTP
    - registros recibidos
    """

    start_str = _format_date(start)
    end_str = _format_date(end)

    params = {
        "tipo": "puntual",
        "series_id": SAN_NICOLAS_SERIES_ID,
        "timestart": start_str,
        "timeend": end_str,
    }

    info = {
        "endpoint": INA_A5_URL,
        "series_id": SAN_NICOLAS_SERIES_ID,
        "tipo": "puntual",
        "desde": start_str,
        "hasta": end_str,
        "http_status": None,
        "registros": 0,
        "error": None,
    }

    try:

        response = requests.get(
            INA_A5_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        info["http_status"] = response.status_code

        if response.status_code != 200:

            info["error"] = (
                f"HTTP {response.status_code}"
            )

            return info

        try:

            data = response.json()

        except ValueError:

            info["error"] = "Respuesta no JSON"

            return info

        records = _extract_records(data)

        info["registros"] = len(records)

        return info

    except Exception as exc:

        info["error"] = str(exc)

        return info


# ============================================================
# METADATOS DE PRONÓSTICO
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
            "Los datos observados son obtenidos del servicio "
            "público del Instituto Nacional del Agua. "
            "El pronóstico corresponde al modelo experimental "
            "propio de la plataforma Paraná · San Nicolás."
        ),
    }
