# ============================================================
# PARANÁ · SAN NICOLÁS
# src/ina.py
#
# V2 - Diagnóstico detallado de respuesta INA A5
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
    Convierte una fecha a YYYY-MM-DD.
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
# EXTRAER REGISTROS
# ============================================================

def _extract_records(data):
    """
    Busca registros de observaciones dentro de la respuesta
    del INA, considerando distintas estructuras posibles.
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
        "values",
        "observations",
    ]

    for key in possible_keys:

        value = data.get(key)

        if isinstance(value, list):
            return value

        if isinstance(value, dict):

            nested = _extract_records(value)

            if nested:
                return nested

    # Si el diccionario ya parece ser un registro
    lower_keys = {
        str(key).lower()
        for key in data.keys()
    }

    possible_date_keys = {
        "fecha",
        "datetime",
        "date",
        "timestamp",
        "timestart",
        "time",
    }

    possible_value_keys = {
        "valor",
        "value",
        "obsvalue",
        "nivel",
        "altura",
        "level",
    }

    if (
        lower_keys.intersection(possible_date_keys)
        and lower_keys.intersection(possible_value_keys)
    ):
        return [data]

    return []


# ============================================================
# BUSCAR COLUMNA
# ============================================================

def _find_column(df, candidates):
    """
    Busca columnas ignorando mayúsculas y minúsculas.
    """

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        candidate_lower = candidate.lower()

        if candidate_lower in lookup:
            return lookup[candidate_lower]

    return None


# ============================================================
# NORMALIZAR DATOS
# ============================================================

def _normalize_observations(records):
    """
    Convierte los registros del INA al formato esperado
    por app.py.
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
            "fechaHora",
            "fecha_hora",
            "inicio",
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
            "nivel",
            "altura",
            "level",
            "valorNumerico",
            "valor_numerico",
        ],
    )

    # --------------------------------------------------------
    # BUSQUEDA FLEXIBLE DE FECHA
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
    # BUSQUEDA FLEXIBLE DE VALOR
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
        subset=[
            "datetime",
            "value",
        ]
    ).copy()

    if result.empty:
        return pd.DataFrame()

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

def _request_ina(start, end, series_id):
    """
    Ejecuta la consulta al INA y devuelve:
    response, json, error
    """

    start_str = _format_date(start)
    end_str = _format_date(end)

    params = {
        "tipo": "puntual",
        "series_id": int(series_id),
        "timestart": start_str,
        "timeend": end_str,
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "Parana-San-Nicolas-App/2.0",
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
            "No fue posible conectarse con el INA."
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
# GET SERIES
# ============================================================

def get_series(
    start,
    end,
    series_id=SAN_NICOLAS_SERIES_ID,
):
    """
    Devuelve observaciones normalizadas.
    """

    start_str = _format_date(start)
    end_str = _format_date(end)

    if pd.to_datetime(start_str) > pd.to_datetime(end_str):

        raise ValueError(
            "La fecha Desde no puede ser posterior a Hasta."
        )

    response, data, params = _request_ina(
        start=start,
        end=end,
        series_id=series_id,
    )

    records = _extract_records(data)

    if not records:
        return pd.DataFrame()

    return _normalize_observations(records)


# ============================================================
# OBSERVED
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
            series_id=SAN_NICOLAS_SERIES_ID,
        )

        if df.empty:

            return (
                pd.DataFrame(),
                (
                    "El INA respondió correctamente, "
                    "pero no se pudieron interpretar "
                    "observaciones válidas para San Nicolás."
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
# DIAGNÓSTICO DETALLADO
# ============================================================

def diagnostic(start, end):
    """
    Diagnóstico completo de la respuesta del INA.
    """

    start_str = _format_date(start)
    end_str = _format_date(end)

    info = {
        "endpoint": INA_A5_URL,
        "series_id": SAN_NICOLAS_SERIES_ID,
        "tipo": "puntual",
        "desde": start_str,
        "hasta": end_str,
        "http_status": None,
        "registros": 0,
        "error": None,

        # Nuevos campos de diagnóstico
        "json_tipo": None,
        "json_claves": None,
        "primer_registro": None,
        "columnas_detectadas": None,
        "respuesta_preview": None,
    }

    try:

        response, data, params = _request_ina(
            start=start,
            end=end,
            series_id=SAN_NICOLAS_SERIES_ID,
        )

        info["http_status"] = response.status_code

        # ----------------------------------------------------
        # TIPO DE JSON
        # ----------------------------------------------------

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
                "Respuesta raíz tipo lista"
            ]

        # ----------------------------------------------------
        # PREVIEW DE RESPUESTA
        # ----------------------------------------------------

        info["respuesta_preview"] = str(data)[:1000]

        # ----------------------------------------------------
        # EXTRAER REGISTROS
        # ----------------------------------------------------

        records = _extract_records(data)

        info["registros"] = len(records)

        # ----------------------------------------------------
        # PRIMER REGISTRO
        # ----------------------------------------------------

        if records:

            info["primer_registro"] = str(
                records[0]
            )[:1000]

            try:

                temp_df = pd.json_normalize(
                    records
                )

                info["columnas_detectadas"] = list(
                    temp_df.columns
                )

            except Exception:

                try:

                    temp_df = pd.DataFrame(
                        records
                    )

                    info["columnas_detectadas"] = list(
                        temp_df.columns
                    )

                except Exception:

                    info["columnas_detectadas"] = [
                        "No se pudieron detectar"
                    ]

        return info

    except Exception as exc:

        info["error"] = str(exc)

        return info


# ============================================================
# FORECAST META
# ============================================================

def forecast_meta():

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
