# ============================================================
# PARANÁ · SAN NICOLÁS
# src/ina.py
#
# V3 - INA robusto con A5 + fallback API pública
# ============================================================

from datetime import date, datetime

import pandas as pd
import requests


# ============================================================
# CONFIGURACIÓN
# ============================================================

INA_A5_URL = "https://alerta.ina.gob.ar/a5/getObservaciones"

INA_PUBLIC_URL = "https://alerta.ina.gob.ar/pub/datos/datos"

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
# UTILIDAD
# ============================================================

def _safe_json(response):

    if response.status_code != 200:

        raise RuntimeError(
            f"INA respondió HTTP {response.status_code}."
        )

    if not response.text.strip():

        raise RuntimeError(
            "INA respondió sin contenido."
        )

    try:

        return response.json()

    except ValueError as exc:

        preview = response.text[:500]

        raise RuntimeError(
            "INA no devolvió JSON válido. "
            f"Respuesta inicial: {preview}"
        ) from exc


# ============================================================
# EXTRACCIÓN RECURSIVA
# ============================================================

def _looks_like_record(obj):

    if not isinstance(obj, dict):
        return False

    keys = {
        str(k).lower()
        for k in obj.keys()
    }

    date_tokens = [
        "fecha",
        "date",
        "time",
        "timestamp",
        "timestart",
    ]

    value_tokens = [
        "valor",
        "value",
        "nivel",
        "altura",
        "obsvalue",
    ]

    has_date = any(
        any(token in key for token in date_tokens)
        for key in keys
    )

    has_value = any(
        any(token in key for token in value_tokens)
        for key in keys
    )

    return has_date and has_value


def _extract_records(data):
    """
    Extrae observaciones aunque el JSON venga anidado.
    """

    records = []

    def walk(obj):

        if isinstance(obj, list):

            for item in obj:

                if _looks_like_record(item):
                    records.append(item)
                else:
                    walk(item)

        elif isinstance(obj, dict):

            if _looks_like_record(obj):

                records.append(obj)
                return

            for value in obj.values():
                walk(value)

    walk(data)

    return records


# ============================================================
# BUSCAR COLUMNAS
# ============================================================

def _find_column(df, candidates):

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
# NORMALIZACIÓN
# ============================================================

def _normalize_observations(records):

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
            "nivel",
            "altura",
            "level",
        ],
    )

    # --------------------------------------------------------
    # BÚSQUEDA FLEXIBLE FECHA
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
    # BÚSQUEDA FLEXIBLE VALOR
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
        return pd.DataFrame()

    # Rango amplio de seguridad
    result = result[
        result["value"].between(
            -5,
            20,
        )
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

    result["series_id"] = SAN_NICOLAS_SERIES_ID

    result["variable"] = "Nivel hidrométrico"

    result["unit"] = "m"

    return result


# ============================================================
# CONSULTA A5
# ============================================================

def _request_a5(start, end):

    params = {
        "tipo": "puntual",
        "series_id": SAN_NICOLAS_SERIES_ID,
        "timestart": _format_date(start),
        "timeend": _format_date(end),
    }

    response = requests.get(
        INA_A5_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={
            "Accept": "application/json",
            "User-Agent": "Parana-San-Nicolas-App/3.0",
        },
    )

    data = _safe_json(response)

    return response, data, params


# ============================================================
# CONSULTA API PÚBLICA
# ============================================================

def _request_public(start, end):

    params = {
        "timeStart": _format_date(start),
        "timeEnd": _format_date(end),
        "seriesId": SAN_NICOLAS_SERIES_ID,
        "format": "json",
    }

    response = requests.get(
        INA_PUBLIC_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={
            "Accept": "application/json",
            "User-Agent": "Parana-San-Nicolas-App/3.0",
        },
    )

    data = _safe_json(response)

    if isinstance(data, dict):

        if data.get("mensaje"):

            raise RuntimeError(
                f"INA: {data.get('mensaje')}"
            )

    return response, data, params


# ============================================================
# GET SERIES
# ============================================================

def get_series(start, end):

    start_str = _format_date(start)
    end_str = _format_date(end)

    if (
        pd.to_datetime(start_str)
        > pd.to_datetime(end_str)
    ):

        raise ValueError(
            "La fecha Desde no puede ser posterior a Hasta."
        )

    # ========================================================
    # INTENTO 1 - A5
    # ========================================================

    try:

        _, data_a5, _ = _request_a5(
            start=start,
            end=end,
        )

        records_a5 = _extract_records(
            data_a5
        )

        df_a5 = _normalize_observations(
            records_a5
        )

        if not df_a5.empty:

            return df_a5

    except Exception:
        pass

    # ========================================================
    # INTENTO 2 - API PÚBLICA
    # ========================================================

    try:

        _, data_public, _ = _request_public(
            start=start,
            end=end,
        )

        records_public = _extract_records(
            data_public
        )

        df_public = _normalize_observations(
            records_public
        )

        if not df_public.empty:

            return df_public

    except Exception:
        pass

    return pd.DataFrame()


# ============================================================
# OBSERVED
# ============================================================

def observed(start, end):

    try:

        df = get_series(
            start=start,
            end=end,
        )

        if df.empty:

            return (
                pd.DataFrame(),
                (
                    "INA respondió, pero no fue posible "
                    "obtener observaciones válidas para "
                    "San Nicolás."
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
# DIAGNÓSTICO
# ============================================================

def diagnostic(start, end):

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

        "fuente_utilizada": None,
    }

    # ========================================================
    # DIAGNÓSTICO A5
    # ========================================================

    try:

        response, data, _ = _request_a5(
            start=start,
            end=end,
        )

        info["http_status"] = (
            response.status_code
        )

        info["json_tipo"] = (
            type(data).__name__
        )

        if isinstance(data, dict):

            info["json_claves"] = list(
                data.keys()
            )

        elif isinstance(data, list):

            info["json_claves"] = [
                "respuesta_raiz_lista"
            ]

        info["respuesta_preview"] = (
            str(data)[:1500]
        )

        records = _extract_records(data)

        info["registros"] = len(records)

        if records:

            info["primer_registro"] = (
                str(records[0])[:1000]
            )

            try:

                temp = pd.json_normalize(
                    records
                )

                info[
                    "columnas_detectadas"
                ] = list(temp.columns)

            except Exception:

                pass

            df = _normalize_observations(
                records
            )

            if not df.empty:

                info["fuente_utilizada"] = (
                    "INA A5"
                )

                info["registros"] = len(df)

                return info

    except Exception as exc:

        info["error"] = (
            f"A5: {exc}"
        )

    # ========================================================
    # DIAGNÓSTICO API PÚBLICA
    # ========================================================

    try:

        response, data, _ = _request_public(
            start=start,
            end=end,
        )

        records = _extract_records(data)

        df = _normalize_observations(
            records
        )

        if not df.empty:

            info["http_status"] = (
                response.status_code
            )

            info["registros"] = len(df)

            info["fuente_utilizada"] = (
                "INA API pública"
            )

            info["respuesta_preview"] = (
                str(data)[:1500]
            )

            if records:

                info["primer_registro"] = (
                    str(records[0])[:1000]
                )

            try:

                temp = pd.json_normalize(
                    records
                )

                info[
                    "columnas_detectadas"
                ] = list(temp.columns)

            except Exception:

                pass

            info["error"] = None

            return info

    except Exception as exc:

        previous_error = (
            info.get("error")
            or ""
        )

        info["error"] = (
            previous_error
            + " | API pública: "
            + str(exc)
        )

    return info


# ============================================================
# FORECAST META
# ============================================================

def forecast_meta():

    return {
        "fuente": (
            "Instituto Nacional del Agua (INA)"
        ),
        "servicio": (
            "INA A5 + API pública"
        ),
        "estacion": "San Nicolás",
        "serie": SAN_NICOLAS_SERIES_ID,
        "variable": "Nivel hidrométrico",
        "unidad": "metros",
        "estado": "Consulta online",
        "observacion": (
            "Los datos observados son obtenidos "
            "del Instituto Nacional del Agua. "
            "La plataforma intenta primero el "
            "servicio INA A5 y utiliza como respaldo "
            "la API pública de datos observados."
        ),
    }
