import pandas as pd
import requests


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/ina.py
#
# Consulta INA mediante API pública documentada
# ============================================================


INA_URL = "https://alerta.ina.gob.ar/pub/datos/datos"

REQUEST_TIMEOUT = 60

SAN_NICOLAS_SITE_CODE = 36
LEVEL_VAR_ID = 2


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
    parsed = pd.to_datetime(
        value,
        errors="raise",
    )

    return parsed.strftime("%Y-%m-%d")


# ============================================================
# CONSULTA HTTP
# ============================================================

def _request_ina(start, end):

    start_str = _format_date(start)
    end_str = _format_date(end)

    if pd.to_datetime(start_str) > pd.to_datetime(end_str):
        raise ValueError(
            "La fecha Desde no puede ser posterior a Hasta."
        )

    params = {
        "timeStart": start_str,
        "timeEnd": end_str,
        "siteCode": SAN_NICOLAS_SITE_CODE,
        "varId": LEVEL_VAR_ID,
        "format": "json",
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "Parana-San-Nicolas/INA-1.0",
    }

    try:
        response = requests.get(
            INA_URL,
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
            "INA respondió HTTP 200 pero sin contenido."
        )

    try:
        data = response.json()

    except ValueError as exc:
        raise RuntimeError(
            "INA no devolvió JSON válido. "
            f"Respuesta inicial: {response.text[:500]}"
        ) from exc

    return response, data, params


# ============================================================
# DETECTAR MENSAJES DE ERROR INA
# ============================================================

def _check_api_error(data):

    if not isinstance(data, dict):
        return

    mensaje = data.get("mensaje")

    if mensaje:
        raise RuntimeError(
            f"INA: {mensaje}"
        )


# ============================================================
# DETECTAR REGISTRO
# ============================================================

def _looks_like_observation(obj):

    if not isinstance(obj, dict):
        return False

    keys = {
        str(key).strip().lower()
        for key in obj.keys()
    }

    date_words = [
        "fecha",
        "date",
        "time",
        "timestamp",
        "timestart",
    ]

    value_words = [
        "valor",
        "value",
        "nivel",
        "altura",
        "obsvalue",
    ]

    has_date = any(
        any(word in key for word in date_words)
        for key in keys
    )

    has_value = any(
        any(word in key for word in value_words)
        for key in keys
    )

    return has_date and has_value


# ============================================================
# EXTRAER REGISTROS RECURSIVAMENTE
# ============================================================

def _extract_records(data):

    _check_api_error(data)

    records = []

    def walk(obj):

        if isinstance(obj, list):

            for item in obj:

                if _looks_like_observation(item):
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

def _normalize_records(records):

    if not records:
        return pd.DataFrame()

    try:
        raw = pd.json_normalize(records)

    except Exception:
        raw = pd.DataFrame(records)

    if raw.empty:
        return pd.DataFrame()

    datetime_col = _find_column(
        raw,
        [
            "timestart",
            "timeStart",
            "datetime",
            "fecha",
            "date",
            "timestamp",
            "time",
            "fecha_hora",
            "fechahora",
        ],
    )

    value_col = _find_column(
        raw,
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
    # BÚSQUEDA FLEXIBLE FECHA
    # --------------------------------------------------------

    if datetime_col is None:

        for column in raw.columns:

            name = str(column).lower()

            if (
                "fecha" in name
                or "date" in name
                or "time" in name
            ):
                datetime_col = column
                break

    # --------------------------------------------------------
    # BÚSQUEDA FLEXIBLE VALOR
    # --------------------------------------------------------

    if value_col is None:

        for column in raw.columns:

            name = str(column).lower()

            if (
                "valor" in name
                or "value" in name
                or "nivel" in name
                or "altura" in name
                or "obsvalue" in name
            ):
                value_col = column
                break

    if datetime_col is None:
        raise RuntimeError(
            "INA devolvió registros pero no se encontró "
            f"la columna temporal. Columnas: {list(raw.columns)}"
        )

    if value_col is None:
        raise RuntimeError(
            "INA devolvió registros pero no se encontró "
            f"la columna de nivel. Columnas: {list(raw.columns)}"
        )

    result = pd.DataFrame()

    result["datetime"] = pd.to_datetime(
        raw[datetime_col],
        errors="coerce",
        utc=True,
    )

    values = (
        raw[value_col]
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
        raise RuntimeError(
            "INA entregó registros, pero no quedaron "
            "fechas y niveles numéricos válidos."
        )

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

    result["station_code"] = SAN_NICOLAS_SITE_CODE

    result["variable_id"] = LEVEL_VAR_ID

    result["variable"] = "Nivel hidrométrico"

    result["unit"] = "m"

    return result


# ============================================================
# GET SERIES
# ============================================================

def get_series(start, end):

    response, data, params = _request_ina(
        start=start,
        end=end,
    )

    records = _extract_records(data)

    if not records:
        raise RuntimeError(
            "INA respondió HTTP 200, pero no se encontraron "
            "observaciones para siteCode 36 / varId 2 "
            "en el período seleccionado."
        )

    return _normalize_records(records)


# ============================================================
# FUNCIÓN QUE USA APP.PY
# ============================================================

def observed(start, end):

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

    info = {
        "endpoint": INA_URL,
        "series_id": "siteCode 36",
        "tipo": "varId 2 · nivel",
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

        "url_final": None,
    }

    try:

        response, data, params = _request_ina(
            start=start,
            end=end,
        )

        info["http_status"] = response.status_code

        info["url_final"] = response.url

        info["json_tipo"] = type(data).__name__

        if isinstance(data, dict):

            info["json_claves"] = list(
                data.keys()
            )

        elif isinstance(data, list):

            info["json_claves"] = [
                "respuesta_raiz_lista"
            ]

        info["respuesta_preview"] = str(
            data
        )[:2500]

        _check_api_error(data)

        records = _extract_records(data)

        info["registros"] = len(records)

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

        return info

    except Exception as exc:

        info["error"] = str(exc)

        return info


# ============================================================
# INFORMACIÓN DEL SERVICIO
# ============================================================

def forecast_meta():

    return {
        "fuente": "Instituto Nacional del Agua (INA)",
        "servicio": "Web API pública INA",
        "estacion": "San Nicolás",
        "serie": "siteCode 36 / varId 2",
        "variable": "Nivel hidrométrico",
        "unidad": "metros",
        "estado": "Consulta online",
        "observacion": (
            "Los datos observados se consultan mediante "
            "la Web API pública del Instituto Nacional del Agua. "
            "La predicción corresponde al modelo experimental "
            "propio de la plataforma."
        ),
    }
