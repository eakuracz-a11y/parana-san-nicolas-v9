import pandas as pd
import requests


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/ina.py
#
# Consulta estable mediante API pública INA
# San Nicolás: siteCode=36
# Nivel hidrométrico: varId=2
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
        "User-Agent": "Parana-San-Nicolas-App/6.0",
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
            "INA respondió sin contenido."
        )

    try:

        data = response.json()

    except ValueError as exc:

        raise RuntimeError(
            "INA no devolvió JSON válido."
        ) from exc

    return response, data, params


# ============================================================
# EXTRAER REGISTROS
# ============================================================

def _extract_records(data):

    if data is None:
        return []

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    # --------------------------------------------------------
    # MENSAJE DE ERROR DEL INA
    # --------------------------------------------------------

    mensaje = data.get("mensaje")

    if mensaje:

        raise RuntimeError(
            f"INA: {mensaje}"
        )

    # --------------------------------------------------------
    # ESTRUCTURAS COMUNES
    # --------------------------------------------------------

    possible_keys = [
        "data",
        "datos",
        "observaciones",
        "results",
        "result",
        "items",
        "values",
    ]

    for key in possible_keys:

        value = data.get(key)

        if isinstance(value, list):
            return value

        if isinstance(value, dict):

            nested = _extract_records(value)

            if nested:
                return nested

    # --------------------------------------------------------
    # BÚSQUEDA RECURSIVA
    # --------------------------------------------------------

    records = []

    def walk(obj):

        if isinstance(obj, list):

            for item in obj:

                if isinstance(item, dict):
                    records.append(item)
                else:
                    walk(item)

        elif isinstance(obj, dict):

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
# NORMALIZAR
# ============================================================

def _normalize_records(records):

    if not records:
        return pd.DataFrame()

    try:

        df = pd.json_normalize(records)

    except Exception:

        df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame()

    # ========================================================
    # FECHA
    # ========================================================

    datetime_column = _find_column(
        df,
        [
            "timestart",
            "datetime",
            "fecha",
            "date",
            "timestamp",
            "time",
            "fecha_hora",
            "fechahora",
        ],
    )

    # ========================================================
    # VALOR
    # ========================================================

    value_column = _find_column(
        df,
        [
            "valor",
            "value",
            "obsvalue",
            "obs_value",
            "nivel",
            "altura",
            "level",
        ],
    )

    # ========================================================
    # BÚSQUEDA FLEXIBLE
    # ========================================================

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

    # ========================================================
    # SI NO ENCUENTRA COLUMNAS
    # ========================================================

    if datetime_column is None:

        raise RuntimeError(
            "No se encontró la columna de fecha. "
            f"Columnas recibidas: {list(df.columns)}"
        )

    if value_column is None:

        raise RuntimeError(
            "No se encontró la columna de nivel. "
            f"Columnas recibidas: {list(df.columns)}"
        )

    # ========================================================
    # RESULTADO
    # ========================================================

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
            "El INA devolvió registros, "
            "pero no quedaron fechas y niveles válidos."
        )

    # ========================================================
    # ORDEN
    # ========================================================

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
            "INA respondió HTTP 200, "
            "pero no devolvió registros para "
            "siteCode 36 / varId 2."
        )

    return _normalize_records(records)


# ============================================================
# OBSERVED
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

        # Mantengo esta clave para que app.py no se rompa
        "series_id": "siteCode 36",

        "tipo": "nivel · varId 2",

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

        # ====================================================
        # CLAVES
        # ====================================================

        if isinstance(data, dict):

            info["json_claves"] = list(
                data.keys()
            )

        elif isinstance(data, list):

            info["json_claves"] = [
                "respuesta_raiz_lista"
            ]

        # ====================================================
        # PREVIEW
        # ====================================================

        info["respuesta_preview"] = str(
            data
        )[:2000]

        # ====================================================
        # REGISTROS
        # ====================================================

        records = _extract_records(data)

        info["registros"] = len(records)

        # ====================================================
        # PRIMER REGISTRO
        # ====================================================

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
                    f"Error: {exc}"
                ]

        return info

    except Exception as exc:

        info["error"] = str(exc)

        return info


# ============================================================
# INFORMACIÓN
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
            "Los datos observados se obtienen "
            "del servicio público del Instituto "
            "Nacional del Agua mediante la estación "
            "San Nicolás (siteCode 36) y la variable "
            "nivel hidrométrico (varId 2)."
        ),
    }
