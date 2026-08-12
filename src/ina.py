import requests
import pandas as pd


INA_BASE_URL = "https://alerta.ina.gob.ar/pub/datos"


# Serie de nivel del río Paraná en San Nicolás
# Se puede cambiar posteriormente si confirmamos otra serie.
SAN_NICOLAS_SERIES_ID = 26206


def get_series(
    start,
    end,
    site_code=None,
    var_id=None,
    series_id=SAN_NICOLAS_SERIES_ID
):
    """
    Consulta datos hidrométricos observados del INA.

    Servicio actual:
    https://alerta.ina.gob.ar/pub/datos/datos
    """

    params = {
        "timeStart": start,
        "timeEnd": end,
        "format": "json",
    }

    # Preferimos seriesId porque identifica directamente
    # la serie hidrométrica.
    if series_id:
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
            f"No fue posible consultar el INA: {exc}"
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            "El INA devolvió una respuesta que no es JSON válido."
        ) from exc

    # ---------------------------------------------------------
    # Procesamiento de la respuesta
    # ---------------------------------------------------------

    if isinstance(data, dict):

        # Respuesta de error del INA
        if "mensaje" in data:
            raise RuntimeError(
                f"INA: {data.get('mensaje')}"
            )

        if "message" in data:
            raise RuntimeError(
                f"INA: {data.get('message')}"
            )

        # Diferentes formatos posibles de respuesta
        if "data" in data:
            data = data["data"]

        elif "results" in data:
            data = data["results"]

        elif "observaciones" in data:
            data = data["observaciones"]

    # Si no hay registros
    if data is None:
        return pd.DataFrame()

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if df.empty:
        return df

    # ---------------------------------------------------------
    # Normalización de nombres de columnas
    # ---------------------------------------------------------

    rename_map = {}

    for column in df.columns:

        name = str(column).lower().strip()

        if name in [
            "fecha",
            "datetime",
            "date",
            "timestamp",
            "time",
            "timestart"
        ]:
            rename_map[column] = "datetime"

        elif name in [
            "valor",
            "value",
            "nivel",
            "altura",
            "altura_m"
        ]:
            rename_map[column] = "value"

    df = df.rename(columns=rename_map)

    # ---------------------------------------------------------
    # Intentar encontrar automáticamente la fecha
    # ---------------------------------------------------------

    if "datetime" not in df.columns:

        possible_datetime = [
            "fecha",
            "date",
            "timestamp",
            "time",
            "timestart"
        ]

        for column in possible_datetime:

            if column in df.columns:
                df["datetime"] = df[column]
                break

    # ---------------------------------------------------------
    # Intentar encontrar automáticamente el nivel
    # ---------------------------------------------------------

    if "value" not in df.columns:

        possible_value = [
            "valor",
            "value",
            "nivel",
            "altura"
        ]

        for column in possible_value:

            if column in df.columns:
                df["value"] = df[column]
                break

    # ---------------------------------------------------------
    # Convertir fecha
    # ---------------------------------------------------------

    if "datetime" in df.columns:

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

    # ---------------------------------------------------------
    # Convertir nivel a número
    # ---------------------------------------------------------

    if "value" in df.columns:

        df["value"] = pd.to_numeric(
            df["value"],
            errors="coerce"
        )

    return df


def forecast_meta():
    """
    Información sobre el pronóstico publicado por INA.
    """

    return {
        "fuente": "INA",
        "servicio": "Sistema de Información Hidrológica",
        "estado": "Consulta disponible",
        "observacion": (
            "La aplicación utiliza datos hidrométricos "
            "observados del INA. El pronóstico estadístico "
            "de Paraná San Nicolás es generado por el "
            "modelo experimental propio."
        ),
    }
