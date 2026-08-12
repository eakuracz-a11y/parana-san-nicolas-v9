import requests
import pandas as pd


INA_BASE_URL = "https://alerta.ina.gob.ar/pub/datos"

# Estaciones utilizadas por la aplicación
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

# Serie observada del nivel del río Paraná
# San Nicolás - INA
SAN_NICOLAS_SERIES_ID = 36

# Variable 2 = nivel / altura hidrométrica
SAN_NICOLAS_VAR_ID = 2


def observed(start: str, end: str):
    """
    Obtiene datos observados del nivel del río Paraná
    para la estación San Nicolás.

    Fuente:
    Instituto Nacional del Agua (INA)
    """

    params = {
        "timeStart": start,
        "timeEnd": end,
        "seriesId": SAN_NICOLAS_SERIES_ID,
        "format": "json",
    }

    url = f"{INA_BASE_URL}/datos"

    try:

        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        # Algunas respuestas pueden venir dentro
        # de "data" o "results".
        if isinstance(data, dict):

            if "data" in data:
                data = data["data"]

            elif "results" in data:
                data = data["results"]

            else:
                data = [data]

        df = pd.DataFrame(data)

        if df.empty:
            return pd.DataFrame()

        return df

    except requests.exceptions.RequestException as exc:

        raise RuntimeError(
            f"No fue posible consultar los datos del INA: {exc}"
        ) from exc

    except ValueError as exc:

        raise RuntimeError(
            f"La respuesta del INA no pudo interpretarse: {exc}"
        ) from exc


def get_series(
    start,
    end,
    site_code=None,
    var_id=None,
    series_id=None
):
    """
    Función genérica para consultar series del INA.

    Puede utilizar:
    - seriesId
    o
    - siteCode + varId
    """

    params = {
        "timeStart": start,
        "timeEnd": end,
        "format": "json",
    }

    if series_id is not None:

        params["seriesId"] = series_id

    else:

        if site_code is None or var_id is None:

            raise ValueError(
                "Debe proporcionar series_id "
                "o site_code + var_id."
            )

        params["siteCode"] = site_code
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

        if isinstance(data, dict):

            if "data" in data:
                data = data["data"]

            elif "results" in data:
                data = data["results"]

            else:
                data = [data]

        df = pd.DataFrame(data)

        return df

    except requests.exceptions.RequestException as exc:

        raise RuntimeError(
            f"Error al consultar la API del INA: {exc}"
        ) from exc


def forecast_meta():
    """
    Información sobre el pronóstico.
    """

    return {
        "fuente": "INA",
        "servicio": "Sistema de información hidrológica",
        "estado": "Consulta disponible",
        "observacion": (
            "La aplicación utiliza datos hidrométricos "
            "observados del INA. El pronóstico de Paraná "
            "San Nicolás es generado por el modelo "
            "experimental propio."
        ),
    }
