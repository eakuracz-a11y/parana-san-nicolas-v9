import requests
import pandas as pd


INA_BASE_URL = "https://alerta.ina.gob.ar/pub/datos"


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


def get_series(
    start,
    end,
    site_code=None,
    var_id=None,
    series_id=None
):
    """
    Consulta datos hidrométricos observados del INA.

    La API pública actual del INA utiliza:

    https://alerta.ina.gob.ar/pub/datos/datos

    Parámetros:
    - timeStart
    - timeEnd
    - seriesId
    o:
    - siteCode
    - varId
    """

    params = {
        "timeStart": start,
        "timeEnd": end,
        "format": "json",
    }

    # La API permite utilizar seriesId
    # o siteCode + varId.
    if series_id is not None:
        params["seriesId"] = series_id

    else:
        if site_code is None or var_id is None:
            raise ValueError(
                "Debe proporcionar series_id o "
                "site_code + var_id."
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

        # Algunas respuestas del INA pueden venir
        # encapsuladas dentro de "data" o "results".
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
            f"Error al consultar la API del INA: {exc}"
        ) from exc

    except ValueError as exc:

        raise RuntimeError(
            f"El INA devolvió una respuesta que "
            f"no pudo interpretarse: {exc}"
        ) from exc


def forecast_meta():
    """
    Información sobre el pronóstico publicado por INA.
    """

    return {
        "fuente": "INA",
        "servicio": "Sistema de información hidrológica",
        "estado": "Consulta disponible",
        "observacion": (
            "La aplicación utiliza datos hidrométricos "
            "observados del INA. El pronóstico estadístico "
            "de Paraná San Nicolás es generado por el "
            "modelo experimental propio."
        ),
    }
