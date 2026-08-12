import requests
import pandas as pd

INA_BASE_URL = "https://alerta.ina.gob.ar/pub/datos"


def get_series(start, end, site_code=None, var_id=None, series_id=None):
    """
    Consulta datos hidrométricos observados del INA.

    La API actual del INA utiliza:
    https://alerta.ina.gob.ar/pub/datos/datos
    """

    params = {
        "timeStart": start,
        "timeEnd": end,
        "format": "json",
    }

    if series_id:
        params["seriesId"] = series_id
    else:
        if site_code:
            params["siteCode"] = site_code

        if var_id:
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

        df = pd.DataFrame(data)

        if df.empty:
            return pd.DataFrame()

        return df

    except Exception as e:
        raise RuntimeError(
            f"No fue posible consultar el INA: {e}"
        )


def observed(start, end):
    """
    Compatibilidad con la versión anterior de Paraná San Nicolás V9.
    """

    # Serie que utilizaremos posteriormente para San Nicolás.
    return get_series(
        start=start,
        end=end
    )
