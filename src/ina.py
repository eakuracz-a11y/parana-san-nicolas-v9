import requests
import pandas as pd


# ============================================================
# CONFIGURACIÓN DEL INA
# ============================================================

INA_URL = "https://alerta.ina.gob.ar"

VAR_ID = 2


# ============================================================
# ESTACIONES UTILIZADAS POR PARANÁ SAN NICOLÁS V9
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
# OBTENER DATOS OBSERVADOS
# ============================================================

def observed(start: str, end: str):
    """
    Obtiene datos hidrométricos observados del INA.

    Consulta la API pública:
        /pub/datos/datos

    Devuelve:

        result
        errors

    result contiene una columna datetime y una columna
    para cada estación.
    """

    errors = []
    frames = []

    start = str(start).replace("/", "-")
    end = str(end).replace("/", "-")

    for station, site_code in STATION_CODES.items():

        url = f"{INA_URL}/pub/datos/datos"

        params = {
            "timeStart": start,
            "timeEnd": end,
            "siteCode": site_code,
            "varId": VAR_ID,
            "format": "json",
        }

        try:

            response = requests.get(
                url,
                params=params,
                timeout=30,
            )

            response.raise_for_status()

            data = response.json()

            df = _normalize_ina_data(
                data,
                station,
            )

            if df is not None and not df.empty:
                frames.append(df)

        except Exception as exc:

            errors.append(
                f"{station}: {exc}"
            )


    # ========================================================
    # SI NINGUNA ESTACIÓN DEVOLVIÓ DATOS
    # ========================================================

    if not frames:

        raise RuntimeError(
            "No fue posible obtener datos observados del INA. "
            + (
                " | ".join(errors)
                if errors
                else "La API no devolvió datos."
            )
        )


    # ========================================================
    # UNIR LAS ESTACIONES
    # ========================================================

    result = frames[0]

    for frame in frames[1:]:

        result = pd.merge(
            result,
            frame,
            on="datetime",
            how="outer",
        )


    result = result.sort_values(
        "datetime"
    )


    return result, errors


# ============================================================
# NORMALIZAR RESPUESTA DEL INA
# ============================================================

def _normalize_ina_data(data, station):
    """
    Convierte la respuesta JSON del INA a:

        datetime | station | value
    """

    if isinstance(data, dict):

        for key in (
            "data",
            "datos",
            "results",
            "result",
            "observations",
        ):

            if key in data:

                data = data[key]

                break


    if isinstance(data, dict):

        data = [data]


    if not isinstance(data, list):

        return None


    rows = []


    for item in data:

        if not isinstance(item, dict):

            continue


        # ----------------------------------------------------
        # FECHA / HORA
        # ----------------------------------------------------

        dt = (
            item.get("fecha")
            or item.get("datetime")
            or item.get("date")
            or item.get("timestamp")
            or item.get("time")
        )


        # ----------------------------------------------------
        # VALOR HIDROMÉTRICO
        # ----------------------------------------------------

        value = item.get("valor")

        if value is None:

            value = item.get("value")

        if value is None:

            value = item.get("nivel")


        if dt is None or value is None:

            continue


        rows.append(
            {
                "datetime": dt,
                "station": station,
                "value": value,
            }
        )


    if not rows:

        return None


    df = pd.DataFrame(rows)

    return _pivot(df)


# ============================================================
# CONVERTIR A TABLA POR ESTACIÓN
# ============================================================

def _pivot(df):

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
        utc=True,
    )


    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )


    df = df.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )


    if df.empty:

        return None


    out = df.pivot_table(
        index="datetime",
        columns="station",
        values="value",
        aggfunc="last",
    ).reset_index()


    out.columns.name = None


    return out


# ============================================================
# INFORMACIÓN DE LA FUENTE
# ============================================================

def forecast_meta():
    """
    Información básica sobre la fuente de datos.

    Esta función es utilizada por app.py.
    """

    return {
        "fuente": "INA",
        "url": f"{INA_URL}/pub/datos/datos",
        "mensaje": (
            "Datos hidrométricos observados "
            "obtenidos desde la API pública "
            "del INA."
        ),
    }
