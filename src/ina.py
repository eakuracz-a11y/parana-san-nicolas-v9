import requests
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

INA_URL = "https://alerta.ina.gob.ar/pub/datos/datos"

SAN_NICOLAS_SITE_CODE = 36
NIVEL_VAR_ID = 2


# ============================================================
# CONSULTA AL INA
# ============================================================

def get_series(start, end):

    params = {
        "timeStart": str(start),
        "timeEnd": str(end),
        "siteCode": SAN_NICOLAS_SITE_CODE,
        "varId": NIVEL_VAR_ID,
        "format": "json",
    }

    response = requests.get(
        INA_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    # La API puede devolver lista directamente
    # o un diccionario que contiene los datos.
    if isinstance(data, dict):

        if "data" in data:
            data = data["data"]

        elif "results" in data:
            data = data["results"]

        elif "observaciones" in data:
            data = data["observaciones"]

        else:
            # Intentar detectar alguna lista dentro del JSON
            lista = None

            for value in data.values():
                if isinstance(value, list):
                    lista = value
                    break

            data = lista if lista is not None else []

    if not isinstance(data, list):
        data = []

    df = pd.DataFrame(data)

    if df.empty:
        return df

    # ========================================================
    # NORMALIZAR NOMBRES DE COLUMNAS
    # ========================================================

    rename_map = {}

    for col in df.columns:

        name = str(col).strip().lower()

        if name in [
            "fecha",
            "datetime",
            "timestamp",
            "timestart",
            "time_start",
            "date",
        ]:
            rename_map[col] = "datetime"

        elif name in [
            "valor",
            "value",
            "nivel",
            "altura",
            "valor_num",
        ]:
            rename_map[col] = "value"

    df = df.rename(columns=rename_map)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

    if "value" in df.columns:
        df["value"] = pd.to_numeric(
            df["value"],
            errors="coerce"
        )

    return df


# ============================================================
# FUNCIÓN UTILIZADA POR app.py
# ============================================================

def observed(start, end):

    try:

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # Fecha actual
        today = pd.Timestamp.today().normalize()

        # No consultar el futuro
        if end_dt > today:
            end_dt = today

        # Corregir rango inválido
        if start_dt >= end_dt:
            start_dt = end_dt - pd.Timedelta(days=30)

        df = get_series(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
        )

        if df.empty:
            return (
                pd.DataFrame(),
                "El INA respondió, pero no devolvió observaciones para San Nicolás en el período seleccionado."
            )

        if "datetime" not in df.columns:
            return (
                pd.DataFrame(),
                f"El INA devolvió datos pero no se encontró la columna de fecha. Columnas recibidas: {list(df.columns)}"
            )

        if "value" not in df.columns:
            return (
                pd.DataFrame(),
                f"El INA devolvió datos pero no se encontró la columna de nivel. Columnas recibidas: {list(df.columns)}"
            )

        df = df.dropna(
            subset=["datetime", "value"]
        )

        if df.empty:
            return (
                pd.DataFrame(),
                "El INA devolvió observaciones, pero no contienen valores válidos."
            )

        df = (
            df
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        return df, None

    except requests.exceptions.Timeout:

        return (
            pd.DataFrame(),
            "La consulta al INA excedió el tiempo máximo de espera."
        )

    except requests.exceptions.RequestException as exc:

        return (
            pd.DataFrame(),
            f"Error de conexión con el INA: {exc}"
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            f"Error procesando datos del INA: {exc}"
        )


# ============================================================
# METADATOS
# ============================================================

def forecast_meta():

    return {
        "fuente": "Instituto Nacional del Agua (INA)",
        "estacion": "San Nicolás",
        "siteCode": SAN_NICOLAS_SITE_CODE,
        "variable": "Nivel hidrométrico",
        "varId": NIVEL_VAR_ID,
        "unidad": "m",
    }
