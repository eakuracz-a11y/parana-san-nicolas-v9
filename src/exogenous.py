# ============================================================
# PARANÁ · SAN NICOLÁS
# src/exogenous.py
# V11.10 COMPLETO
#
# MODELO HIDROLÓGICO MULTIESTACIÓN
#
# - Lluvia histórica por estación
# - Pronóstico de lluvia por estación
# - Caudal INA A5 por estación
# - Validación real de series mediante getObservaciones
# - Mantiene precip_mm y caudal_m3s por compatibilidad
# - Genera q_* y rain_* por estación
# - Compatible con horizonte 15 / 30 / 45 / 60 días
# ============================================================


from functools import lru_cache
import unicodedata

import numpy as np
import pandas as pd
import requests


VERSION = "V11.10"


# ============================================================
# ENDPOINTS
# ============================================================

FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

HISTORICAL_WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

INA_A5_BASE_URL = (
    "https://alerta.ina.gob.ar/a5"
)

INA_SERIES_GEOJSON_URL = (
    INA_A5_BASE_URL
    + "/obs/puntual/series"
)

INA_OBSERVATIONS_URL = (
    INA_A5_BASE_URL
    + "/getObservaciones"
)


# ============================================================
# CONFIG
# ============================================================

VAR_ID_CAUDAL = 4

REQUEST_TIMEOUT = 45

MAX_FORECAST_DAYS = 60

CAUDAL_MIN = 0.0
CAUDAL_MAX = 200000.0


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


RAIN_POINTS = {

    "Corrientes": {
        "latitude": -27.4692,
        "longitude": -58.8306,
    },

    "Goya": {
        "latitude": -29.1400,
        "longitude": -59.2634,
    },

    "La Paz": {
        "latitude": -30.7449,
        "longitude": -59.6457,
    },

    "Paraná": {
        "latitude": -31.7319,
        "longitude": -60.5238,
    },

    "Diamante": {
        "latitude": -32.0664,
        "longitude": -60.6384,
    },

    "Rosario": {
        "latitude": -32.9442,
        "longitude": -60.6505,
    },

    "Villa Constitución": {
        "latitude": -33.2272,
        "longitude": -60.3296,
    },

    "San Nicolás": {
        "latitude": -33.3358,
        "longitude": -60.2252,
    },
}


# ============================================================
# REQUEST SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
            "Parana-San-Nicolas/11.10",

        "Accept":
            "application/json",
    }
)


# ============================================================
# UTILIDADES
# ============================================================

def normalizar_texto(value):

    if value is None:
        return ""

    text = str(value).strip().lower()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        c
        for c in text
        if not unicodedata.combining(c)
    )

    return " ".join(
        text.split()
    )


def slug_estacion(name):

    return (
        normalizar_texto(name)
        .replace(" ", "_")
    )


def normalizar_fecha(value):

    dt = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(dt):

        raise ValueError(
            f"Fecha inválida: {value}"
        )

    return dt.strftime(
        "%Y-%m-%d"
    )


def datetime_naive(values):

    return (
        pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
    )


def safe_int(value, default=None):

    try:

        if value is None:
            return default

        return int(float(value))

    except Exception:

        return default


def safe_float(value, default=np.nan):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def request_json(
    url,
    params=None,
):

    response = SESSION.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# LLUVIA HISTÓRICA
# ============================================================

def lluvia_historica_punto(
    latitude,
    longitude,
    start,
    end,
):

    params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "start_date":
            normalizar_fecha(start),

        "end_date":
            normalizar_fecha(end),

        "daily":
            "precipitation_sum",

        "timezone":
            "America/Argentina/Buenos_Aires",
    }

    try:

        data = request_json(
            HISTORICAL_WEATHER_URL,
            params=params,
        )

    except Exception:

        return pd.DataFrame(
            columns=[
                "datetime",
                "rain",
            ]
        )


    daily = data.get(
        "daily",
        {},
    )

    dates = daily.get(
        "time",
        [],
    )

    values = daily.get(
        "precipitation_sum",
        [],
    )


    if not dates:

        return pd.DataFrame(
            columns=[
                "datetime",
                "rain",
            ]
        )


    df = pd.DataFrame(
        {
            "datetime":
                dates,

            "rain":
                values,
        }
    )


    df["datetime"] = datetime_naive(
        df["datetime"]
    )


    df["rain"] = (
        pd.to_numeric(
            df["rain"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )


    return (
        df
        .dropna(
            subset=["datetime"]
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def get_rain_history(
    start,
    end,
):

    base = pd.DataFrame(
        {
            "datetime":
                pd.date_range(
                    start=pd.to_datetime(start),
                    end=pd.to_datetime(end),
                    freq="D",
                )
        }
    )


    rain_columns = []


    for station in STATIONS:

        coords = RAIN_POINTS[
            station
        ]

        frame = lluvia_historica_punto(
            coords["latitude"],
            coords["longitude"],
            start,
            end,
        )

        column = (
            "rain_"
            + slug_estacion(station)
        )

        rain_columns.append(
            column
        )


        if frame.empty:

            base[column] = np.nan

            continue


        frame = frame.rename(
            columns={
                "rain":
                    column
            }
        )


        base = base.merge(
            frame,
            on="datetime",
            how="left",
        )


    for col in rain_columns:

        if col not in base.columns:
            base[col] = np.nan

        base[col] = (
            pd.to_numeric(
                base[col],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
        )


    # promedio del corredor
    # se mantiene por compatibilidad

    base["precip_mm"] = (
        base[
            rain_columns
        ]
        .mean(
            axis=1,
            skipna=True,
        )
        .fillna(0.0)
    )


    return base


# ============================================================
# LLUVIA FUTURA
# ============================================================

def lluvia_futura_punto(
    latitude,
    longitude,
):

    params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "daily":
            "precipitation_sum",

        "forecast_days":
            16,

        "timezone":
            "America/Argentina/Buenos_Aires",
    }


    try:

        data = request_json(
            FORECAST_URL,
            params=params,
        )

    except Exception:

        return pd.DataFrame(
            columns=[
                "datetime",
                "rain",
            ]
        )


    daily = data.get(
        "daily",
        {},
    )


    df = pd.DataFrame(
        {
            "datetime":
                daily.get(
                    "time",
                    [],
                ),

            "rain":
                daily.get(
                    "precipitation_sum",
                    [],
                ),
        }
    )


    if df.empty:

        return df


    df["datetime"] = datetime_naive(
        df["datetime"]
    )


    df["rain"] = (
        pd.to_numeric(
            df["rain"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )


    return df


def get_rain_forecast(
    start_future,
    days=60,
):

    days = min(
        max(int(days), 1),
        MAX_FORECAST_DAYS,
    )


    future = pd.DataFrame(
        {
            "datetime":
                pd.date_range(
                    start=pd.to_datetime(
                        start_future
                    ),
                    periods=days,
                    freq="D",
                )
        }
    )


    rain_columns = []


    for station in STATIONS:

        coords = RAIN_POINTS[
            station
        ]


        frame = lluvia_futura_punto(
            coords["latitude"],
            coords["longitude"],
        )


        column = (
            "rain_"
            + slug_estacion(station)
        )


        rain_columns.append(
            column
        )


        if frame.empty:

            future[column] = np.nan

            continue


        frame = frame.rename(
            columns={
                "rain":
                    column
            }
        )


        future = future.merge(
            frame,
            on="datetime",
            how="left",
        )


    # --------------------------------------------------------
    # Después del horizonte meteorológico real:
    # no inventamos lluvia.
    # --------------------------------------------------------

    for col in rain_columns:

        if col not in future.columns:
            future[col] = 0.0

        future[col] = (
            pd.to_numeric(
                future[col],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
        )


    future["precip_mm"] = (
        future[
            rain_columns
        ]
        .mean(
            axis=1,
            skipna=True,
        )
        .fillna(0.0)
    )


    return future


# ============================================================
# CATÁLOGO INA
# ============================================================

@lru_cache(maxsize=1)
def get_ina_catalog():

    try:

        data = request_json(
            INA_SERIES_GEOJSON_URL,
            params={
                "format":
                    "geojson"
            },
        )

    except Exception:

        return pd.DataFrame()


    features = data.get(
        "features",
        [],
    )


    rows = []


    for feature in features:

        if not isinstance(
            feature,
            dict,
        ):
            continue


        props = dict(
            feature.get(
                "properties",
                {},
            )
        )


        if not props:
            continue


        if props.get(
            "series_id"
        ) is None:

            props["series_id"] = (
                feature.get("id")
            )


        geometry = feature.get(
            "geometry",
            {},
        )


        coords = geometry.get(
            "coordinates",
            [],
        )


        if len(coords) >= 2:

            props["longitude"] = (
                coords[0]
            )

            props["latitude"] = (
                coords[1]
            )


        rows.append(
            props
        )


    if not rows:

        return pd.DataFrame()


    df = pd.DataFrame(
        rows
    )


    for col in [
        "series_id",
        "var_id",
        "proc_id",
        "unit_id",
        "count",
    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )


    for col in [
        "timestart",
        "timeend",
    ]:

        if col in df.columns:

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
                utc=True,
            )


    if "nombre" not in df.columns:
        df["nombre"] = ""

    if "rio" not in df.columns:
        df["rio"] = ""


    df["_name"] = (
        df["nombre"]
        .apply(normalizar_texto)
    )


    df["_river"] = (
        df["rio"]
        .apply(normalizar_texto)
    )


    return df


# ============================================================
# PARSER INA
# ============================================================

def extract_records(data):

    if isinstance(data, list):

        if (
            data
            and isinstance(
                data[0],
                dict,
            )
        ):

            sample_keys = set()

            for row in data[:5]:
                sample_keys.update(
                    row.keys()
                )

            if sample_keys.intersection(
                {
                    "timestart",
                    "datetime",
                    "fecha",
                    "valor",
                    "value",
                }
            ):

                return data


        for item in data:

            records = extract_records(
                item
            )

            if records:
                return records


    elif isinstance(data, dict):

        preferred = [
            "observaciones",
            "observations",
            "datos",
            "data",
            "records",
            "values",
            "result",
        ]


        for key in preferred:

            if key in data:

                records = extract_records(
                    data[key]
                )

                if records:
                    return records


        for value in data.values():

            records = extract_records(
                value
            )

            if records:
                return records


    return []


def normalizar_caudal(
    data,
):

    records = extract_records(
        data
    )


    rows = []


    date_fields = [
        "timestart",
        "datetime",
        "timestamp",
        "fecha",
        "date",
        "time",
        "obs_date",
    ]


    value_fields = [
        "valor",
        "value",
        "caudal",
        "obs_value",
        "valor_num",
    ]


    for record in records:

        if not isinstance(
            record,
            dict,
        ):
            continue


        dt = None
        value = None


        for field in date_fields:

            if record.get(
                field
            ) is not None:

                dt = record[
                    field
                ]

                break


        for field in value_fields:

            if record.get(
                field
            ) is not None:

                value = record[
                    field
                ]

                break


        dt = pd.to_datetime(
            dt,
            errors="coerce",
            utc=True,
        )


        value = safe_float(
            value
        )


        if pd.isna(dt):
            continue


        if not np.isfinite(
            value
        ):
            continue


        if not (
            CAUDAL_MIN
            <= value
            <= CAUDAL_MAX
        ):

            continue


        rows.append(
            {
                "datetime":
                    dt,

                "caudal":
                    value,
            }
        )


    if not rows:

        return pd.DataFrame(
            columns=[
                "datetime",
                "caudal",
            ]
        )


    df = pd.DataFrame(
        rows
    )


    df["datetime"] = (
        datetime_naive(
            df["datetime"]
        )
        .dt
        .normalize()
    )


    return (
        df
        .groupby(
            "datetime",
            as_index=False,
        )["caudal"]
        .median()
        .sort_values("datetime")
        .reset_index(drop=True)
    )


# ============================================================
# CONSULTA INA
# ============================================================

def query_caudal_series(
    series_id,
    start,
    end,
):

    params = {

        "tipo":
            "puntual",

        "series_id":
            int(series_id),

        "timestart":
            normalizar_fecha(start),

        "timeend":
            normalizar_fecha(end),
    }


    data = request_json(
        INA_OBSERVATIONS_URL,
        params=params,
    )


    return normalizar_caudal(
        data
    )


# ============================================================
# SCORE ESTACIÓN
# ============================================================

def station_match(
    station,
    name,
):

    station_norm = normalizar_texto(
        station
    )

    name_norm = normalizar_texto(
        name
    )


    if name_norm == station_norm:
        return 1000


    if name_norm.startswith(
        station_norm
    ):
        return 900


    if (
        f" {station_norm} "
        in f" {name_norm} "
    ):
        return 850


    if station_norm in name_norm:
        return 750


    return 0


# ============================================================
# CANDIDATOS POR ESTACIÓN
# ============================================================

def candidatos_caudal_estacion(
    station,
    start,
    end,
):

    catalog = get_ina_catalog()


    if catalog.empty:

        return pd.DataFrame()


    if "var_id" not in catalog.columns:

        return pd.DataFrame()


    df = catalog[
        catalog["var_id"]
        == VAR_ID_CAUDAL
    ].copy()


    if df.empty:
        return df


    rows = []


    requested_start = pd.to_datetime(
        start,
        errors="coerce",
        utc=True,
    )

    requested_end = pd.to_datetime(
        end,
        errors="coerce",
        utc=True,
    )


    for _, row in df.iterrows():

        name = str(
            row.get(
                "nombre",
                "",
            )
        )


        score_station = station_match(
            station,
            name,
        )


        if score_station <= 0:
            continue


        river = normalizar_texto(
            row.get(
                "rio",
                "",
            )
        )


        score = score_station


        # río Paraná
        if "parana" in river:
            score += 500


        proc = safe_int(
            row.get("proc_id"),
            -1,
        )


        if proc == 1:
            score += 100

        elif proc == 2:
            score += 60


        count = safe_int(
            row.get("count"),
            0,
        )


        if count > 10000:
            score += 100

        elif count > 1000:
            score += 70

        elif count > 100:
            score += 40


        time_start = row.get(
            "timestart"
        )

        time_end = row.get(
            "timeend"
        )


        overlap = True


        if (
            pd.notna(time_end)
            and pd.notna(requested_start)
            and time_end
            < requested_start
        ):
            overlap = False


        if (
            pd.notna(time_start)
            and pd.notna(requested_end)
            and time_start
            > requested_end
        ):
            overlap = False


        if overlap:
            score += 150
        else:
            score -= 500


        rows.append(
            {
                "series_id":
                    safe_int(
                        row.get(
                            "series_id"
                        )
                    ),

                "station":
                    station,

                "series_name":
                    name,

                "river":
                    row.get(
                        "rio"
                    ),

                "proc_id":
                    proc,

                "count":
                    count,

                "timestart":
                    time_start,

                "timeend":
                    time_end,

                "score":
                    score,
            }
        )


    if not rows:

        return pd.DataFrame()


    return (
        pd.DataFrame(rows)
        .dropna(
            subset=["series_id"]
        )
        .sort_values(
            "score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# VALIDAR SERIE PARA UNA ESTACIÓN
# ============================================================

def seleccionar_serie_caudal(
    station,
    start,
    end,
):

    candidates = (
        candidatos_caudal_estacion(
            station,
            start,
            end,
        )
    )


    if candidates.empty:

        return None


    req_start = pd.to_datetime(
        start
    )

    req_end = pd.to_datetime(
        end
    )


    for index, row in (
        candidates
        .head(20)
        .iterrows()
    ):

        series_id = int(
            row["series_id"]
        )


        cat_start = row.get(
            "timestart"
        )

        cat_end = row.get(
            "timeend"
        )


        test_end = req_end


        if pd.notna(cat_end):

            cat_end_naive = (
                cat_end
                .tz_localize(None)
            )

            test_end = min(
                test_end,
                cat_end_naive,
            )


        test_start = max(
            req_start,
            test_end
            - pd.Timedelta(
                days=180
            ),
        )


        if pd.notna(cat_start):

            cat_start_naive = (
                cat_start
                .tz_localize(None)
            )

            test_start = max(
                test_start,
                cat_start_naive,
            )


        if test_start > test_end:
            continue


        try:

            test = query_caudal_series(
                series_id,
                test_start,
                test_end,
            )

        except Exception:

            continue


        if test.empty:
            continue


        valid = (
            pd.to_numeric(
                test["caudal"],
                errors="coerce",
            )
            .dropna()
        )


        if len(valid) < 3:
            continue


        return {
            "station":
                station,

            "series_id":
                series_id,

            "series_name":
                row.get(
                    "series_name"
                ),

            "river":
                row.get(
                    "river"
                ),

            "proc_id":
                safe_int(
                    row.get(
                        "proc_id"
                    )
                ),

            "records_validation":
                len(valid),

            "last_validation_flow":
                float(
                    valid.iloc[-1]
                ),

            "candidate_number":
                index + 1,
        }


    return None


# ============================================================
# CAUDAL HISTÓRICO POR ESTACIÓN
# ============================================================

def get_caudal_station(
    station,
    start,
    end,
):

    info = seleccionar_serie_caudal(
        station,
        start,
        end,
    )


    if info is None:

        return (
            pd.DataFrame(),
            {
                "station":
                    station,

                "status":
                    "sin_serie",
            },
        )


    try:

        df = query_caudal_series(
            info["series_id"],
            start,
            end,
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            {
                **info,

                "status":
                    "error",

                "error":
                    str(exc),
            },
        )


    if df.empty:

        return (
            pd.DataFrame(),
            {
                **info,

                "status":
                    "sin_datos",
            },
        )


    column = (
        "q_"
        + slug_estacion(station)
    )


    df = df.rename(
        columns={
            "caudal":
                column
        }
    )


    meta = {
        **info,

        "status":
            "ok",

        "records":
            int(len(df)),

        "first_date":
            str(
                df["datetime"]
                .min()
            ),

        "last_date":
            str(
                df["datetime"]
                .max()
            ),

        "current_flow":
            float(
                df[column]
                .dropna()
                .iloc[-1]
            ),
    }


    return (
        df,
        meta,
    )


# ============================================================
# TODOS LOS CAUDALES
# ============================================================

def get_all_caudales(
    start,
    end,
):

    base = pd.DataFrame(
        {
            "datetime":
                pd.date_range(
                    start=pd.to_datetime(start),
                    end=pd.to_datetime(end),
                    freq="D",
                )
        }
    )


    metadata = {}


    for station in STATIONS:

        column = (
            "q_"
            + slug_estacion(station)
        )


        try:

            frame, meta = (
                get_caudal_station(
                    station,
                    start,
                    end,
                )
            )

        except Exception as exc:

            frame = pd.DataFrame()

            meta = {
                "station":
                    station,

                "status":
                    "error",

                "error":
                    str(exc),
            }


        metadata[
            station
        ] = meta


        if frame.empty:

            base[column] = np.nan

        else:

            base = base.merge(
                frame,
                on="datetime",
                how="left",
            )


    return (
        base,
        metadata,
    )


# ============================================================
# DETERMINAR CAUDAL PRINCIPAL
# ============================================================

def elegir_caudal_principal(
    df,
):

    priority = [
        "q_san_nicolas",
        "q_villa_constitucion",
        "q_rosario",
        "q_diamante",
        "q_parana",
        "q_la_paz",
        "q_goya",
        "q_corrientes",
    ]


    for column in priority:

        if (
            column
            in df.columns
            and df[column]
            .notna()
            .sum()
            >= 3
        ):

            return (
                column,
                df[column],
            )


    return (
        None,
        pd.Series(
            np.nan,
            index=df.index,
        ),
    )


# ============================================================
# PROYECCIÓN DE UNA SERIE DE CAUDAL
# ============================================================

def proyectar_serie_caudal(
    series,
    days,
):

    values = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
        .tail(21)
    )


    if values.empty:

        return np.repeat(
            np.nan,
            days,
        )


    current = float(
        values.iloc[-1]
    )


    if len(values) >= 5:

        x = np.arange(
            len(values),
            dtype=float,
        )

        try:

            slope = float(
                np.polyfit(
                    x,
                    values.values,
                    1,
                )[0]
            )

        except Exception:

            slope = 0.0

    else:

        slope = 0.0


    max_change = max(
        abs(current)
        * 0.025,
        50.0,
    )


    slope = np.clip(
        slope,
        -max_change,
        max_change,
    )


    output = []

    value = current


    for step in range(
        1,
        days + 1,
    ):

        damping = np.exp(
            -step / 18.0
        )


        change = (
            slope
            * damping
        )


        value = max(
            0.0,
            value + change,
        )


        output.append(
            value
        )


    return np.array(
        output,
        dtype=float,
    )


# ============================================================
# FEATURES DE LLUVIA
# ============================================================

def agregar_features_lluvia(
    df,
):

    result = df.copy()


    rain_cols = [
        c
        for c in result.columns
        if c.startswith(
            "rain_"
        )
    ]


    for col in rain_cols:

        values = (
            pd.to_numeric(
                result[col],
                errors="coerce",
            )
            .fillna(0.0)
        )


        result[
            col + "_3d"
        ] = (
            values
            .rolling(
                3,
                min_periods=1,
            )
            .sum()
        )


        result[
            col + "_7d"
        ] = (
            values
            .rolling(
                7,
                min_periods=1,
            )
            .sum()
        )


        result[
            col + "_15d"
        ] = (
            values
            .rolling(
                15,
                min_periods=1,
            )
            .sum()
        )


        result[
            col + "_30d"
        ] = (
            values
            .rolling(
                30,
                min_periods=1,
            )
            .sum()
        )


    return result


# ============================================================
# FEATURES DE CAUDAL
# ============================================================

def agregar_features_caudal(
    df,
):

    result = df.copy()


    q_cols = [
        c
        for c in result.columns
        if c.startswith(
            "q_"
        )
        and not c.endswith(
            (
                "_diff_1d",
                "_diff_3d",
                "_diff_7d",
                "_mean_3d",
                "_mean_7d",
                "_mean_14d",
            )
        )
    ]


    for col in q_cols:

        values = pd.to_numeric(
            result[col],
            errors="coerce",
        )


        result[
            col + "_diff_1d"
        ] = values.diff(1)


        result[
            col + "_diff_3d"
        ] = values.diff(3)


        result[
            col + "_diff_7d"
        ] = values.diff(7)


        result[
            col + "_mean_3d"
        ] = (
            values
            .rolling(
                3,
                min_periods=1,
            )
            .mean()
        )


        result[
            col + "_mean_7d"
        ] = (
            values
            .rolling(
                7,
                min_periods=1,
            )
            .mean()
        )


        result[
            col + "_mean_14d"
        ] = (
            values
            .rolling(
                14,
                min_periods=1,
            )
            .mean()
        )


    return result


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def get_exogenous_data(
    start,
    end,
    forecast_days=15,
):

    forecast_days = min(
        max(
            int(forecast_days),
            1,
        ),
        MAX_FORECAST_DAYS,
    )


    # ========================================================
    # LLUVIA HISTÓRICA
    # ========================================================

    rain_history = (
        get_rain_history(
            start,
            end,
        )
    )


    # ========================================================
    # CAUDALES
    # ========================================================

    (
        flow_history,
        flow_meta,
    ) = get_all_caudales(
        start,
        end,
    )


    # ========================================================
    # MERGE HISTORY
    # ========================================================

    history = rain_history.merge(
        flow_history,
        on="datetime",
        how="outer",
    )


    history["datetime"] = (
        datetime_naive(
            history["datetime"]
        )
    )


    history = (
        history
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"]
        )
        .reset_index(drop=True)
    )


    # ========================================================
    # CAUDAL PRINCIPAL LEGACY
    # ========================================================

    (
        primary_q_column,
        primary_q,
    ) = elegir_caudal_principal(
        history
    )


    history[
        "caudal_m3s"
    ] = primary_q


    # ========================================================
    # INTERPOLACIÓN LIMITADA
    # ========================================================

    q_columns = [
        c
        for c in history.columns
        if c.startswith("q_")
    ]


    for col in q_columns:

        history[col] = pd.to_numeric(
            history[col],
            errors="coerce",
        )


        if history[col].notna().sum() >= 5:

            history[col] = (
                history[col]
                .interpolate(
                    limit=5,
                    limit_direction="both",
                )
            )


    history = agregar_features_lluvia(
        history
    )


    history = agregar_features_caudal(
        history
    )


    # ========================================================
    # FUTURO
    # ========================================================

    future_start = (
        pd.to_datetime(end)
        + pd.Timedelta(days=1)
    )


    future = get_rain_forecast(
        future_start,
        forecast_days,
    )


    # ========================================================
    # PROYECTAR CADA CAUDAL
    # ========================================================

    for station in STATIONS:

        q_col = (
            "q_"
            + slug_estacion(station)
        )


        if q_col not in history.columns:

            future[q_col] = np.nan

            continue


        future[q_col] = (
            proyectar_serie_caudal(
                history[q_col],
                forecast_days,
            )
        )


    (
        future_primary_col,
        future_primary_q,
    ) = elegir_caudal_principal(
        future
    )


    future[
        "caudal_m3s"
    ] = future_primary_q


    future = agregar_features_lluvia(
        future
    )


    future = agregar_features_caudal(
        future
    )


    # ========================================================
    # METADATA
    # ========================================================

    stations_with_flow = []


    for station in STATIONS:

        meta_station = (
            flow_meta.get(
                station,
                {}
            )
        )


        if (
            meta_station.get(
                "status"
            )
            == "ok"
        ):

            stations_with_flow.append(
                station
            )


    valid_primary = (
        pd.to_numeric(
            history[
                "caudal_m3s"
            ],
            errors="coerce",
        )
        .dropna()
    )


    meta = {

        "version":
            VERSION,

        "rain_source":
            "Open-Meteo",

        "caudal_source":
            "INA A5",

        "rain_stations":
            STATIONS,

        "flow_stations":
            stations_with_flow,

        "flow_station_count":
            len(
                stations_with_flow
            ),

        "flow_series":
            flow_meta,

        "primary_flow_column":
            primary_q_column,

        "future_primary_flow_column":
            future_primary_col,

        "uses_caudal":
            bool(
                len(valid_primary)
                > 0
            ),

        "current_flow_m3s":
            (
                float(
                    valid_primary.iloc[-1]
                )
                if len(valid_primary)
                else None
            ),

        "forecast_days":
            forecast_days,

        "real_rain_forecast_days":
            min(
                forecast_days,
                16,
            ),

        "rain_after_real_forecast":
            (
                "0 mm: no se inventa "
                "precipitación meteorológica"
            ),

        "datetime_timezone":
            "naive",
    }


    return (
        history,
        future,
        meta,
    )


# ============================================================
# DIAGNÓSTICO
# ============================================================

def diagnostic(
    start,
    end,
):

    result = {

        "version":
            VERSION,

        "caudal_var_id":
            VAR_ID_CAUDAL,

        "catalog_url":
            INA_SERIES_GEOJSON_URL,

        "observations_url":
            INA_OBSERVATIONS_URL,

        "start":
            str(start),

        "end":
            str(end),

        "stations":
            {},
    }


    try:

        catalog = get_ina_catalog()

        result[
            "catalog_records"
        ] = len(catalog)

        if (
            not catalog.empty
            and "var_id"
            in catalog.columns
        ):

            result[
                "caudal_catalog_records"
            ] = int(
                (
                    catalog["var_id"]
                    == VAR_ID_CAUDAL
                ).sum()
            )

    except Exception as exc:

        result[
            "catalog_error"
        ] = str(exc)


    for station in STATIONS:

        station_result = {}


        try:

            candidates = (
                candidatos_caudal_estacion(
                    station,
                    start,
                    end,
                )
            )


            station_result[
                "candidate_count"
            ] = len(
                candidates
            )


            if not candidates.empty:

                station_result[
                    "top_candidates"
                ] = (
                    candidates[
                        [
                            c
                            for c in [
                                "series_id",
                                "series_name",
                                "river",
                                "count",
                                "score",
                            ]
                            if c
                            in candidates.columns
                        ]
                    ]
                    .head(5)
                    .astype(str)
                    .to_dict(
                        orient="records"
                    )
                )


            selected = (
                seleccionar_serie_caudal(
                    station,
                    start,
                    end,
                )
            )


            station_result[
                "selected"
            ] = selected


            if selected is not None:

                flow, meta = (
                    get_caudal_station(
                        station,
                        start,
                        end,
                    )
                )


                station_result[
                    "records"
                ] = len(flow)


                station_result[
                    "meta"
                ] = meta


        except Exception as exc:

            station_result[
                "error"
            ] = str(exc)


        result[
            "stations"
        ][
            station
        ] = station_result


    return result
