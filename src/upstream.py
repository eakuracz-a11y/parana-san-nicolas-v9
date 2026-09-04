# ============================================================
# PARANÁ · SAN NICOLÁS
# src/upstream.py
# V11.14 COMPLETO
#
# Mejora principal:
# - combina TODAS las series históricas válidas de nivel de INA
#   para cada estación, en lugar de usar una única serie "actual";
# - consulta por bloques para evitar fallos en ventanas largas;
# - incorpora Diamante;
# - prioriza observación directa y mayor cobertura;
# - conserva metadatos de series usadas.
# ============================================================

import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

VERSION = "V11.14"

INA_SERIES_URL = "https://alerta.ina.gob.ar/pub/datos/series"
INA_A5_URL = "https://alerta.ina.gob.ar/a5/getObservaciones"

UPSTREAM_STATIONS = [
    "Corrientes",
    "Goya",
    "La Paz",
    "Paraná",
    "Diamante",
    "Rosario",
    "Villa Constitución",
]

VAR_ID_LEVEL = 2
REQUEST_TIMEOUT = 45
HISTORY_BLOCK_YEARS = 4
MAX_CANDIDATE_SERIES = 12


def normalizar_texto(texto):
    if texto is None:
        return ""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.strip().lower()


def _naive_date(value):
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).tz_convert(None).normalize()


def request_json(url, params=None, timeout=REQUEST_TIMEOUT):
    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={
            "User-Agent": "Parana-San-Nicolas-V11.14/1.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=1)
def obtener_catalogo():
    try:
        data = request_json(INA_SERIES_URL)
    except Exception:
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value
    return []


def buscar_series_nivel(estacion):
    """Devuelve todas las series de nivel candidatas ordenadas por calidad."""
    objetivo = normalizar_texto(estacion)
    candidatos = []

    for row in obtener_catalogo():
        if not isinstance(row, dict):
            continue

        nombre = normalizar_texto(row.get("estacion_nombre", ""))
        if nombre != objetivo:
            continue

        try:
            varid = int(row.get("varid", -1))
        except Exception:
            continue
        if varid != VAR_ID_LEVEL:
            continue

        series_id = row.get("seriesid") or row.get("seriesId")
        try:
            series_id = int(series_id)
        except Exception:
            continue

        try:
            procid = int(row.get("procid", -1))
        except Exception:
            procid = -1

        try:
            obs_count = int(row.get("obs_count", 0) or 0)
        except Exception:
            obs_count = 0

        from_date = _naive_date(row.get("from_date"))
        to_date = _naive_date(row.get("to_date"))
        span_days = 0
        if pd.notna(from_date) and pd.notna(to_date) and to_date >= from_date:
            span_days = int((to_date - from_date).days) + 1

        score = 0.0
        if procid == 1:
            score += 120.0
        elif procid not in (4, 8):
            score += 45.0
        if procid == 4:
            score -= 120.0
        if procid == 8:
            score -= 100.0

        score += min(50.0, np.log10(max(obs_count, 1)) * 12.0)
        score += min(45.0, span_days / 365.25 * 2.0)

        if pd.notna(to_date):
            age = (pd.Timestamp.today().normalize() - to_date).days
            if age <= 7:
                score += 45.0
            elif age <= 60:
                score += 30.0
            elif age <= 365:
                score += 15.0

        candidatos.append(
            {
                "station": estacion,
                "series_id": series_id,
                "procid": procid,
                "proc_name": row.get("proc_nombre"),
                "obs_count": obs_count,
                "from_date": from_date,
                "to_date": to_date,
                "unit": row.get("unit_nombre"),
                "score": float(score),
            }
        )

    candidatos.sort(key=lambda x: x["score"], reverse=True)
    return candidatos


def buscar_serie_nivel(estacion):
    """Compatibilidad con versiones anteriores: devuelve la mejor serie."""
    candidates = buscar_series_nivel(estacion)
    return candidates[0] if candidates else None


def _parse_observations(data):
    if not isinstance(data, list):
        return pd.DataFrame(columns=["datetime", "value"])
    df = pd.DataFrame(data)
    if df.empty or "timestart" not in df.columns or "valor" not in df.columns:
        return pd.DataFrame(columns=["datetime", "value"])

    df["datetime"] = (
        pd.to_datetime(df["timestart"], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    df["value"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["datetime", "value"])
    df = df[(df["value"] >= -5.0) & (df["value"] <= 20.0)]
    if df.empty:
        return pd.DataFrame(columns=["datetime", "value"])
    return df.groupby("datetime", as_index=False)["value"].mean()


def consultar_serie(series_id, start, end):
    params = {
        "tipo": "puntual",
        "series_id": int(series_id),
        "timestart": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "timeend": pd.Timestamp(end).strftime("%Y-%m-%d"),
    }
    try:
        return _parse_observations(request_json(INA_A5_URL, params=params))
    except Exception:
        return pd.DataFrame(columns=["datetime", "value"])


def consultar_serie_bloques(series_id, start, end):
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    frames = []
    block_start = start_ts

    while block_start <= end_ts:
        block_end = min(
            block_start + pd.DateOffset(years=HISTORY_BLOCK_YEARS) - pd.Timedelta(days=1),
            end_ts,
        )
        part = consultar_serie(series_id, block_start, block_end)
        if not part.empty:
            frames.append(part)
        block_start = block_end + pd.Timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["datetime", "value"])

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )


def _clip_candidate_window(info, requested_start, requested_end):
    start = pd.Timestamp(requested_start).normalize()
    end = pd.Timestamp(requested_end).normalize()
    cstart = info.get("from_date")
    cend = info.get("to_date")
    if pd.notna(cstart):
        start = max(start, pd.Timestamp(cstart).normalize())
    if pd.notna(cend):
        end = min(end, pd.Timestamp(cend).normalize())
    if start > end:
        return None, None
    return start, end


def _station_history_from_all_series(estacion, start, end):
    candidates = buscar_series_nivel(estacion)[:MAX_CANDIDATE_SERIES]
    frames = []
    tested = []

    for rank, info in enumerate(candidates):
        qstart, qend = _clip_candidate_window(info, start, end)
        if qstart is None:
            continue

        df = consultar_serie_bloques(info["series_id"], qstart, qend)
        records = int(len(df))
        tested.append(
            {
                "series_id": info["series_id"],
                "procid": info.get("procid"),
                "records": records,
                "from": qstart,
                "to": qend,
                "score": info.get("score"),
            }
        )
        if df.empty:
            continue

        # Menor rank = serie preferida. Ante superposición prevalece la mejor.
        df = df.copy()
        df["_rank"] = rank
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["datetime", "value"]), {
            "station": estacion,
            "status": "sin_datos",
            "tested": tested,
            "series_used": [],
        }

    merged = pd.concat(frames, ignore_index=True)
    merged = (
        merged.sort_values(["datetime", "_rank"])
        .drop_duplicates("datetime", keep="first")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    used = sorted(
        set(
            int(item["series_id"])
            for item in tested
            if item.get("records", 0) > 0
        )
    )

    return merged[["datetime", "value"]], {
        "station": estacion,
        "status": "ok",
        "records": int(len(merged)),
        "start": merged["datetime"].min(),
        "end": merged["datetime"].max(),
        "series_used": used,
        "tested": tested,
    }


def get_upstream_history(start, end):
    """
    Descarga y une el histórico de niveles aguas arriba.

    V11.14 no depende de una única serie INA por estación. Une las series
    fragmentadas históricas y actuales, lo que aumenta fuertemente la
    superposición Corrientes ↔ San Nicolás para el análisis de propagación.
    """
    resultado = None
    metadata = {"version": VERSION, "stations": {}}

    for estacion in UPSTREAM_STATIONS:
        df, info = _station_history_from_all_series(estacion, start, end)
        metadata["stations"][estacion] = info
        if df.empty:
            continue

        nombre_columna = "nivel_" + normalizar_texto(estacion).replace(" ", "_")
        # Compatibilidad con nombre esperado por el resto del proyecto.
        if nombre_columna == "nivel_villa_constitucion":
            pass
        df = df.rename(columns={"value": nombre_columna})

        if resultado is None:
            resultado = df
        else:
            resultado = resultado.merge(df, on="datetime", how="outer")

    if resultado is None:
        resultado = pd.DataFrame(columns=["datetime"])

    resultado = (
        resultado.sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )
    return resultado, metadata


def preparar_upstream_features(df):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    level_cols = [c for c in out.columns if c.startswith("nivel_")]

    for col in level_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].interpolate(
            limit=3,
            limit_direction="both",
            limit_area="inside",
        )
        out[f"{col}_diff1"] = out[col].diff()
        out[f"{col}_trend3"] = (out[col] - out[col].shift(3)) / 3.0
        for lag in [1, 2, 3, 5, 7, 10]:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)

    return out
