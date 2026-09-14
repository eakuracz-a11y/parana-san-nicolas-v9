"""
PARANÁ · SAN NICOLÁS
Seguimiento mensual V11.21

- Consulta INA antes de decidir si recalcula.
- Detecta cambios de fecha, hora o valor observado.
- Conserva la primera emisión mensual.
- Guarda una copia de cada nuevo pronóstico.
- Actualiza observaciones sin reemplazar predicciones.
"""

from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import gzip
import io
import json
import math
import os
import tempfile

import numpy as np
import pandas as pd

from filelock import FileLock


VERSION = "V11.21"
TZ = ZoneInfo("America/Argentina/Buenos_Aires")
ROOT = Path(__file__).resolve().parents[1]

DATA = Path(
    os.environ.get(
        "RIO_DATA_DIR",
        str(ROOT / "data" / "monthly_v1"),
    )
)


def today_local():
    return datetime.now(TZ).date()


def encode(value):
    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "frame",
            "data": value.to_json(
                orient="split",
                date_format="iso",
            ),
            "dates": [
                i
                for i, column in enumerate(value.columns)
                if pd.api.types.is_datetime64_any_dtype(
                    value[column]
                )
            ],
        }

    if isinstance(value, pd.Series):
        return {
            "__type__": "series",
            "data": encode(value.to_frame()),
        }

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return {
            "__type__": "date",
            "data": value.isoformat(),
        }

    if isinstance(value, dict):
        return {
            str(key): encode(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(item) for item in value]

    if isinstance(value, np.generic):
        return encode(value.item())

    if isinstance(value, float) and not math.isfinite(value):
        return {
            "__type__": "nonfinite",
            "data": str(value),
        }

    return value


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]

    if not isinstance(value, dict):
        return value

    kind = value.get("__type__")

    if kind == "nonfinite":
        return float(value["data"])

    if kind == "frame":
        frame = pd.read_json(
            io.StringIO(value["data"]),
            orient="split",
            convert_dates=False,
        )

        for position in value.get("dates", []):
            column = frame.columns[position]
            frame[column] = pd.to_datetime(frame[column])

        return frame

    if kind == "series":
        return decode(value["data"]).iloc[:, 0]

    if kind == "date":
        return pd.Timestamp(value["data"])

    return {
        key: decode(item)
        for key, item in value.items()
    }


def write_json(path, value, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        encode(value),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    if compressed:
        payload = gzip.compress(payload, mtime=0)

    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
    )

    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(name, path)

    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path, compressed=False):
    payload = Path(path).read_bytes()

    if compressed:
        payload = gzip.decompress(payload)

    return decode(json.loads(payload))


def load_latest():
    path = DATA / "latest.json.gz"

    if not path.exists():
        return None

    return read_json(path, compressed=True)


def prepare_actual(raw, cutoff=None):
    """
    Conserva la última lectura de cada día argentino.
    No interpola ni completa días sin observaciones.
    """
    columns = ["timestamp", "day", "value"]

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(columns=columns)

    if "datetime" not in raw.columns:
        raise ValueError("La respuesta INA no contiene datetime.")

    value_column = next(
        (
            column
            for column in (
                "value",
                "nivel",
                "nivel_san_nicolas",
            )
            if column in raw.columns
        ),
        None,
    )

    if value_column is None:
        raise ValueError("La respuesta INA no contiene nivel.")

    values = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                raw["datetime"],
                utc=True,
                errors="coerce",
            ).dt.tz_convert(TZ),
            "value": pd.to_numeric(
                raw[value_column],
                errors="coerce",
            ),
        }
    )

    values = values.dropna(subset=["timestamp", "value"])
    values = values[np.isfinite(values["value"])].copy()

    cutoff = pd.Timestamp(
        cutoff if cutoff is not None else datetime.now(TZ)
    )

    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(TZ)
    else:
        cutoff = cutoff.tz_convert(TZ)

    values = values[
        values["timestamp"] < cutoff
    ].copy()

    values = values.sort_values("timestamp", kind="stable")
    values["day"] = values["timestamp"].dt.date

    return (
        values.drop_duplicates("day", keep="last")
        .reset_index(drop=True)[columns]
    )


def latest_observation_changed(latest, observation):
    if not isinstance(latest, dict):
        return True

    if latest.get("pipeline_version") != VERSION:
        return True

    try:
        old_timestamp = pd.Timestamp(
            latest["observation_timestamp"]
        )
        new_timestamp = pd.Timestamp(observation["timestamp"])

        if old_timestamp.tzinfo is None:
            return True

        old_level = float(latest["observation_level"])
        new_level = float(observation["value"])

        return (
            old_timestamp.tz_convert("UTC")
            != new_timestamp.tz_convert("UTC")
            or not math.isclose(
                old_level,
                new_level,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

    except (KeyError, TypeError, ValueError, OverflowError):
        return True


def freeze_month(result, issued):
    """Conserva la primera emisión de cada mes."""
    key = issued.strftime("%Y-%m")
    path = DATA / (key + ".json")

    if path.exists():
        return

    forecast = result["forecast"]

    if not isinstance(forecast, pd.DataFrame) or forecast.empty:
        raise ValueError("No hay pronóstico para registrar.")

    predictions = {}
    has_future = False

    for _, row in forecast.iterrows():
        target = pd.Timestamp(row["datetime"]).date()
        estimate = float(row["prediction"])

        if issued < target and math.isfinite(estimate):
            has_future = True

            if target.strftime("%Y-%m") == key:
                predictions[target.isoformat()] = estimate

    if not has_future:
        raise ValueError(
            "El pronóstico no contiene fechas futuras. "
            "No se creó un registro mensual vacío."
        )

    write_json(
        path,
        {
            "month": key,
            "issued_date": issued.isoformat(),
            "generated_at": datetime.now(TZ).isoformat(),
            "base_observation": result.get(
                "observation_timestamp",
                pd.Timestamp(
                    result["sn_history"]["datetime"].max()
                ).isoformat(),
            ),
            "base_level": result.get("observation_level"),
            "app_version": VERSION,
            "training_years": result.get("training_years", 8),
            "predictions": predictions,
            "actual": {},
            "checked_at": None,
        },
    )


def archive_forecast(result):
    """
    Guarda cada nueva emisión en un archivo diferente.
    No reemplaza la referencia mensual.
    """
    generated = datetime.now(TZ)
    stamp = generated.strftime("%Y%m%dT%H%M%S_%f")

    write_json(
        DATA / "emissions" / (stamp + ".json.gz"),
        {
            "generated_at": generated.isoformat(),
            "base_observation": result.get(
                "observation_timestamp"
            ),
            "base_level": result.get("observation_level"),
            "pipeline_version": result.get("pipeline_version"),
            "forecast": result["forecast"],
        },
        compressed=True,
    )


def record_actual(raw, now):
    """Actualiza lecturas reales, nunca las predicciones."""
    cutoff = min(
        pd.Timestamp(datetime.now(TZ)),
        (
            pd.Timestamp(now) + pd.Timedelta(days=1)
        ).tz_localize(TZ),
    )

    values = prepare_actual(raw, cutoff=cutoff)

    if values.empty:
        return

    for path in sorted(DATA.glob("????-??.json")):
        month = read_json(path)
        first = date.fromisoformat(month["issued_date"])
        actual = month.setdefault("actual", {})

        for _, row in values.iterrows():
            day = row["day"]

            if not (
                first <= day <= now
                and day.strftime("%Y-%m") == month["month"]
            ):
                continue

            key = day.isoformat()
            timestamp = pd.Timestamp(row["timestamp"])
            previous = actual.get(key)

            # Una respuesta parcial no debe sustituir una
            # observación más reciente por otra anterior.
            if previous and previous.get("timestamp"):
                previous_timestamp = pd.Timestamp(
                    previous["timestamp"]
                )

                if previous_timestamp.tzinfo is None:
                    previous_timestamp = (
                        previous_timestamp.tz_localize(TZ)
                    )

                if timestamp < previous_timestamp:
                    continue

            actual[key] = {
                "value": float(row["value"]),
                "timestamp": timestamp.isoformat(),
            }

        month["checked_at"] = datetime.now(TZ).isoformat()
        write_json(path, month)


def run_update(
    training_years=8,
    force=False,
    observations_only=False,
):
    """
    Consulta primero INA.

    Recalcula cuando:
    - no hay resultado guardado;
    - cambió la lectura oficial;
    - cambió el día de ejecución;
    - cambió la configuración de entrenamiento;
    - se fuerza la ejecución;
    - el resultado pertenece al pipeline anterior.
    """
    from src.ina import observed

    DATA.mkdir(parents=True, exist_ok=True)

    with FileLock(str(DATA / "update.lock"), timeout=1):
        today = today_local()
        latest = load_latest()

        start = today - timedelta(days=7)

        for path in sorted(DATA.glob("????-??.json")):
            month = read_json(path)
            start = min(
                start,
                date.fromisoformat(month["issued_date"]),
            )

        if latest and latest.get("observation_timestamp"):
            previous_day = pd.Timestamp(
                latest["observation_timestamp"]
            ).date()
            start = min(start, previous_day)

        raw, error = observed(
            start.isoformat(),
            (today + timedelta(days=1)).isoformat(),
        )

        if error:
            raise RuntimeError(
                "Falló la consulta INA. Se conservaron "
                "los pronósticos guardados: " + str(error)
            )

        values = prepare_actual(raw)

        if values.empty:
            raise RuntimeError(
                "INA no devolvió lecturas válidas. "
                "Se conservaron los resultados anteriores."
            )

        current = values.iloc[-1]

        # Actualiza los meses existentes aunque posteriormente
        # falle el entrenamiento.
        record_actual(raw, today)

        if observations_only:
            return latest

        last_run_day = None

        if latest and latest.get("last_update") is not None:
            timestamp = pd.Timestamp(latest["last_update"])

            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert(TZ)

            last_run_day = timestamp.date()

        stored_years = (
            latest.get("training_years")
            if latest
            else None
        )

        needs_forecast = (
            force
            or latest_observation_changed(latest, current)
            or last_run_day != today
            or stored_years != training_years
        )

        if needs_forecast:
            from src.pipeline import calculate

            candidate = calculate(
                today,
                training_years=training_years,
            )

            if candidate.get("pipeline_version") != VERSION:
                raise RuntimeError(
                    "Primero reemplazá src/pipeline.py "
                    "por la versión V11.21."
                )

            candidate_timestamp = pd.Timestamp(
                candidate["observation_timestamp"]
            )
            checked_timestamp = pd.Timestamp(
                current["timestamp"]
            )

            if candidate_timestamp < checked_timestamp:
                raise RuntimeError(
                    "El cálculo recibió una observación anterior "
                    "a la recién consultada. No se reemplazó "
                    "el pronóstico guardado."
                )

            if candidate_timestamp == checked_timestamp:
                if not math.isclose(
                    float(candidate["observation_level"]),
                    float(current["value"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise RuntimeError(
                        "INA devolvió valores distintos para "
                        "la misma hora entre consultas. "
                        "Reintentá la actualización."
                    )

            candidate["training_years"] = training_years
            candidate["last_update"] = datetime.now(TZ)

            # Verifica que exista al menos una fecha futura
            # respecto de la emisión, sin desplazar fechas.
            forecast_dates = pd.to_datetime(
                candidate["forecast"]["datetime"]
            ).dt.date

            if not (forecast_dates > today).any():
                raise RuntimeError(
                    "La observación está demasiado atrasada "
                    "para producir fechas futuras. "
                    "Se conservó el resultado anterior."
                )

            archive_forecast(candidate)
            freeze_month(candidate, today)

            write_json(
                DATA / "latest.json.gz",
                candidate,
                compressed=True,
            )

            latest = candidate

        elif latest:
            freeze_month(latest, today)

        # Incluye el mes que acaba de crearse.
        record_actual(raw, today)
        return latest


def month_table(month, today=None):
    today = today or today_local()

    start = pd.Timestamp(month["month"] + "-01")

    dates = pd.date_range(
        start,
        start + pd.offsets.MonthEnd(0),
        freq="D",
    )

    first = date.fromisoformat(month["issued_date"])
    rows = []

    for day in dates:
        key = day.strftime("%Y-%m-%d")
        estimate = month.get("predictions", {}).get(key)

        actual = (
            month.get("actual", {})
            .get(key, {})
            .get("value")
            if day.date() <= today
            else None
        )

        if day.date() < first:
            estimate = None
            actual = None

        delta = (
            (actual - estimate) * 100
            if actual is not None and estimate is not None
            else np.nan
        )

        rows.append(
            {
                "Fecha": day,
                "Estimado [m]": estimate,
                "Real INA [m]": actual,
                "Error [cm]": delta,
                "Error absoluto [cm]": abs(delta),
            }
        )

    return pd.DataFrame(rows)


def render_monthly():
    import streamlit as st
    import plotly.graph_objects as go

    st.subheader(
        "Control mensual · pronóstico original y lectura INA"
    )

    st.caption(
        "La referencia mensual permanece fija. "
        "Los nuevos cálculos diarios no modifican "
        "el pronóstico original de este gráfico."
    )

    paths = sorted(
        DATA.glob("????-??.json"),
        reverse=True,
    )

    if not paths:
        st.info(
            "Todavía no hay seguimiento mensual. "
            "En GitHub ejecutá Actions → Actualizar Rio Parana "
            "→ Run workflow."
        )
        return

    selected = st.selectbox(
        "Mes",
        [path.stem for path in paths],
        format_func=lambda value: pd.Timestamp(
            value + "-01"
        ).strftime("%m/%Y"),
    )

    try:
        month = read_json(DATA / (selected + ".json"))
    except Exception as exc:
        st.error(
            "No se pudo leer el mes. "
            f"No se modificó el archivo: {exc}"
        )
        return

    table = month_table(month)
    start = table["Fecha"].min().date()
    end = table["Fecha"].max().date()

    interval = st.date_input(
        "Período a visualizar",
        (start, end),
        min_value=start,
        max_value=end,
        format="DD/MM/YYYY",
        key="range_" + selected,
    )

    if not isinstance(interval, (tuple, list)) or len(interval) != 2:
        st.info("Seleccioná la fecha final del período.")
        return

    view = table[
        table["Fecha"].dt.date.between(
            interval[0], interval[1]
        )
    ].copy()

    confirmed = view["Error [cm]"].dropna()
    a, b, c, d = st.columns(4)

    a.metric("Días comparados", len(confirmed))
    b.metric(
        "Error medio absoluto",
        (
            f"{confirmed.abs().mean():.1f} cm"
            if len(confirmed)
            else "Pendiente"
        ),
    )
    c.metric(
        "Error máximo absoluto",
        (
            f"{confirmed.abs().max():.1f} cm"
            if len(confirmed)
            else "Pendiente"
        ),
    )
    d.metric(
        "Sesgo medio · real − estimado",
        (
            f"{confirmed.mean():+.1f} cm"
            if len(confirmed)
            else "Pendiente"
        ),
    )

    fig = go.Figure()

    for column, color in [
        ("Estimado [m]", "#2563eb"),
        ("Real INA [m]", "#16a34a"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=view["Fecha"],
                y=view[column],
                name=column,
                mode="lines+markers",
                connectgaps=False,
                line={"color": color},
                hovertemplate=(
                    "%{x|%d/%m/%Y}: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=340,
        margin={"l": 10, "r": 10, "t": 15, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        yaxis={"title": "Nivel [m]", "range": [0, 7]},
        xaxis={"tickformat": "%d/%m"},
    )

    st.plotly_chart(fig, use_container_width=True)

    issued = pd.Timestamp(
        month["issued_date"]
    ).strftime("%d/%m/%Y")

    st.caption(
        f"Emisión fija: {issued}. "
        "Error = lectura real − pronóstico original. "
        "Un valor positivo indica que el río quedó "
        "por encima de lo pronosticado."
    )

    if pd.Timestamp(month["issued_date"]).day != 1:
        st.caption(
            "Mes iniciado parcialmente: los días anteriores "
            "a la emisión permanecen vacíos."
        )

    st.caption(
        "Última consulta de observaciones: "
        + (month.get("checked_at") or "Pendiente")
        + ". La lectura del día en curso es provisional "
        "hasta que termine el día y puede actualizarse."
    )

    st.caption(
        "Este control evalúa la emisión mensual fija, "
        "no el pronóstico actualizado cada día. "
        "No debe confundirse su error con el de "
        "un pronóstico emitido ayer para hoy."
    )

    with st.expander("Ver valores y errores día a día"):
        display = view.copy()
        display["Fecha"] = display["Fecha"].dt.strftime(
            "%d/%m/%Y"
        )

        st.dataframe(
            display.style.format(
                {
                    column: "{:.2f}"
                    for column in display.columns
                    if column != "Fecha"
                },
                na_rep="",
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Descargar mes en CSV",
            display.to_csv(
                index=False,
                sep=";",
                decimal=",",
            ).encode("utf-8-sig"),
            selected + ".csv",
            "text/csv",
        )

    if st.button("Actualizar solo lecturas reales"):
        try:
            with st.spinner(
                "Consultando INA sin modificar pronósticos..."
            ):
                run_update(observations_only=True)

            st.rerun()

        except Exception as exc:
            st.warning(str(exc))
