"""
PARANÁ · SAN NICOLÁS
Seguimiento mensual V11.19.

Conserva la primera emisión de cada mes.
Los días anteriores al inicio quedan vacíos.
Las lecturas reales se actualizan sin reemplazar
las predicciones originales.
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

    if value is pd.NA or value is pd.NaT:
        return None

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

        for position in value["dates"]:
            column = frame.columns[position]
            frame[column] = pd.to_datetime(
                frame[column]
            )

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
    """
    Guarda primero en un archivo temporal y luego
    reemplaza el destino para evitar archivos incompletos.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = json.dumps(
        encode(value),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    if compressed:
        payload = gzip.compress(
            payload,
            mtime=0,
        )

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
    payload = path.read_bytes()

    if compressed:
        payload = gzip.decompress(payload)

    return decode(json.loads(payload))


def load_latest():
    path = DATA / "latest.json.gz"

    if not path.exists():
        return None

    return read_json(
        path,
        compressed=True,
    )


def freeze_month(result, issued):
    """
    Conserva la primera emisión del mes.

    Esta función se ejecuta dentro del bloqueo
    de run_update para evitar escrituras simultáneas.
    """
    key = issued.strftime("%Y-%m")
    path = DATA / (key + ".json")

    if path.exists():
        return

    forecast = result["forecast"]
    predictions = {}

    for _, row in forecast.iterrows():
        target = pd.Timestamp(
            row["datetime"]
        ).date()

        estimate = float(row["prediction"])

        if (
            issued < target
            and target.strftime("%Y-%m") == key
            and math.isfinite(estimate)
        ):
            predictions[target.isoformat()] = estimate

    # No se desplazan fechas si el dato INA está atrasado.
    has_future = (
        pd.to_datetime(forecast["datetime"]).dt.date
        > issued
    ).any()

    if not predictions and not has_future:
        raise ValueError(
            "El modelo no produjo fechas futuras "
            "dentro del mes. No se creó un registro vacío."
        )

    write_json(
        path,
        {
            "month": key,
            "issued_date": issued.isoformat(),
            "generated_at": datetime.now(TZ).isoformat(),
            "base_observation": pd.Timestamp(
                result["sn_history"]["datetime"].max()
            ).isoformat(),
            "app_version": "V11.19",
            "training_years": result.get(
                "training_years",
                8,
            ),
            "predictions": predictions,
            "actual": {},
            "checked_at": None,
        },
    )


def record_actual(raw, now):
    """
    Registra la última lectura disponible de cada
    día argentino, sin modificar las predicciones.
    """
    if raw.empty:
        return

    values = raw.copy()

    values["datetime"] = (
        pd.to_datetime(
            values["datetime"],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(TZ)
    )

    values["value"] = pd.to_numeric(
        values["value"],
        errors="coerce",
    )

    values = (
        values
        .dropna(
            subset=[
                "datetime",
                "value",
            ]
        )
        .sort_values("datetime")
    )

    for path in sorted(DATA.glob("????-??.json")):
        month = read_json(path)

        first = date.fromisoformat(
            month["issued_date"]
        )

        for _, row in values.iterrows():
            day = row["datetime"].date()

            if (
                first <= day <= now
                and day.strftime("%Y-%m") == month["month"]
            ):
                month["actual"][day.isoformat()] = {
                    "value": float(row["value"]),
                    "timestamp": row["datetime"].isoformat(),
                }

        month["checked_at"] = (
            datetime.now(TZ).isoformat()
        )

        write_json(path, month)


def run_update(
    training_years=8,
    force=False,
    observations_only=False,
):
    """
    Ejecuta el cálculo diario cuando corresponde
    y completa las observaciones reales.

    observations_only=True evita entrenar el modelo.
    """
    from src.ina import observed

    DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FileLock(
        str(DATA / "update.lock"),
        timeout=1,
    ):
        today = today_local()
        latest = load_latest()

        needs_forecast = (
            force
            or latest is None
            or pd.Timestamp(
                latest["base_date"]
            ).date() != today
        )

        if needs_forecast and not observations_only:
            from src.pipeline import calculate

            latest = calculate(
                today,
                training_years=training_years,
            )

            latest["training_years"] = training_years
            latest["last_update"] = datetime.now(TZ)

            freeze_month(latest, today)

            write_json(
                DATA / "latest.json.gz",
                latest,
                compressed=True,
            )

        elif (
            latest
            and not observations_only
            and pd.Timestamp(
                latest["base_date"]
            ).date() == today
        ):
            freeze_month(latest, today)

        months = sorted(
            DATA.glob("????-??.json")
        )

        start = today - timedelta(days=7)

        if months:
            first_month = read_json(months[0])

            start = min(
                start,
                date.fromisoformat(
                    first_month["issued_date"]
                ),
            )

        raw, error = observed(
            start.isoformat(),
            (
                today + timedelta(days=1)
            ).isoformat(),
        )

        if error:
            raise RuntimeError(
                "Pronósticos conservados; falló la consulta "
                "de lecturas reales: "
                + str(error)
            )

        record_actual(raw, today)

        return latest


def month_table(month, today=None):
    today = today or today_local()

    start = pd.Timestamp(
        month["month"] + "-01"
    )

    dates = pd.date_range(
        start,
        start + pd.offsets.MonthEnd(0),
        freq="D",
    )

    rows = []

    for day in dates:
        key = day.strftime("%Y-%m-%d")

        estimate = month["predictions"].get(key)

        actual = (
            month["actual"]
            .get(key, {})
            .get("value")
            if day.date() <= today
            else None
        )

        if day.date() < date.fromisoformat(
            month["issued_date"]
        ):
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
        "Seguimiento mensual · "
        "pronóstico original y nivel real"
    )

    paths = sorted(
        DATA.glob("????-??.json"),
        reverse=True,
    )

    if not paths:
        st.info(
            "Todavía no se ejecutó el primer cálculo. "
            "En GitHub, abrí Actions → Actualizar Rio Parana "
            "→ Run workflow. El seguimiento comenzará "
            "en esa primera ejecución exitosa."
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
        month = read_json(
            DATA / (selected + ".json")
        )

    except Exception as exc:
        st.error(
            "No se pudo leer el mes; "
            f"no se modificó el archivo: {exc}"
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

    if len(interval) != 2:
        st.info(
            "Seleccioná la fecha final del período."
        )
        return

    view = table[
        table["Fecha"].dt.date.between(
            interval[0],
            interval[1],
        )
    ]

    confirmed = view["Error [cm]"].dropna()

    a, b, c = st.columns(3)

    a.metric(
        "Días comparados",
        len(confirmed),
    )

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
                line={
                    "color": color,
                },
                hovertemplate=(
                    "%{x|%d/%m/%Y}: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=340,
        margin={
            "l": 10,
            "r": 10,
            "t": 15,
            "b": 10,
        },
        hovermode="x unified",
        legend={
            "orientation": "h",
            "y": 1.12,
        },
        yaxis={
            "title": "Nivel [m]",
            "range": [0, 7],
        },
        xaxis={
            "tickformat": "%d/%m",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    issued = pd.Timestamp(
        month["issued_date"]
    ).strftime("%d/%m/%Y")

    st.caption(
        f"Emisión fija: {issued}. "
        "Error = real − estimado. "
        "El día de emisión es referencia; la evaluación "
        "comienza con las fechas futuras disponibles."
    )

    if pd.Timestamp(month["issued_date"]).day != 1:
        st.caption(
            "Mes iniciado parcialmente: los días anteriores "
            "a la emisión quedan vacíos. "
            "No se simuló una emisión el día 1."
        )

    st.caption(
        "Última consulta real: "
        + (month["checked_at"] or "Pendiente")
        + ". La lectura de hoy puede actualizarse "
        "durante el día."
    )

    with st.expander(
        "Ver valores y errores día a día"
    ):
        display = view.copy()

        display["Fecha"] = (
            display["Fecha"]
            .dt.strftime("%d/%m/%Y")
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

    if st.button(
        "Actualizar solo lecturas reales"
    ):
        try:
            with st.spinner(
                "Consultando INA, sin entrenar..."
            ):
                run_update(
                    observations_only=True
                )

            st.rerun()

        except Exception as exc:
            st.warning(str(exc))
