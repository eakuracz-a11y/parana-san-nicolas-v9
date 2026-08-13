import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.stress_scenario import (
    build_stress_scenario,
)


# ============================================================
# FECHAS
# ============================================================

def format_date(
    value,
):

    if value is None:

        return "Sin datos"

    try:

        return pd.to_datetime(
            value
        ).strftime(
            "%d/%m/%Y"
        )

    except Exception:

        return str(
            value
        )


# ============================================================
# INTERFAZ
# ============================================================

def render_stress_scenario(
    df,
    models,
    exog_history,
    upstream_history,
):

    st.divider()

    st.subheader(
        "⚠️ Escenario histórico de creciente · 60 días"
    )

    st.markdown(
        """
        **¿Qué pasaría partiendo del nivel actual si durante los
        próximos 60 días se reprodujeran las condiciones
        históricamente más elevadas correspondientes a cada
        época del año?**

        Para cada fecha futura se consulta todo el historial
        disponible de precipitación y caudal y se estima la
        respuesta del nivel de San Nicolás.
        """
    )

    # ========================================================
    # NIVEL BASE REAL
    # ========================================================

    current_level = None
    current_date = None

    if (
        isinstance(
            df,
            pd.DataFrame,
        )
        and not df.empty
        and "nivel"
        in df.columns
        and "datetime"
        in df.columns
    ):

        local = df.copy()

        local[
            "nivel"
        ] = pd.to_numeric(
            local[
                "nivel"
            ],
            errors="coerce",
        )

        local[
            "datetime"
        ] = pd.to_datetime(
            local[
                "datetime"
            ],
            errors="coerce",
        )

        local = local.dropna(
            subset=[
                "nivel",
                "datetime",
            ]
        )

        if not local.empty:

            local = local.sort_values(
                "datetime"
            )

            current_level = float(
                local[
                    "nivel"
                ].iloc[-1]
            )

            current_date = (
                local[
                    "datetime"
                ].iloc[-1]
            )

    # ========================================================
    # SELECTOR
    # ========================================================

    scenario_label = st.selectbox(
        "Escenario",
        [
            "Alto · P90 histórico",
            "Severo · P95 histórico",
            "Extremo histórico por fecha",
        ],
        index=2,
        key="historical_stress_selector",
    )

    scenario_map = {
        "Alto · P90 histórico":
            "alto",

        "Severo · P95 histórico":
            "severo",

        "Extremo histórico por fecha":
            "extremo",
    }

    scenario_key = scenario_map[
        scenario_label
    ]

    # ========================================================
    # CALCULAR
    # ========================================================

    try:

        with st.spinner(
            "Analizando todo el histórico disponible..."
        ):

            (
                scenario_df,
                meta,
            ) = build_stress_scenario(
                models=models,

                exog_history=exog_history,

                upstream_history=upstream_history,

                days=60,

                scenario=scenario_key,

                current_level=current_level,

                current_date=current_date,
            )

    except Exception as exc:

        st.error(
            "No fue posible construir el escenario "
            f"histórico: {exc}"
        )

        return

    if scenario_df.empty:

        st.warning(
            "El escenario no devolvió resultados."
        )

        return

    # ========================================================
    # RESULTADOS
    # ========================================================

    current_level = meta[
        "current_level"
    ]

    max_level = meta[
        "max_level"
    ]

    growth = meta[
        "growth_m"
    ]

    growth_pct = meta[
        "growth_pct"
    ]

    # ========================================================
    # FILA 1
    # ========================================================

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Nivel base observado",
        f"{current_level:.2f} m",
    )

    c2.metric(
        "Máximo estimado",
        f"{max_level:.2f} m",
    )

    c3.metric(
        "Crecimiento estimado",
        f"+{growth:.2f} m",
    )

    c4.metric(
        "Variación relativa",
        (
            f"+{growth_pct:.1f}%"
            if pd.notna(
                growth_pct
            )
            else "Sin datos"
        ),
    )

    # ========================================================
    # FILA 2
    # ========================================================

    d1, d2, d3, d4 = st.columns(
        4
    )

    d1.metric(
        "Mayor lluvia aplicada",
        (
            f"{meta['rain_max_60d']:.1f} mm/día"
        ),
    )

    d2.metric(
        "Acumulado escenario 60 d",
        (
            f"{meta['rain_total_60d']:.1f} mm"
        ),
    )

    d3.metric(
        "Mayor caudal aplicado",
        (
            f"{meta['flow_max_60d']:,.0f} m³/s"
        ),
    )

    d4.metric(
        "Retardo histórico",
        (
            f"{meta['response_lag_days']} días"
        ),
    )

    # ========================================================
    # FILA 3
    # ========================================================

    e1, e2, e3 = st.columns(
        3
    )

    e1.metric(
        "Fecha máximo nivel",
        format_date(
            meta[
                "max_level_date"
            ]
        ),
    )

    e2.metric(
        "Nivel día 30",
        (
            f"{meta['level_day_30']:.2f} m"
        ),
    )

    e3.metric(
        "Nivel día 60",
        (
            f"{meta['level_day_60']:.2f} m"
        ),
    )

    # ========================================================
    # GRÁFICO
    # ========================================================

    fig = go.Figure()

    # --------------------------------------------------------
    # HISTÓRICO RECIENTE
    # --------------------------------------------------------

    if (
        isinstance(
            df,
            pd.DataFrame,
        )
        and not df.empty
    ):

        obs = df.tail(
            45
        )

        fig.add_trace(
            go.Scatter(
                x=obs[
                    "datetime"
                ],
                y=obs[
                    "nivel"
                ],
                mode="lines",
                name="Nivel observado",
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Nivel observado: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    # --------------------------------------------------------
    # INCERTIDUMBRE
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=scenario_df[
                "datetime"
            ],
            y=scenario_df[
                "upper"
            ],
            mode="lines",
            line=dict(
                width=0,
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=scenario_df[
                "datetime"
            ],
            y=scenario_df[
                "lower"
            ],
            mode="lines",
            line=dict(
                width=0,
            ),
            fill="tonexty",
            name="Incertidumbre experimental",
            hoverinfo="skip",
        )
    )

    # --------------------------------------------------------
    # NIVEL ESCENARIO
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=scenario_df[
                "datetime"
            ],
            y=scenario_df[
                "prediction"
            ],
            mode="lines+markers",
            line=dict(
                dash="dash",
                width=3,
            ),
            marker=dict(
                size=5,
            ),
            name=scenario_label,
            customdata=(
                scenario_df[
                    [
                        "precip_mm",
                        "caudal_m3s",
                    ]
                ]
            ),
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>"
                "Nivel estimado: %{y:.2f} m"
                "<br>"
                "Lluvia histórica: %{customdata[0]:.1f} mm"
                "<br>"
                "Caudal histórico: %{customdata[1]:,.0f} m³/s"
                "<extra></extra>"
            ),
        )
    )

    # --------------------------------------------------------
    # NIVEL BASE
    # --------------------------------------------------------

    fig.add_hline(
        y=current_level,
        line_dash="dot",
        annotation_text=(
            f"Nivel base {current_level:.2f} m"
        ),
        annotation_position="bottom right",
    )

    # --------------------------------------------------------
    # MÁXIMO
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[
                meta[
                    "max_level_date"
                ]
            ],
            y=[
                max_level
            ],
            mode="markers+text",
            marker=dict(
                size=13,
                symbol="diamond",
            ),
            text=[
                f"{max_level:.2f} m"
            ],
            textposition="top center",
            name="Máximo estimado",
        )
    )

    fig.update_layout(
        height=570,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.05,
        ),
    )

    fig.update_xaxes(
        title_text="Fecha",
        type="date",
        tickformat="%d/%m/%Y",
    )

    fig.update_yaxes(
        title_text=(
            "Nivel hidrométrico (m)"
        ),
        range=[
            0,
            7,
        ],
        dtick=0.5,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # EXPLICACIÓN
    # ========================================================

    st.info(
        f"La simulación comienza en el último nivel observado, "
        f"**{current_level:.2f} m**. Para cada uno de los "
        "60 días se buscan las condiciones históricas "
        "correspondientes a esa época del año. "
        "El escenario de creciente nunca utiliza un nivel "
        "inicial inferior al nivel observado."
    )

    # ========================================================
    # TABLA DÍA POR DÍA
    # ========================================================

    with st.expander(
        "📋 Ver máximos históricos y nivel estimado día por día"
    ):

        table = scenario_df.copy()

        table[
            "Fecha futura"
        ] = pd.to_datetime(
            table[
                "datetime"
            ]
        ).dt.strftime(
            "%d/%m/%Y"
        )

        table[
            "Nivel estimado (m)"
        ] = (
            table[
                "prediction"
            ]
            .round(
                2
            )
        )

        table[
            "Lluvia histórica (mm)"
        ] = (
            table[
                "precip_mm"
            ]
            .round(
                1
            )
        )

        table[
            "Caudal histórico (m³/s)"
        ] = (
            table[
                "caudal_m3s"
            ]
            .round(
                0
            )
        )

        table[
            "Fecha histórica lluvia"
        ] = pd.to_datetime(
            table[
                "rain_source_date"
            ],
            errors="coerce",
        ).dt.strftime(
            "%d/%m/%Y"
        )

        table[
            "Fecha histórica caudal"
        ] = pd.to_datetime(
            table[
                "flow_source_date"
            ],
            errors="coerce",
        ).dt.strftime(
            "%d/%m/%Y"
        )

        st.dataframe(
            table[
                [
                    "Fecha futura",
                    "Nivel estimado (m)",
                    "Lluvia histórica (mm)",
                    "Fecha histórica lluvia",
                    "Caudal histórico (m³/s)",
                    "Fecha histórica caudal",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # COBERTURA HISTÓRICA
    # ========================================================

    with st.expander(
        "🗂️ Cobertura histórica utilizada"
    ):

        st.write(
            "**Precipitación:**",
            format_date(
                meta.get(
                    "rain_history_start"
                )
            ),
            "→",
            format_date(
                meta.get(
                    "rain_history_end"
                )
            ),
        )

        st.write(
            "**Nivel San Nicolás:**",
            format_date(
                meta.get(
                    "level_history_start"
                )
            ),
            "→",
            format_date(
                meta.get(
                    "level_history_end"
                )
            ),
        )

        st.write(
            "**Caudal INA:**",
            format_date(
                meta.get(
                    "flow_history_start"
                )
            ),
            "→",
            format_date(
                meta.get(
                    "flow_history_end"
                )
            ),
        )

        st.caption(
            "La aplicación no limita el escenario a los últimos "
            "5 años. Utiliza toda la historia que cada fuente "
            "puede proporcionar."
        )

    # ========================================================
    # MODELO
    # ========================================================

    with st.expander(
        "🧠 Diagnóstico del modelo histórico de creciente"
    ):

        st.write(
            "**Registros históricos coincidentes utilizados:**",
            meta.get(
                "flood_training_rows",
                "-",
            ),
        )

        st.write(
            "**RMSE del crecimiento:**",
            (
                f"{meta['flood_model_rmse']:.3f} m"
            ),
        )

        st.write(
            "**MAE del crecimiento:**",
            (
                f"{meta['flood_model_mae']:.3f} m"
            ),
        )

        st.write(
            "**Crecimiento P95 histórico en 14 días:**",
            (
                f"{meta['historical_p95_growth']:.2f} m"
            ),
        )

        st.write(
            "**Mayor crecimiento histórico observado en 14 días:**",
            (
                f"{meta['historical_max_growth']:.2f} m"
            ),
        )

    # ========================================================
    # ADVERTENCIA
    # ========================================================

    st.warning(
        "Este escenario combina máximos históricos correspondientes "
        "a cada época del año para realizar una prueba de estrés. "
        "No significa que esas condiciones vayan a repetirse ni "
        "constituye un pronóstico oficial o un modelo hidráulico."
    )
