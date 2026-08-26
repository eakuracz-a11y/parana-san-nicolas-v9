import streamlit as st
import traceback


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# MOSTRAR PANTALLA ANTES DE CUALQUIER IMPORT
# ============================================================

st.title("🌊 PARANÁ · SAN NICOLÁS")

st.caption(
    "V12.0 · Diagnóstico de módulos"
)

st.success(
    "✅ Streamlit inició correctamente."
)

st.write(
    "Ahora verificaremos cada archivo del proyecto por separado."
)

st.divider()


# ============================================================
# INA
# ============================================================

st.subheader("1. Verificando src/ina.py")

try:

    from src.ina import observed

    st.success(
        "✅ src/ina.py funciona correctamente"
    )

except Exception as exc:

    st.error(
        f"❌ Error en src/ina.py: {exc}"
    )

    st.code(
        traceback.format_exc(),
        language="text",
    )


# ============================================================
# MODEL
# ============================================================

st.subheader("2. Verificando src/model.py")

try:

    from src.model import (
        train,
        predict,
        resumen_niveles_estaciones,
    )

    st.success(
        "✅ src/model.py funciona correctamente"
    )

except Exception as exc:

    st.error(
        f"❌ Error en src/model.py: {exc}"
    )

    st.code(
        traceback.format_exc(),
        language="text",
    )


# ============================================================
# EXOGENOUS
# ============================================================

st.subheader("3. Verificando src/exogenous.py")

try:

    from src.exogenous import (
        get_exogenous_data,
    )

    st.success(
        "✅ src/exogenous.py funciona correctamente"
    )

except Exception as exc:

    st.error(
        f"❌ Error en src/exogenous.py: {exc}"
    )

    st.code(
        traceback.format_exc(),
        language="text",
    )


# ============================================================
# UPSTREAM
# ============================================================

st.subheader("4. Verificando src/upstream.py")

try:

    from src.upstream import (
        get_upstream_history,
    )

    st.success(
        "✅ src/upstream.py funciona correctamente"
    )

except Exception as exc:

    st.error(
        f"❌ Error en src/upstream.py: {exc}"
    )

    st.code(
        traceback.format_exc(),
        language="text",
    )


# ============================================================
# STRESS
# ============================================================

st.subheader("5. Verificando src/stress_ui.py")

try:

    from src.stress_ui import (
        render_stress_scenario,
    )

    st.success(
        "✅ src/stress_ui.py funciona correctamente"
    )

except Exception as exc:

    st.warning(
        f"⚠️ src/stress_ui.py presenta un problema: {exc}"
    )

    st.code(
        traceback.format_exc(),
        language="text",
    )


# ============================================================
# FIN
# ============================================================

st.divider()

st.info(
    "Si podés leer esta pantalla, Streamlit y app.py "
    "están funcionando. El mensaje en rojo indicará "
    "exactamente qué archivo debemos corregir."
)

st.caption(
    "Paraná · San Nicolás V12.0 · Diagnóstico"
)
