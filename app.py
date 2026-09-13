
from pathlib import Path
import sqlite3
import secrets
import calendar
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

APP_VERSION = "V2.1"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB = str(DATA_DIR / "tareas.db")
LOGO = ASSETS_DIR / "sevion_logo.png"

PEOPLE = [
    ("Camille Maia", "camille.maia@sevion.com.br"),
    ("Eduardo Matos", "eduardo.matos@sevion.com.br"),
    ("Bruno Maia", "bruno.maia@sevion.com.br"),
    ("Ana Nolasco", "ana.nolasco@sevion.com.br"),
    ("Flavia Guedes", "flavia.guedes@sevion.com.br"),
    ("Marcela Roque", "marcela.roque@sevion.com.br"),
]

SEED = [('Relatório de resultados de amostras de adjuvantes (Lotes)', 'Cerrada', '—'), ('Relatório de teste de emulsão (Projeto ADJ G)', 'Cerrada', '—'), ('Produção de volume teste (Projeto ADJ G)', 'Cerrada', 'Novo volume será necessário caso haja continuidade dos testes ou produção em maior escala.'), ('Agendamento da avaliação nas propriedades (Projeto ADJ G)', 'Cerrada', '—'), ('Revisão do manejo dos produtores e organização da pesquisa, materiais e protocolos para teste de compatibilidade (Projeto ADJ G)', 'Cerrada', '—'), ('Relatórios dos testes – Fazenda Multiagri', 'Cerrada', '—'), ('Formulação com D-Limoneno', 'Cerrada', '—'), ('Treinamento de Brigadista', 'Cerrada', '—'), ('Treinamento de Emergência Química', 'Cerrada', '—'), ('Troca da coluna de resina do deionizador', 'Cerrada', 'Substituição realizada.'), ('Treinamento de Uso Correto de EPI', 'Cerrada', 'Conclusão prevista para 25/07.'), ('Avaliar embalagem tipo bag de adjuvantes', 'Cerrada', 'Aprovado'), ('Teste de compatibilidade de calda e relatório (ADJ G).', 'Cerrada', 'Refazer teste e identificar outros manejos.'), ('Ajuste de performance do AAS', 'Cerrada', 'Após reunião com Tecnal.'), ('Teste de emulsão e relatório de Formulação com D-Limoneno', 'Cerrada', '—'), ('Teste de pulverização aérea (drone)', 'Cerrada', '—'), ('Conferência de estoque e atualização da planilha', 'En ejecución', 'Mensalmente, dia 25'), ('Mapa mensal da PF', 'En ejecución', 'Mensalmente, dia 26'), ('Revisão da curva de calibração de Cu (alta sensibilidade)', 'En ejecución', 'Refazer.'), ('Teste de pulverização aérea (drone - Derquian)', 'En ejecución', 'Previsão de conclusão até 13/08.'), ('Testes e relatório de Formulação com D-Limoneno', 'En ejecución', 'Fazer teste de compatibilidade com herbicidas e inseticidas, sem fungicidas.'), ('Teste de compatibilidade de calda e relatório (MSO-TC).', 'En ejecución', ''), ('Organização do laboratório sugeridas durante o treinamento de análises de solo', 'Pendiente', 'Necessário definir prioridade e cronograma.'), ('Identificação de balanças', 'Pendiente', 'Verificar modelo e imprimir'), ('Fazer curva de Ca e Mg e analisar amostra de água mineral', 'Pendiente', 'Necessário definir prioridade e cronograma.'), ('Repetição das formulações de fertilizantes para confirmação das garantias', 'Pendiente', 'Necessário definir prioridade e cronograma.'), ('Preparo das formulações discutidas na consultoria', 'Pendiente', 'Necessário definir prioridade e cronograma.'), ('Análises de CQ dos fertilizantes formulados', 'Pendiente', 'Dependente da conclusão das formulações.')]

SECTORES = {
    "Laboratorio": "LAB",
    "Producción": "PRD",
    "Logística": "LOG",
    "Mantenimiento": "MANT",
}

AREAS = {
    "Fungos": "FUN",
    "Biológico": "BIO",
    "Fertilizante": "FER",
    "Adjuvante": "ADJ",
}

TIPOS_MANT = ["Preventivo", "Correctivo", "Proactivo", "Predictivo"]
PRIORIDADES = ["Crítica", "Alta", "Media", "Baja"]
RECURRENCIAS = ["No", "Semanal", "Mensual", "Trimestral", "Semestral", "Anual"]

st.set_page_config(
    page_title="SEV | Control de Tareas",
    page_icon="✅",
    layout="wide",
)

# ============================================================
# IDENTIDAD VISUAL SEVION · MINIMALISTA
# ============================================================

BRAND_GREEN = "#19734A"
BRAND_DARK = "#183D2D"
BRAND_BG = "#F7F8F6"
BRAND_SOFT = "#F0F3F0"
BRAND_BORDER = "#DDE5DF"
TEXT_MUTED = "#66766E"

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: Inter, "Segoe UI", Arial, sans-serif;
    }}

    .stApp {{
        background: {BRAND_BG};
        color: {BRAND_DARK};
    }}

    .block-container {{
        padding-top: 2.8rem;
        padding-bottom: 2.2rem;
        max-width: 1480px;
    }}

    [data-testid="stSidebar"] {{
        background: #F3F5F3;
        border-right: 1px solid {BRAND_BORDER};
    }}

    [data-testid="stSidebar"] * {{
        color: #253A30 !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label {{
        border-radius: 10px;
        padding: .30rem .35rem;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background: #E8EEE9;
    }}

    h1, h2, h3 {{
        color: {BRAND_DARK};
        letter-spacing: -0.025em;
    }}

    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid {BRAND_BORDER};
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(22,55,40,.03);
    }}

    div[data-testid="stMetricLabel"] {{
        color: #607168;
        font-weight: 600;
    }}

    div[data-testid="stMetricValue"] {{
        color: {BRAND_DARK};
    }}

    .sev-header {{
        border-bottom: 1px solid {BRAND_BORDER};
        padding: 0 0 16px 0;
        margin: 0 0 20px 0;
    }}

    .sev-kicker {{
        color: {BRAND_GREEN};
        font-weight: 700;
        font-size: .76rem;
        letter-spacing: .10em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }}

    .sev-title {{
        font-size: 2.15rem;
        font-weight: 750;
        color: {BRAND_DARK};
        line-height: 1.08;
        margin: 0;
    }}

    .sev-subtitle {{
        color: {TEXT_MUTED};
        font-size: .92rem;
        margin-top: 6px;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BRAND_BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}

    .stButton > button {{
        border-radius: 10px;
    }}

    button[kind="primary"] {{
        background: {BRAND_GREEN} !important;
        color: white !important;
        border-color: {BRAND_GREEN} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def render_header():
    logo_col, text_col = st.columns([1.05, 5.8], vertical_alignment="center")

    with logo_col:
        if LOGO.exists():
            st.image(str(LOGO), width=175)
        else:
            st.markdown("### Sevion")

    with text_col:
        st.markdown(
            f"""
            <div class="sev-header">
                <div class="sev-kicker">Gestión operacional</div>
                <div class="sev-title">Control de Tareas</div>
                <div class="sev-subtitle">
                    {APP_VERSION} · planificación, ejecución y cumplimiento
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# BASE DE DATOS
# ============================================================

def con():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = con()
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS people(
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY,
            code TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            sector TEXT,
            area TEXT,
            maintenance_type TEXT,
            assignee_id INTEGER,
            priority TEXT,
            requested TEXT,
            start_date TEXT,
            due_date TEXT,
            status TEXT,
            progress REAL DEFAULT 0,
            observation TEXT,
            token TEXT UNIQUE,
            accepted_at TEXT,
            finished_at TEXT,
            closed_at TEXT,
            imported INTEGER DEFAULT 0,
            recurrence TEXT,
            recurrence_day INTEGER,
            recurrence_parent_id INTEGER,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS updates(
            id INTEGER PRIMARY KEY,
            task_id INTEGER,
            update_date TEXT,
            progress REAL,
            hours REAL,
            work_done TEXT,
            blockers TEXT,
            created_at TEXT
        );
        """
    )
    for name, email in PEOPLE:
        c.execute(
            "INSERT OR IGNORE INTO people(name,email,active) VALUES(?,?,1)",
            (name, email),
        )
    c.commit()

    if c.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 0:
        camille = c.execute(
            "SELECT id FROM people WHERE email=?",
            ("camille.maia@sevion.com.br",),
        ).fetchone()["id"]

        for i, (title, status, obs) in enumerate(SEED, 1):
            recurrence = "Mensual" if "Mensalmente" in obs else None
            recurrence_day = 25 if "dia 25" in obs else 26 if "dia 26" in obs else None
            progress = 100 if status == "Cerrada" else 25 if status == "En ejecución" else 0

            c.execute(
                """
                INSERT INTO tasks(
                    code,title,sector,area,assignee_id,priority,requested,status,
                    progress,observation,token,imported,recurrence,recurrence_day,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"SEV-LAB-FER-2026-{i:04d}",
                    title,
                    "LAB",
                    "FER",
                    camille,
                    "Media",
                    "2026-08-13",
                    status,
                    progress,
                    obs,
                    secrets.token_urlsafe(24),
                    1,
                    recurrence,
                    recurrence_day,
                    datetime.now().isoformat(),
                ),
            )
        c.commit()
    c.close()

st.markdown(
    '<div style="border-top:1px solid #DDE5DF;margin-top:2rem;padding-top:1rem;color:#66766E;font-size:.78rem;">SEV · Control de Tareas · V2.1</div>',
    unsafe_allow_html=True,
)

def next_code(sector, area):
    year = date.today().year
    c = con()
    rows = c.execute(
        "SELECT code FROM tasks WHERE code LIKE ?",
        (f"SEV-%-{year}-%",),
    ).fetchall()
    c.close()
    nums = []
    for r in rows:
        try:
            nums.append(int(r["code"].split("-")[-1]))
        except Exception:
            pass
    return f"SEV-{sector}-{area}-{year}-{max(nums, default=0)+1:04d}"

# ============================================================
# LÓGICA DE CUMPLIMIENTO
# ============================================================

def theoretical(row, reference=None):
    reference = reference or date.today()

    if row["status"] == "Cerrada":
        return 100.0
    if not row["start_date"] or not row["due_date"]:
        return None

    start = pd.to_datetime(row["start_date"]).date()
    due = pd.to_datetime(row["due_date"]).date()

    if reference <= start:
        return 0.0
    if reference >= due:
        return 100.0

    total = max((due - start).days, 1)
    elapsed = (reference - start).days
    return round(100 * elapsed / total, 1)

def traffic_light(row):
    if row["status"] == "Cerrada":
        return "🟢 Cerrada"
    if row["status"] == "Terminada - espera cierre":
        return "🔵 Espera cierre"

    th = theoretical(row)
    if th is None:
        return "⚪ Sin cronograma"

    real = float(row["progress"] or 0)

    if row["due_date"] and date.today() > pd.to_datetime(row["due_date"]).date() and real < 100:
        return "🔴 Vencida"

    delta = real - th
    if delta >= -5:
        return "🟢 En término"
    if delta >= -15:
        return "🟡 Atención"
    return "🔴 Atrasada"

def schedule_delta(row):
    th = theoretical(row)
    if th is None:
        return None
    return round(float(row["progress"] or 0) - th, 1)

# ============================================================
# RECURRENCIAS
# ============================================================

def add_months(d, months):
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return date(y, m, day)

def next_recurrence(d, kind):
    if kind == "Semanal":
        return d + timedelta(days=7)
    if kind == "Mensual":
        return add_months(d, 1)
    if kind == "Trimestral":
        return add_months(d, 3)
    if kind == "Semestral":
        return add_months(d, 6)
    if kind == "Anual":
        return add_months(d, 12)
    return None

def generate_recurring():
    c = con()
    masters = c.execute(
        """
        SELECT * FROM tasks
        WHERE recurrence IS NOT NULL
          AND recurrence != 'No'
          AND recurrence_parent_id IS NULL
          AND due_date IS NOT NULL
          AND due_date != ''
        """
    ).fetchall()

    for r in masters:
        due = pd.to_datetime(r["due_date"]).date()
        nxt = next_recurrence(due, r["recurrence"])

        if not nxt or nxt > date.today() + timedelta(days=45):
            continue

        exists = c.execute(
            "SELECT 1 FROM tasks WHERE recurrence_parent_id=? AND due_date=?",
            (r["id"], nxt.isoformat()),
        ).fetchone()
        if exists:
            continue

        start = nxt
        if r["start_date"]:
            old_start = pd.to_datetime(r["start_date"]).date()
            duration = max((due - old_start).days, 0)
            start = nxt - timedelta(days=duration)

        code = next_code(r["sector"], r["area"])
        c.execute(
            """
            INSERT INTO tasks(
                code,title,description,sector,area,maintenance_type,assignee_id,priority,
                requested,start_date,due_date,status,progress,observation,token,recurrence,
                recurrence_day,recurrence_parent_id,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                code,
                r["title"],
                r["description"],
                r["sector"],
                r["area"],
                r["maintenance_type"],
                r["assignee_id"],
                r["priority"],
                date.today().isoformat(),
                start.isoformat(),
                nxt.isoformat(),
                "Asignada",
                0,
                r["observation"],
                secrets.token_urlsafe(24),
                r["recurrence"],
                nxt.day,
                r["id"],
                datetime.now().isoformat(),
            ),
        )
    c.commit()
    c.close()

# ============================================================
# CARGA
# ============================================================

init_db()
generate_recurring()

# ============================================================
# PORTAL RESPONSABLE
# ============================================================

token = st.query_params.get("token")

if token:
    render_header()
    c = con()
    row = c.execute(
        """
        SELECT t.*, p.name, p.email
        FROM tasks t
        JOIN people p ON p.id=t.assignee_id
        WHERE t.token=?
        """,
        (token,),
    ).fetchone()

    if not row:
        st.error("Enlace inválido o revocado.")
        c.close()
        st.stop()

    st.subheader(row["code"])
    st.write(f"**Responsable:** {row['name']}")
    st.write(f"**Tarea:** {row['title']}")
    st.write(f"**Prioridad:** {row['priority']} · **Estado:** {row['status']}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Avance real", f"{float(row['progress'] or 0):.0f}%")
    th = theoretical(row)
    m2.metric("Avance teórico", "—" if th is None else f"{th:.0f}%")
    m3.metric("Cumplimiento", traffic_light(row))

    if row["observation"]:
        st.info(row["observation"])

    if not row["accepted_at"] and row["status"] not in ("Cerrada", "Terminada - espera cierre"):
        if st.button("Aceptar tarea", type="primary", use_container_width=True):
            c.execute(
                "UPDATE tasks SET accepted_at=?,status='Aceptada' WHERE id=?",
                (datetime.now().isoformat(), row["id"]),
            )
            c.commit()
            st.rerun()

    if row["status"] not in ("Cerrada", "Terminada - espera cierre"):
        with st.form("daily_update"):
            progress = st.slider("Avance acumulado (%)", 0, 100, int(row["progress"] or 0))
            hours = st.number_input("Horas trabajadas hoy", 0.0, 24.0, 0.0, 0.5)
            work = st.text_area("Trabajo realizado")
            blockers = st.text_area("Problemas / bloqueos")
            submitted = st.form_submit_button("Guardar actualización", type="primary", use_container_width=True)

        if submitted:
            c.execute(
                """
                INSERT INTO updates(task_id,update_date,progress,hours,work_done,blockers,created_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    row["id"],
                    date.today().isoformat(),
                    progress,
                    hours,
                    work,
                    blockers,
                    datetime.now().isoformat(),
                ),
            )

            status = "Terminada - espera cierre" if progress == 100 else "En ejecución"
            finished_at = datetime.now().isoformat() if progress == 100 else None

            c.execute(
                """
                UPDATE tasks
                SET progress=?,status=?,finished_at=COALESCE(?,finished_at)
                WHERE id=?
                """,
                (progress, status, finished_at, row["id"]),
            )
            c.commit()
            st.rerun()

    history = pd.read_sql_query(
        """
        SELECT update_date,progress,hours,work_done,blockers
        FROM updates WHERE task_id=? ORDER BY id DESC
        """,
        c,
        params=(row["id"],),
    )
    if not history.empty:
        st.subheader("Historial diario")
        st.dataframe(history, hide_index=True, use_container_width=True)
    c.close()
    st.stop()

# ============================================================
# ADMIN
# ============================================================

render_header()

page = st.sidebar.radio(
    "NAVEGACIÓN",
    [
        "Tablero",
        "Nueva tarea",
        "Tareas",
        "Calendario / Gantt",
        "Recurrentes",
        "Mantenimiento",
        "Operarios",
        "Cierres pendientes",
    ],
)

c = con()
tasks = pd.read_sql_query(
    """
    SELECT t.*,p.name AS assignee,p.email
    FROM tasks t JOIN people p ON p.id=t.assignee_id
    ORDER BY t.id DESC
    """,
    c,
)
people = pd.read_sql_query(
    "SELECT * FROM people WHERE active=1 ORDER BY name",
    c,
)

# ============================================================
# TABLERO DINÁMICO
# ============================================================

if page == "Tablero":
    st.subheader("Tablero de cumplimiento")
    st.caption("Visión ejecutiva del avance, cronograma y desvíos de las tareas.")

    f1, f2, f3, f4 = st.columns(4)
    sector_filter = f1.selectbox("Sector", ["Todos"] + list(SECTORES.values()))
    area_filter = f2.selectbox("Área", ["Todas"] + list(AREAS.values()))
    operator_filter = f3.selectbox("Operario", ["Todos"] + people["name"].tolist())
    status_filter = f4.selectbox(
        "Estado",
        ["Todos", "Pendiente", "Asignada", "Aceptada", "En ejecución", "Terminada - espera cierre", "Cerrada"],
    )

    view = tasks.copy()
    if sector_filter != "Todos":
        view = view[view["sector"] == sector_filter]
    if area_filter != "Todas":
        view = view[view["area"] == area_filter]
    if operator_filter != "Todos":
        view = view[view["assignee"] == operator_filter]
    if status_filter != "Todos":
        view = view[view["status"] == status_filter]

    view["Teórico %"] = view.apply(theoretical, axis=1)
    view["Desvío pp"] = view.apply(schedule_delta, axis=1)
    view["Semáforo"] = view.apply(traffic_light, axis=1)

    open_count = int((view["status"] != "Cerrada").sum())
    execution_count = int((view["status"] == "En ejecución").sum())
    waiting_close = int((view["status"] == "Terminada - espera cierre").sum())
    overdue_count = int(view["Semáforo"].astype(str).str.contains("Vencida|Atrasada").sum())
    requested_month = int(
        (
            pd.to_datetime(view["requested"], errors="coerce").dt.to_period("M")
            == pd.Period(date.today(), freq="M")
        ).sum()
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Abiertas", open_count)
    k2.metric("En ejecución", execution_count)
    k3.metric("Con atraso", overdue_count)
    k4.metric("Esperan cierre", waiting_close)
    k5.metric("Solicitadas este mes", requested_month)

    st.markdown("### Avance real vs. teórico")
    chart_df = view[
        view["Teórico %"].notna()
    ][["code", "progress", "Teórico %", "assignee"]].copy()

    if not chart_df.empty:
        chart_long = chart_df.melt(
            id_vars=["code", "assignee"],
            value_vars=["progress", "Teórico %"],
            var_name="Serie",
            value_name="Avance",
        )
        fig_progress = px.bar(
            chart_long,
            x="code",
            y="Avance",
            color="Serie",
            barmode="group",
            hover_data=["assignee"],
            labels={"code": "Tarea", "Avance": "Avance (%)"},
            color_discrete_map={
                "progress": "#19734A",
                "Teórico %": "#AAB8B0",
            },
        )
        fig_progress.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#183D2D"),
            yaxis=dict(gridcolor="#E8ECE9", range=[0, 105]),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_progress, use_container_width=True)
    else:
        st.info("Las tareas históricas sin fechas no tienen avance teórico calculable.")

    st.markdown("### Cronograma de cumplimiento · Gantt")
    gantt = view[
        view["start_date"].notna()
        & (view["start_date"] != "")
        & view["due_date"].notna()
        & (view["due_date"] != "")
    ].copy()

    if not gantt.empty:
        gantt["Inicio"] = pd.to_datetime(gantt["start_date"])
        gantt["Final"] = pd.to_datetime(gantt["due_date"])
        gantt["Etiqueta"] = gantt["code"] + " · " + gantt["title"].str.slice(0, 42)
        gantt["Cumplimiento"] = gantt["Semáforo"]

        fig_gantt = px.timeline(
            gantt.sort_values(["Final", "priority"]),
            x_start="Inicio",
            x_end="Final",
            y="Etiqueta",
            color="Cumplimiento",
            hover_data={
                "assignee": True,
                "progress": ":.0f",
                "Teórico %": ":.0f",
                "priority": True,
                "Inicio": "|%d/%m/%Y",
                "Final": "|%d/%m/%Y",
            },
        )
        fig_gantt.update_yaxes(autorange="reversed", title=None)
        fig_gantt.update_xaxes(title="Cronograma")
        fig_gantt.add_vline(
            x=pd.Timestamp(date.today()).timestamp() * 1000,
            line_width=1,
            line_dash="dash",
        )
        fig_gantt.update_layout(
            height=max(420, min(900, 36 * len(gantt) + 160)),
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#183D2D"),
        )
        st.plotly_chart(fig_gantt, use_container_width=True)
    else:
        st.info("No hay tareas con fecha de inicio y finalización para mostrar en Gantt.")

    st.markdown("### Tareas que requieren atención")
    attention = view[
        view["Semáforo"].astype(str).str.contains("🔴|🟡")
    ].copy()

    if attention.empty:
        st.success("No hay tareas con alerta en el filtro seleccionado.")
    else:
        st.dataframe(
            attention[
                [
                    "Semáforo","code","title","assignee","priority","start_date",
                    "due_date","progress","Teórico %","Desvío pp"
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("### Solicitudes por mes")
    monthly = view.copy()
    monthly["Mes"] = pd.to_datetime(monthly["requested"], errors="coerce").dt.to_period("M").astype(str)
    monthly = monthly.groupby("Mes").size().reset_index(name="Tareas")
    if not monthly.empty:
        fig_month = px.line(monthly, x="Mes", y="Tareas", markers=True)
        fig_month.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_month, use_container_width=True)

# ============================================================
# NUEVA TAREA
# ============================================================

elif page == "Nueva tarea":
    st.subheader("Nueva tarea")

    sector_name = st.selectbox("Sector", list(SECTORES))
    area_name = st.selectbox("Área / familia", list(AREAS))
    maintenance_type = None
    if SECTORES[sector_name] == "MANT":
        maintenance_type = st.selectbox("Tipo de mantenimiento", TIPOS_MANT)

    with st.form("new_task"):
        title = st.text_input("Tarea")
        description = st.text_area("Descripción")

        a, b = st.columns(2)
        priority = a.selectbox("Prioridad", PRIORIDADES)
        assignee_id = b.selectbox(
            "Responsable",
            people["id"].tolist(),
            format_func=lambda x: people.loc[people["id"] == x, "name"].iloc[0],
        )

        a, b = st.columns(2)
        start = a.date_input("Fecha de inicio", format="DD/MM/YYYY")
        due = b.date_input("Fecha de finalización", format="DD/MM/YYYY")

        recurrence = st.selectbox("Recurrencia", RECURRENCIAS)
        observation = st.text_area("Observación")

        submit = st.form_submit_button(
            "Crear y asignar tarea",
            type="primary",
            use_container_width=True,
        )

    if submit:
        if not title.strip():
            st.error("Ingresa el nombre de la tarea.")
        elif due < start:
            st.error("La fecha final no puede ser anterior a la fecha de inicio.")
        else:
            sec = SECTORES[sector_name]
            area = AREAS[area_name]
            code = next_code(sec, area)
            task_token = secrets.token_urlsafe(24)

            c.execute(
                """
                INSERT INTO tasks(
                    code,title,description,sector,area,maintenance_type,assignee_id,
                    priority,requested,start_date,due_date,status,progress,observation,
                    token,recurrence,recurrence_day,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    code,
                    title.strip(),
                    description.strip(),
                    sec,
                    area,
                    maintenance_type,
                    int(assignee_id),
                    priority,
                    date.today().isoformat(),
                    start.isoformat(),
                    due.isoformat(),
                    "Asignada",
                    0,
                    observation.strip(),
                    task_token,
                    None if recurrence == "No" else recurrence,
                    due.day,
                    datetime.now().isoformat(),
                ),
            )
            c.commit()
            st.success(f"Tarea creada: {code}")
            st.write("Enlace del responsable:")
            st.code(f"?token={task_token}", language=None)

# ============================================================
# TAREAS
# ============================================================

elif page == "Tareas":
    st.subheader("Tareas")
    view = tasks.copy()
    view["Teórico %"] = view.apply(theoretical, axis=1)
    view["Semáforo"] = view.apply(traffic_light, axis=1)
    st.dataframe(
        view[
            [
                "Semáforo","code","title","sector","area","maintenance_type",
                "assignee","priority","status","start_date","due_date","progress",
                "Teórico %","observation"
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

# ============================================================
# CALENDARIO / GANTT
# ============================================================

elif page == "Calendario / Gantt":
    st.subheader("Calendario y Gantt")

    gantt = tasks[
        tasks["start_date"].notna()
        & (tasks["start_date"] != "")
        & tasks["due_date"].notna()
        & (tasks["due_date"] != "")
    ].copy()

    if gantt.empty:
        st.info("No existen tareas con cronograma.")
    else:
        gantt["Inicio"] = pd.to_datetime(gantt["start_date"])
        gantt["Final"] = pd.to_datetime(gantt["due_date"])
        gantt["Semáforo"] = gantt.apply(traffic_light, axis=1)
        gantt["Teórico %"] = gantt.apply(theoretical, axis=1)
        gantt["Etiqueta"] = gantt["code"] + " · " + gantt["title"].str.slice(0, 55)

        fig = px.timeline(
            gantt.sort_values("Final"),
            x_start="Inicio",
            x_end="Final",
            y="Etiqueta",
            color="Semáforo",
            hover_data={
                "assignee": True,
                "priority": True,
                "progress": ":.0f",
                "Teórico %": ":.0f",
                "Inicio": "|%d/%m/%Y",
                "Final": "|%d/%m/%Y",
            },
        )
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_layout(
            height=max(500, min(1000, 38 * len(gantt) + 150)),
            margin=dict(l=10,r=10,t=20,b=10),
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Cronograma detallado")
        st.dataframe(
            gantt[
                [
                    "Semáforo","code","title","assignee","Inicio","Final",
                    "priority","progress","Teórico %"
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

# ============================================================
# RECURRENTES
# ============================================================

elif page == "Recurrentes":
    st.subheader("Tareas recurrentes")
    recurrent = tasks[
        tasks["recurrence"].notna()
        & (tasks["recurrence"] != "No")
    ]
    st.dataframe(
        recurrent[
            [
                "code","title","sector","area","assignee","recurrence",
                "recurrence_day","due_date","status"
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Las nuevas ocurrencias se generan dentro de una ventana de 45 días.")

# ============================================================
# MANTENIMIENTO
# ============================================================

elif page == "Mantenimiento":
    st.subheader("Mantenimiento")
    maint = tasks[tasks["sector"] == "MANT"].copy()

    if maint.empty:
        st.info("Todavía no existen tareas de mantenimiento.")
    else:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Preventivo", int((maint["maintenance_type"] == "Preventivo").sum()))
        q2.metric("Correctivo", int((maint["maintenance_type"] == "Correctivo").sum()))
        q3.metric("Proactivo", int((maint["maintenance_type"] == "Proactivo").sum()))
        q4.metric("Predictivo", int((maint["maintenance_type"] == "Predictivo").sum()))

        grouped = maint.groupby(["maintenance_type","status"]).size().reset_index(name="Tareas")
        fig = px.bar(grouped, x="maintenance_type", y="Tareas", color="status", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            maint[
                [
                    "code","title","area","maintenance_type","assignee",
                    "priority","status","progress","start_date","due_date"
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

# ============================================================
# OPERARIOS
# ============================================================

elif page == "Operarios":
    st.subheader("Operarios")
    summary = tasks.groupby("assignee").agg(
        Tareas=("id","count"),
        Abiertas=("status", lambda s: (s != "Cerrada").sum()),
        Cerradas=("status", lambda s: (s == "Cerrada").sum()),
        Avance_promedio=("progress","mean"),
    ).reset_index()

    summary["Avance_promedio"] = summary["Avance_promedio"].fillna(0).round(1)
    st.dataframe(summary, hide_index=True, use_container_width=True)

# ============================================================
# CIERRES
# ============================================================

elif page == "Cierres pendientes":
    st.subheader("Cierres pendientes")
    pending = tasks[tasks["status"] == "Terminada - espera cierre"]

    if pending.empty:
        st.success("No hay tareas esperando cierre administrativo.")
    else:
        for _, r in pending.iterrows():
            with st.container(border=True):
                st.write(f"**{r.code} · {r.title}**")
                st.write(f"Responsable: {r.assignee} · Avance: {float(r.progress or 0):.0f}%")
                if st.button("Aprobar cierre", key=f"close_{r.id}", type="primary"):
                    c.execute(
                        "UPDATE tasks SET status='Cerrada',closed_at=? WHERE id=?",
                        (datetime.now().isoformat(), int(r.id)),
                    )
                    c.commit()
                    st.rerun()

c.close()
