import streamlit as st,pandas as pd,plotly.graph_objects as go
from datetime import date,timedelta
from src.ina import observed,forecast_meta,STATIONS
from src.model import train,predict,prob
from src.scenario import run

st.set_page_config(page_title="Paraná San Nicolás | Pronóstico del río",page_icon="🌊",layout="wide")
st.title("🌊 PARANÁ · SAN NICOLÁS")
st.caption("V9 · Plataforma pública de monitoreo y predicción experimental")
if "df" not in st.session_state:st.session_state.df=None
if "pred" not in st.session_state:st.session_state.pred={}
if "met" not in st.session_state:st.session_state.met={}

with st.sidebar:
 st.header("Consulta online")
 start=st.date_input("Desde",date.today()-timedelta(days=365*3));end=st.date_input("Hasta",date.today())
 if st.button("🔄 Actualizar INA",type="primary",use_container_width=True):
  with st.spinner("Consultando estaciones..."):
   try:
    st.session_state.df,st.session_state.errors=observed(start.isoformat(),end.isoformat());st.session_state.pred={};st.session_state.met={};st.success("Datos actualizados")
   except Exception as e:st.error(str(e))
 st.divider();st.write("**Objetivo:** San Nicolás de los Arroyos");st.caption("Fuente hidrométrica: INA")

if st.session_state.df is None:
 st.info("Presione **Actualizar INA** para iniciar la consulta online.");st.stop()
df=st.session_state.df.copy();df["datetime"]=pd.to_datetime(df.datetime,utc=True);df=df.sort_values("datetime")
v=df.dropna(subset=["San Nicolás"]);last=v.iloc[-1];prev=v.iloc[-7] if len(v)>7 else v.iloc[0]
tabs=st.tabs(["📍 Estado","🔮 Pronóstico","🌧️ Lluvia","🚦 Riesgo","ℹ️ Metodología"])

with tabs[0]:
 a,b,c,d=st.columns(4);a.metric("Nivel San Nicolás",f"{last['San Nicolás']:.2f} m");b.metric("Cambio 7 h",f"{last['San Nicolás']-prev['San Nicolás']:+.2f} m")
 c.metric("Villa Constitución",f"{df['Villa Constitución'].dropna().iloc[-1]:.2f} m" if "Villa Constitución" in df else "—");d.metric("Rosario",f"{df['Rosario'].dropna().iloc[-1]:.2f} m" if "Rosario" in df else "—")
 fig=go.Figure()
 for s in ["Corrientes","Goya","La Paz","Paraná","Diamante","Rosario","Villa Constitución","San Nicolás"]:
  if s in df:fig.add_trace(go.Scatter(x=df.datetime,y=df[s],name=s,mode="lines"))
 fig.update_layout(height=520,hovermode="x unified",yaxis_title="Nivel")
 st.plotly_chart(fig,use_container_width=True)
 latest=[{"Estación":s,"Nivel (m)":round(float(df[s].dropna().iloc[-1]),2)} for s in STATIONS if s in df and df[s].notna().any()]
 st.dataframe(pd.DataFrame(latest),hide_index=True,use_container_width=True)

with tabs[1]:
 st.info("Predicción estadística basada en niveles y tendencias aguas arriba. No reemplaza el pronóstico oficial.")
 if st.button("🧠 Ejecutar modelo +24/+48/+72",type="primary"):
  with st.spinner("Entrenando..."):
   models,met=train(df);st.session_state.pred=predict(df,models);st.session_state.met=met
 pred=st.session_state.pred;met=st.session_state.met;cc=st.columns(3)
 for i,h in enumerate([24,48,72]):cc[i].metric(f"+{h} h","—" if h not in pred else f"{pred[h]:.2f} m")
 if met:st.dataframe(pd.DataFrame(met).T.round(3),use_container_width=True)
 st.divider()
 if st.button("Consultar pronóstico publicado por INA"):
  try:st.json(forecast_meta())
  except Exception as e:st.error(str(e))

with tabs[2]:
 st.warning("V9: módulo de sensibilidad. Todavía no es un modelo físico lluvia→escorrentía.")
 rain={}
 for col,n,val in zip(st.columns(3),["Corrientes","Goya","Reconquista"],[100.,150.,100.]):rain[n]=col.number_input(n+" 72h (mm)",0.,1000.,val)
 for col,n,val in zip(st.columns(3),["Esquina","La Paz","Paraná"],[80.,80.,50.]):rain[n]=col.number_input(n+" 72h (mm)",0.,1000.,val)
 if st.button("Calcular impacto"):
  r=run(float(last["San Nicolás"]),rain);cc=st.columns(3)
  for i,k in enumerate(["Bajo","Central","Alto"]):cc[i].metric(k,f"{r[k]:.2f} m")

with tabs[3]:
 st.subheader("Semáforo experimental de riesgo")
 st.caption("Debe configurarse con umbrales oficiales/locales antes de utilizarse como alerta.")
 threshold=st.number_input("Umbral de referencia (m)",0.,20.,float(last["San Nicolás"]+.5),.1)
 if st.session_state.pred:
  rows=[]
  for h,p in st.session_state.pred.items():
   rmse=st.session_state.met.get(h,{}).get("RMSE",.2);pp=prob(p,threshold,rmse)
   estado="ALTO" if pp>=.75 else "MEDIO" if pp>=.40 else "BAJO"
   rows.append({"Horizonte":f"+{h} h","Predicción":round(p,2),"Probabilidad":f"{pp*100:.0f}%","Estado":estado})
  st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
 else:st.info("Ejecute primero el modelo en la pestaña Pronóstico.")

with tabs[4]:
 st.markdown("""
**V9** agrega una interfaz de riesgo y mantiene separadas observaciones INA,
modelo estadístico propio y escenarios de lluvia.

El modelo utiliza rezagos de estaciones aguas arriba y cambios de nivel, con
validación temporal 80/20.

La probabilidad de riesgo es experimental y depende del error del modelo.
Los escenarios de lluvia no están calibrados como lluvia→escorrentía.

La V9 debería integrar precipitación espacial, pronóstico meteorológico,
caudales/erogaciones, tiempos de propagación y calibración por eventos
históricos. El sistema no es una alerta oficial.
""")
st.download_button("⬇️ Descargar datos CSV",df.to_csv(index=False).encode(),"parana_san_nicolas_v08.csv","text/csv")
st.divider();st.caption("PARANÁ · SAN NICOLÁS V9 · Consulta online")
