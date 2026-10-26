from datetime import datetime

import pandas as pd
import streamlit as st

from utils.history import incarca_toate_sesiunile

st.set_page_config(page_title="Istoric prezenta", layout="wide")

st.title("Istoric prezenta")
sesiuni = incarca_toate_sesiunile()

if not sesiuni:
    st.info("Nu exista sesiuni salvate. Ruleaza o recunoastere din pagina principala.")
    st.stop()

st.subheader("Rezumat")

total_sesiuni = len(sesiuni)
zile = []
for s in sesiuni:
    if "data" in s:
        zi = s["data"]
    else:
        zi = s["data_ora"].split(" ")[0]
    zile.append(zi)

total_zile = len(set(zile))

prezenti_unici = set()
for s in sesiuni:
    persoane = s.get("persoane_prezente", {})
    for nume in persoane:
        prezenti_unici.add(nume)

total_prezenti = len(prezenti_unici)

if total_sesiuni > 0:
    medie_prezenti = total_prezenti / total_sesiuni
else:
    medie_prezenti = 0

col1, col2, col3 = st.columns(3)
col1.metric("Total zile", total_zile)
col2.metric("Total prezenti", total_prezenti)
col3.metric("Medie prezenti/sesiune", f"{medie_prezenti:.1f}")

st.subheader("Tabel pe zile")

prezenta_pe_zi = {}
for s in sesiuni:
    if "data" in s:
        zi = s["data"]
    else:
        zi = s["data_ora"].split(" ")[0]
    if zi not in prezenta_pe_zi:
        prezenta_pe_zi[zi] = set()

    persoane = s.get("persoane_prezente", {})
    for nume in persoane:
        prezenta_pe_zi[zi].add(nume)

zile_sortate = []
for zi in prezenta_pe_zi:
    zile_sortate.append(zi)

def ia_data(zi):
    return datetime.strptime(zi, "%d.%m.%Y")

zile_sortate.sort(key=ia_data)

randuri = []
for zi in zile_sortate:
    randuri.append({
        "Zi": zi,
        "Prezenti": len(prezenta_pe_zi[zi]),
    })

df = pd.DataFrame(randuri)

st.dataframe(df, width="stretch", hide_index=True)

