import io
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.detector import align_face
from core.embeddings import get_embedding
from core.recognizer import get_known_classes, predict
from utils.charts import grafic_bara, grafic_pie
from utils.csv_export import genereaza_csv, sumarizeaza
from utils.history import salveaza_sesiune

st.set_page_config(page_title="Acasa", layout="wide")

st.title("Sistem de Identificare a Persoanelor prin Recunoastere Faciala ")

with st.sidebar:
    tip_grafic = st.radio("Tip grafic", options=["Bare", "Pie"], index=0)

uploaded_files = st.file_uploader(
    "Poze (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)


if st.button("Ruleaza", type="primary"):
    detected = {}
    necunoscuti = 0
    erori = []
    rezultate_ui = []

    all_classes = get_known_classes()
    total = len(uploaded_files)

    for i, file in enumerate(uploaded_files):

        continut = file.read()
        img_bytes = np.frombuffer(continut, np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            erori.append(f"{file.name}: nu s-a putut citi")
            continue

        fata_rgb = align_face(img_bgr)
        if fata_rgb is None:
            rezultate_ui.append({
                "file": file.name,
                "img": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                "status": "Fata negasita",
                "name": None,
                "conf": None,
            })
            continue

        try:
            emb = get_embedding(fata_rgb)
            pred = predict(emb)
        except Exception as e:
            erori.append(f"{file.name}: {e}")
            continue

        ora = datetime.now().strftime("%H:%M:%S")

        if pred["known"]:
            name = pred["name"]
            if name not in detected or pred["confidence"] > detected[name]["confidence"]:
                detected[name] = {
                    "confidence": pred["confidence"],
                    "time": ora,
                    "img_rgb": fata_rgb,
                }
            rezultate_ui.append({
                "file": file.name,
                "img": fata_rgb,
                "status": name,
                "name": name,
                "conf": pred["confidence"],
            })
        else:
            necunoscuti += 1
            rezultate_ui.append({
                "file": file.name,
                "img": fata_rgb,
                "status": "Necunoscut",
                "name": None,
                "conf": pred["confidence"],
            })


    st.subheader("Rezultate per poza")
    cols_per_row = 4
    rows = []
    for j in range(0, len(rezultate_ui), cols_per_row):
        rows.append(rezultate_ui[j:j + cols_per_row])

    for row in rows:
        cols = st.columns(cols_per_row)
        for i in range(len(row)):
            col = cols[i]
            r = row[i]
            with col:
                st.image(r["img"], width="stretch")
                st.caption(r["file"])
                st.markdown(f"**{r['status']}**")
                if r["conf"] is not None:
                    st.caption(f"Confidence: {r['conf']:.1%}")

    st.markdown("---")

    stats = sumarizeaza(detected, all_classes, necunoscuti)

    col_stat, col_chart = st.columns([1, 2])
    with col_stat:
        st.subheader("Statistici")
        st.metric("Prezenti", stats["prezenti"])
        st.metric("Absenti", stats["absenti"])
        st.metric("Necunoscuti", stats["necunoscuti"])
        st.metric("Poze totale", len(uploaded_files))

    with col_chart:
        st.subheader("Grafic")
        if tip_grafic == "Bare":
            fig = grafic_bara(stats)
        else:
            fig = grafic_pie(stats)
        st.pyplot(fig)

    if erori:
        with st.expander(f"{len(erori)} erori"):
            for e in erori:
                st.text(e)

    st.markdown("---")
    st.subheader("Descarca raportul")

    filename, csv_bytes = genereaza_csv(detected, all_classes)
    st.download_button(
        label="Descarca CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )

    with st.expander("Preview CSV"):
        preview_csv = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")))
        st.dataframe(preview_csv, width="stretch", hide_index=True)

    salveaza_sesiune(detected, all_classes, necunoscuti, len(uploaded_files))
    st.success("Sesiunea a fost salvata in Istoric.")

    if detected:
        st.markdown("---")
        st.subheader("Persoane recunoscute")
        cols = st.columns(3)
        persoane = sorted(detected.items())
        for j in range(len(persoane)):
            name, info = persoane[j]
            with cols[j % 3]:
                st.image(info["img_rgb"], width=120)
                st.markdown(f"**{name}**")
                st.caption(f"Confidence: {info['confidence']:.1%}")
