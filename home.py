# Home.py
import streamlit as st

st.set_page_config(page_title="🏠 Home", layout="wide")

st.title("🏠 Benvenuto nella tua app Streamlit")
st.markdown("""
Questa è la **home page**. Usa il menu laterale o la cartella `pages/` per accedere
alle varie funzionalità, ad esempio:
- 📊 Analisi COP
- 📈 Grafici temporali
- 🔎 IPMVP Evaluation
- 🩻 DICOM Viewer & Manager
""")
