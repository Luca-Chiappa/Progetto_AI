import streamlit as st
import cinema_page
import libri_page   

st.set_page_config(page_title="DatasetView", page_icon="🎬")
st.title("🎬 DatasetView 📗" )

#selezione del dataset da visualizzare

def gestisci_modalità(tipo_dataset):
    if tipo_dataset == "🎬 Film":
        cinema_page.show_cinema()
    else:
        libri_page.show_libri() 


st.sidebar.title("Impostazioni")
modalita = st.sidebar.radio(
    "Seleziona il Dataset:",
    ["🎬 Film", "📚 Libri"],
    help="Passa dall'analisi del cinema a quella dell'editoria"
)

gestisci_modalità(modalita)



