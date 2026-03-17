import altair as alt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import ast

st.set_page_config(page_title="Movies dataset", page_icon="🎬")
st.title("🎬 Movies dataset")
st.write(
    """
    This app visualizes data from [The Movie Database (TMDB)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
    It shows which movie genre performed best at the box office over the years. Just 
    click on the widgets below to explore!
    """
)

# --- CARICAMENTO TMDB E CHATBOT ---
@st.cache_data
def load_tmdb_data():
    df = pd.read_csv("/workspaces/Progetto_AI/data/tmdb_5000_movies.csv")
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["genres"] = df["genres"].apply(lambda x: [g["name"] for g in ast.literal_eval(x)])
    return df

df_tmdb = load_tmdb_data()

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

st.markdown("""
<style>
.red-round-btn {
    background-color: #d62828;
    color: white;
    width: 55px;
    height: 55px;
    border: none;
    border-radius: 50%;
    font-size: 24px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: 0.25s;
}
.red-round-btn:hover {
    background-color: #b71c1c;
}
</style>
""", unsafe_allow_html=True)

clicked = st.button("🤖", key="robot_btn")

if clicked:
    st.session_state.chat_open = not st.session_state.chat_open

def richiesta(domanda, df):
    q = domanda.lower()
    anno = None
    for y in range(1980, 2030):
        if str(y) in q:
            anno = y
            break

    if "genere" in q and ("più visto" in q or "più guardato" in q or "popolare" in q):
        if anno is None:
            return "Dimmi anche l'anno, così posso cercare il genere più guardato."
        df_year = df[df["year"] == anno]
        if df_year.empty:
            return f"Non ho dati per l'anno {anno}."
        exploded = df_year.explode("genres")
        top_genre = exploded.groupby("genres")["popularity"].sum().idxmax()
        return f"Nel {anno}, il genere più popolare è stato: **{top_genre}**."

    return "Non ho capito la domanda, prova a riformularla."

if st.session_state.chat_open:
    st.subheader("Chatbot")
    domanda = st.text_input("Fai una domanda sul dataset:")
    if domanda:
        st.write(richiesta(domanda, df_tmdb))
 

# --- PRIMA PARTE - ANALISI DEI DATI ---
@st.cache_data
def load_summary_data():
    df = pd.read_csv("data/movies_genres_summary.csv")
    return df

df = load_summary_data()

genres = st.multiselect(
    "Genres",
    df.genre.unique(),
    ["Action", "Adventure", "Biography", "Comedy", "Drama", "Horror"],
)

years = st.slider("Years", 1986, 2006, (2000, 2016))

df_filtered = df[(df["genre"].isin(genres)) & (df["year"].between(years[0], years[1]))]
with st.expander("Vedi i dati grezzi"):
    df_reshaped = df_filtered.pivot_table(
        index="year", columns="genre", values="gross", aggfunc="sum", fill_value=0
    ).sort_values(by="year", ascending=False)
    st.dataframe(
        df_reshaped,
        use_container_width=True,
        column_config={"year": st.column_config.TextColumn("Year")},
    )

df_chart = pd.melt(
    df_reshaped.reset_index(), id_vars="year", var_name="genre", value_name="gross"
)
chart = (
    alt.Chart(df_chart)
    .mark_line(point=True)
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("gross:Q", title="Gross earnings ($)"),
        color="genre:N",
    )
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)

st.write(
    """
    Utilizzando i dati a disposizione come fonte, possiamo fare delle previsioni sulle tendenze 
    future degli incassi cinematografici
    """
)


# --- SECONDA PARTE - PREVISIONE ---
st.divider()
st.subheader("🔮 Previsione Trend Futuri (2017 - 2030)")

years_viz = st.slider("Seleziona anni da visualizzare nel grafico", 1986, 2030, (1995, 2030))

seleziona_tutto = st.checkbox("Seleziona tutti i generi per la previsione")
opzioni_disponibili = df.genre.unique()

# FIX: Gestione corretta dei default per evitare l'errore "DuplicateWidgetID"
default_genres = opzioni_disponibili if seleziona_tutto else ["Action", "Adventure"]

genres_prediction = st.multiselect(
    "Scegli i generi per la previsione",
    opzioni_disponibili,
    default=default_genres 
)

df_prevision = df[(df["genre"].isin(genres_prediction))]

if not df_prevision.empty:
    future_years = np.array(range(2017, 2031)).reshape(-1, 1)
    prediction_list = []

    # FIX: Usiamo 'genres_prediction' invece di 'genres'
    for genre in genres_prediction:
        # FIX: Peschiamo lo storico intero da df_prevision, non da df_filtered che era bloccato dallo slider
        genre_data = df_prevision[df_prevision['genre'] == genre].groupby('year')['gross'].sum().reset_index()
        
        if len(genre_data) >= 2:
            X = genre_data[['year']].values
            y = genre_data['gross'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            preds = model.predict(future_years)
            
            for yr, p in zip(future_years.flatten(), preds):
                prediction_list.append({"year": int(yr), "genre": genre, "gross": max(0, p), "type": "Previsione"})

    # FIX: Anche qui, usiamo df_prevision per tracciare lo storico corretto
    df_historical = df_prevision[['year', 'genre', 'gross']].groupby(['year', 'genre']).sum().reset_index()
    df_historical['type'] = 'Storico'
    
    df_predictions = pd.DataFrame(prediction_list)
    df_all_data = pd.concat([df_historical, df_predictions])

    df_final = df_all_data[df_all_data["year"].between(years_viz[0], years_viz[1])]
    
    forecast_chart = (
        alt.Chart(df_final)
        .mark_line(point=True) 
        .encode(
            x=alt.X("year:O", title="Anno"),
            y=alt.Y("gross:Q", title="Incassi Stimati ($)"),
            color="genre:N",
            strokeDash=alt.condition(
                alt.datum.type == 'Previsione', 
                alt.value([5, 5]), 
                alt.value([0])     
            ),
            tooltip=["year", "genre", "gross", "type"]
        )
        .properties(height=320)
        .interactive() 
    )
    
    st.altair_chart(forecast_chart, use_container_width=True)

    if not df_predictions.empty:
        max_year = df_predictions["year"].max()
        last_year_preds = df_predictions[df_predictions['year'] == max_year]
        
        # Prevenzione di errori se l'array fosse vuoto
        if not last_year_preds.empty:
            winner = last_year_preds.loc[last_year_preds['gross'].idxmax()]
            st.success(f"🏆 Il genere che dominerà il mercato nel **2030** sarà **{winner['genre']}** con un incasso stimato di **${winner['gross']:,.2f}**")
        
        with st.expander("Vedi i dati grezzi delle previsioni"):
            try:
                st.dataframe(df_predictions.pivot(index='year', columns='genre', values='gross'))
            except ValueError:
                st.dataframe(df_predictions)
else:
    st.info("Seleziona i generi e l'intervallo temporale per generare la proiezione dal 2017 in poi.")