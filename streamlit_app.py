import altair as alt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Movies dataset", page_icon="🎬")
st.title("🎬 Movies dataset")
st.write(
    """
    This app visualizes data from [The Movie Database (TMDB)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
    It shows which movie genre performed best at the box office over the years. Just 
    click on the widgets below to explore!
    """
)


# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    df = pd.read_csv("data/movies_genres_summary.csv")
    return df


df = load_data()

# Show a multiselect widget with the genres using `st.multiselect`.
genres = st.multiselect(
    "Genres",
    df.genre.unique(),
    ["Action", "Adventure", "Biography", "Comedy", "Drama", "Horror"],
)

# Show a slider widget with the years using `st.slider`.
years = st.slider("Years", 1986, 2006, (2000, 2016))

# Filter the dataframe based on the widget input and reshape it.
df_filtered = df[(df["genre"].isin(genres)) & (df["year"].between(years[0], years[1]))]
df_reshaped = df_filtered.pivot_table(
    index="year", columns="genre", values="gross", aggfunc="sum", fill_value=0
)
df_reshaped = df_reshaped.sort_values(by="year", ascending=False)


# Display the data as a table using `st.dataframe`.
st.dataframe(
    df_reshaped,
    use_container_width=True,
    column_config={"year": st.column_config.TextColumn("Year")},
)

# Display the data as an Altair chart using `st.altair_chart`.
df_chart = pd.melt(
    df_reshaped.reset_index(), id_vars="year", var_name="genre", value_name="gross"
)
chart = (
    alt.Chart(df_chart)
    .mark_line()
    .encode(
        x=alt.X("year:N", title="Year"),
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


st.divider()
st.subheader("🔮 Previsione Trend Futuri (2017 - 2030)")

# 1. Definiamo l'intervallo futuro (dal 2017 al 2030)
future_years = np.array(range(2017, 2031)).reshape(-1, 1)

if not df_filtered.empty:
    prediction_list = []

    for genre in genres:
        # Otteniamo i dati storici filtrati per quel genere
        genre_data = df_filtered[df_filtered['genre'] == genre].groupby('year')['gross'].sum().reset_index()
        
        # Il modello ha bisogno di almeno due punti storici per tracciare una linea
        if len(genre_data) >= 2:
            X = genre_data[['year']].values
            y = genre_data['gross'].values
            
            # Allenamento del modello
            model = LinearRegression()
            model.fit(X, y)
            
            # Predizione per il periodo 2017-2030
            preds = model.predict(future_years)
            
            for yr, p in zip(future_years.flatten(), preds):
                # max(0, p) impedisce incassi negativi se il trend è in forte calo
                prediction_list.append({"year": int(yr), "genre": genre, "gross": max(0, p), "type": "Previsione"})

    # 2. Prepariamo i dati per il grafico unico
    df_historical = df_filtered[['year', 'genre', 'gross']].copy()
    df_historical['type'] = 'Storico'
    
    df_predictions = pd.DataFrame(prediction_list)
    
    # Uniamo passato e futuro
    df_final = pd.concat([df_historical, df_predictions]).sort_values('year')

    # 3. Creazione del Grafico Altair
    forecast_chart = (
        alt.Chart(df_final)
        .mark_line(point=True) # Aggiungiamo i punti per chiarezza
        .encode(
            x=alt.X("year:O", title="Anno"), # :O tratta l'anno come ordinale (senza virgole)
            y=alt.Y("gross:Q", title="Incassi ($)"),
            color="genre:N",
            strokeDash=alt.condition(
                alt.datum.type == 'Previsione', 
                alt.value([5, 5]), # Tratteggio per il futuro
                alt.value([0])     # Linea continua per il passato
            ),
            tooltip=["year", "genre", "gross", "type"]
        )
        .properties(height=450)
        .interactive() # Permette zoom e spostamento
    )
    
    st.altair_chart(forecast_chart, use_container_width=True)

    # 4. Analisi dei risultati
    if not df_predictions.empty:
        # Troviamo il vincitore nell'ultimo anno (2030)
        last_year_preds = df_predictions[df_predictions['year'] == 2030]
        winner = last_year_preds.loc[last_year_preds['gross'].idxmax()]
        
        st.success(f"🏆 Il genere che dominerà il mercato nel **2030** sarà **{winner['genre']}** con un incasso stimato di **${winner['gross']:,.2f}**")
        
        with st.expander("Vedi i dati grezzi delle previsioni"):
            st.dataframe(df_predictions.pivot(index='year', columns='genre', values='gross'))
else:
    st.info("Seleziona i generi e l'intervallo temporale per generare la proiezione dal 2017 in poi.")
