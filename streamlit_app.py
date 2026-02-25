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
st.subheader("🔮 Previsione Trend Futuri (2025-2030)")

# 1. Prepariamo i dati per il modello
# Usiamo i dati filtrati dall'utente (df_filtered)
if not df_filtered.empty:
    future_years = np.array(range(2025, 2031)).reshape(-1, 1)
    prediction_list = []

    for genre in genres:
        # Filtriamo per singolo genere
        genre_data = df_filtered[df_filtered['genre'] == genre].groupby('year')['gross'].sum().reset_index()
        
        if len(genre_data) > 1: # Servono almeno 2 punti per una linea
            X = genre_data[['year']].values
            y = genre_data['gross'].values
            
            # Creiamo e addestriamo il modello (Regressione Lineare)
            model = LinearRegression()
            model.fit(X, y)
            
            # Prediciamo il futuro
            preds = model.predict(future_years)
            
            # Salviamo i risultati
            for yr, p in zip(future_years.flatten(), preds):
                prediction_list.append({"year": yr, "genre": genre, "gross": max(0, p), "type": "Previsione"})

    # 2. Uniamo i dati storici con le previsioni per il confronto
    df_historical = df_filtered[['year', 'genre', 'gross']].copy()
    df_historical['type'] = 'Storico'
    
    df_predictions = pd.DataFrame(prediction_list)
    df_final = pd.concat([df_historical, df_predictions])

    # 3. Visualizzazione Grafica con Altair
    # Usiamo il tratteggio per distinguere le previsioni
    forecast_chart = (
        alt.Chart(df_final)
        .mark_line()
        .encode(
            x=alt.X("year:N", title="Anno"),
            y=alt.Y("gross:Q", title="Incassi Stimati ($)"),
            color="genre:N",
            strokeDash=alt.condition(
                alt.datum.type == 'Previsione', 
                alt.value([5, 5]),  # Linea tratteggiata per il futuro
                alt.value([0])      # Linea continua per il passato
            )
        )
        .properties(height=400)
    )
    
    st.altair_chart(forecast_chart, use_container_width=True)
    
    # 4. Verdetto finale
    if prediction_list:
        latest_preds = df_predictions[df_predictions['year'] == 2030]
        winner = latest_preds.loc[latest_preds['gross'].idxmax(), 'genre']
        st.success(f"Basandosi sui trend attuali, il genere più redditizio nel 2030 sarà: **{winner}**")
else:
    st.warning("Seleziona almeno un genere e un intervallo di anni per generare la previsione.")


