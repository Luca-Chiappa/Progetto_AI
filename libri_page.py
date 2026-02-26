import altair as alt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

def show_cinema():
    @st.write(
        """
        Quest'app analizza i dati storici degli incassi e delle valutazione di ogni genere di {modalita}]
        """
    )

    # PRIMA PARTE - ANALISI DEI DATI
    @st.cache_data
    def load_data():
        df = pd.read_csv("data/movies_genres_summary.csv")
        return df

    df = load_data()

    genres = st.multiselect(
        "Genres",
        df.genre.unique(),
        ["Action", "Adventure", "Biography", "Comedy", "Drama", "Horror"],
    )

    years = st.slider("Years", 1986, 2006, (2000, 2016))

    df_filtered = df[(df["genre"].isin(genres)) & (df["year"].between(years[0], years[1]))]
    df_reshaped = df_filtered.pivot_table(
        index="year", columns="genre", values="gross", aggfunc="sum", fill_value=0
    )
    df_reshaped = df_reshaped.sort_values(by="year", ascending=False)

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
        .mark_line()
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


    # SECONDA PARTE - PREVISIONE
    st.divider()
    st.subheader("🔮 Previsione Trend Futuri (2017 - 2030)")

    years_viz = st.slider("Seleziona anni da visualizzare nel grafico", 1986, 2030, (1995, 2030))

    seleziona_tutto = st.checkbox("Seleziona tutti i generi per la previsione")
    opzioni_disponibili = df.genre.unique()

    if seleziona_tutto:
        genres_prediction = st.multiselect(
            "Scegli i generi per la previsione",
            opzioni_disponibili,
            default=opzioni_disponibili 
        )
    else:
        genres_prediction = st.multiselect(
            "Scegli i generi per la previsione",
            opzioni_disponibili,
            default=["Action", "Adventure"] # Default standard se la checkbox è falsa
        )

    df_prevision = df[(df["genre"].isin(genres_prediction))]

    if not df_prevision.empty:
        future_years = np.array(range(2016, 2030)).reshape(-1, 1)
        prediction_list = []

        for genre in genres_prediction:
            # Otteniamo i dati storici filtrati per quel genere
            genre_data = df_prevision[df_prevision['genre'] == genre].groupby('year')['gross'].sum().reset_index()
            
            # Il modello ha bisogno di almeno due punti storici per tracciare una linea
            if len(genre_data) >1:
                X = genre_data[['year']].values
                y = genre_data['gross'].values
                
                # Allenamento del modello
                model = LinearRegression()
                model.fit(X, y)
                
                # Predizione per il periodo 2017-2030
                preds = model.predict(future_years)
                
                for yr, p in zip(future_years.flatten(), preds):
                    # max(0, p) impedisce incassi negativi se il trend è in forte calo
                    prediction_list.append({"year": int(yr), 
                                            "genre": genre, 
                                            "gross": max(0, p), 
                                            "type": "Previsione"})

        # 2. Prepariamo i dati per il grafico unico
        df_historical = df_prevision[['year', 'genre', 'gross']].copy()
        df_historical['type'] = 'Storico'
        
        df_predictions = pd.DataFrame(prediction_list)
        df_all_data = pd.concat([df_historical, df_predictions])

        df_final = df_all_data[df_all_data["year"].between(years_viz[0], years_viz[1])].sort_values(by="year")
        
        forecast_chart = (
            alt.Chart(df_final)
            .mark_line(point=True) # Aggiungiamo i punti per chiarezza
            .encode(
                x=alt.X("year:O", title="Anno"),
                y=alt.Y("gross:Q", title="Incassi Stimati ($)"),
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
            last_year_preds = df_predictions[df_predictions['year'] == 2029]
            winner_row = last_year_preds.loc[last_year_preds['gross'].idxmax()]
            
            nome_genere = winner_row['genre']
            valore_gross = winner_row['gross']
            
            st.success(f"🏆 Il genere che dominerà il mercato nel **2030** sarà **{nome_genere}** con un incasso stimato di **${valore_gross:,.2f}**")
            
    else:
        st.info("Seleziona i generi e l'intervallo temporale per generare la proiezione dal 2017 in poi.")
