#Moduli usati
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Url del dataset caricato su github
url = 'https://raw.githubusercontent.com/LucaIorio26/Streamlit_app/main/heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(url,sep=';')

#Variabile Rischio
conditions = [
    (data['time'] <= 70),
    (data['time'] > 71)]
values = [1,0]
data['Rischio'] = np.select(conditions, values)

# Variabili numeriche da standardizzare
X = data[['time','ejection_fraction','serum_creatinine']]
scaler = StandardScaler()
scaler.fit(X)
X_scale = scaler.transform(X)
X_scale = pd.DataFrame(X_scale)

#Aggiungo la variabile dummy
X_scale['Rischio']=data['Rischio']
X_scale.rename(columns={0 : 'time', 1 : 'ejection_fraction', 2 : 'serum_creatinine', 'Rischio' : 'Rischio'})

#Variabile target
y = data['DEATH_EVENT']

#Addestramento del classificatore
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.20, random_state=1234)

classifier = RandomForestClassifier(criterion='entropy', max_depth=7, random_state=1234)

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score = classifier.score(X_test, y_test)
print(score)


# Pagina Web realizzata con il modulo streamlit

def main():
    st.title('CALCOLATORE DI SOPRAVVIVENZA ❤')
    st.sidebar.header("CALCOLATORE DI SOPRAVVIVENZA ❤")
    st.sidebar.info("Semplice applicazione web per le persone che che hanno problemi cardiaci. ☘ ")
    st.sidebar.header('Come funziona?')
    st.sidebar.info("Completando ogni campo a destra è possibile ottenere una previsione abbastanza accurata circa le possibilità di sopravvivenza al termine della terapia.")
    st.sidebar.header("Quali strumenti sono stati utlizzati?")
    st.sidebar.info("Il classificatore utilizzato è il migliore tra i modelli addestrati durante una procedura di ensemble mediante una 5-Fold Cross Validation. In particolare si utilizza l'algoritmo Gradient Boosting, che si basa su una tecnica di ensemble con alberi di decisione, contenuta nel framework Scikit-learn. ")
    st.markdown("<h5> Compila i seguenti campi: </h5>", unsafe_allow_html=True)
    sex = st.selectbox('Indica il tuo genere  (DONNA = 0, UOMO = 1)', (0, 1))
    age = st.slider("Indica la tua età?", min_value=40, max_value=87, value=46)
    smoking = st.selectbox('Sei un consumatore di tabacco? (SI = 1, NO = 0)', (0, 1))
    diabetes = st.selectbox('Sei un soggetto diabetico? (SI = 1, NO = 0)', (0, 1))
    anaemia = st.selectbox('Sei un soggetto anemico? (SI = 1, NO = 0)', (0, 1))
    high_blood_pressure = st.selectbox('Soffri di ipertensione? (SI = 1, NO = 0)', (0, 1))
    st.markdown("<h5> Dalle analisi cliniche più recenti è emerso che: </h5>",unsafe_allow_html=True)
    platelets = st.slider("Il quantitativo di piastrine è circa..", min_value=0, max_value=900000, value=375989)
    creatinine_phosphokinase = st.slider("L'enzima CPK (creatina fosfochinasi) è pari a..", min_value=0, max_value=7800, value=2574)
    serum_sodium = st.slider("Concentrazione di sodio nel sangue", min_value=0, max_value=200, value=59)
    time = st.slider('Da quanti giorni hai iniziato il trattamento?', min_value=0, max_value=300, value=35)
    st.markdown("<h5> A seguito di un ecocardiografia è emerso che: </h5>",unsafe_allow_html=True)
    ejection_fraction = st.slider("La frazione volumetrica di sangue espulsa dal cuore è ", min_value=0, max_value=100, value=43)
    serum_creatinine = st.slider("Concentrazione di creatinina nel sangue", min_value=0.0, max_value=6.0, step=0.2,
                                 value=2.60)
    if time <= 71:
      Rischio = 1
    else:
      Rischio = 0

    inputs = [[ejection_fraction, serum_creatinine, time, Rischio]]

    if st.button('Predict'):
        result = classifier.predict_proba(inputs)
        print(result)
        data_probabilities = pd.DataFrame(result, columns=classifier.classes_)
        if data_probabilities.loc[0][0] > data_probabilities.loc[0][1]:
            st.success(f'La probabilità di sopravvivere è {data_probabilities.loc[0][0].round(3)}')
        if data_probabilities.loc[0][0] < data_probabilities.loc[0][1]:
            st.success(f'La probabilità di morire è {data_probabilities.loc[0][1].round(3)}')


if __name__ == '__main__':
    main()



