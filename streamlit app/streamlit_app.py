import streamlit as st
import requests
import pandas as pd

API_URL = "https://elvin-aliyev-census-income-api.hf.space/predict"

st.title('Census Income Predictor')
st.write('Enter values to predict')
tab1, tab2 = st.tabs(["Prediction", "History"])

if 'pred_history' not in st.session_state:
    st.session_state.pred_history = []

with tab1:
    with st.form('predict_form'):
      col1, col2 = st.columns(2)
      with col1:
         age=st.number_input(label='Age',max_value=100,min_value=0,value=19)
         hours_per_week = st.number_input("Hours per Week", min_value=0, max_value=168, value=40)
         workclass=st.selectbox('Workclass',["Private", "Self-emp-not-inc", "Self-emp-inc",
                                             "Federal-gov", "Local-gov", "State-gov",
                                             "Other"])
         education=st.selectbox('Education',["Bachelors", "Some-college", "HS-grad",
                                                "Masters", "Doctorate", "Assoc-acdm",
                                                "Assoc-voc", "11th", "10th", "9th", "12th",
                                                "1st-4th", "5th-6th", "7th-8th", "Preschool"])
         marital_status=st.selectbox('Marital Staus',["Married-civ-spouse", "Divorced",
                                                            "Never-married", "Separated",
                                                            "Widowed", "Married-spouse-absent"])
      with col2:
         
         
         occupation = st.selectbox("Occupation", ['Craft-repair','Prof-specialty','Exec-managerial',
                                                   'Adm-clerical','Sales','Other-service','Machine-op-inspct',
                                                   'Transport-moving','Handlers-cleaners','Farming-fishing',
                                                   'Tech-support','Protective-serv','Priv-house-serv'])
         relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband",
                                                      "Not-in-family", "Other-relative",
                                                      "Unmarried"])
         race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander",
                                       "Amer-Indian-Eskimo", "Other"])
         sex = st.selectbox("Sex", ["Male", "Female"])
         native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Greece', 'Vietnam', 'China',
         'Taiwan', 'India', 'Philippines', 'Trinadad&Tobago', 'Canada',
         'South', 'Holand-Netherlands', 'Puerto-Rico', 'Poland', 'Iran',
         'England', 'Germany', 'Italy', 'Japan', 'Hong', 'Honduras', 'Cuba',
         'Ireland', 'Cambodia', 'Peru', 'Nicaragua', 'Dominican-Republic',
         'Haiti', 'El-Salvador', 'Hungary', 'Columbia', 'Guatemala',
         'Jamaica', 'Ecuador', 'France', 'Yugoslavia', 'Scotland',
         'Portugal', 'Laos', 'Thailand', 'Outlying-US(Guam-USVI-etc)'])
      capital_gain = st.slider("Capital Gain", min_value=0, max_value=99999, value=0, step=100)
      capital_loss = st.slider("Capital Loss", min_value=0, max_value=4356, value=0, step=50)
      submitted = st.form_submit_button("Predict")


if submitted:
   payload = {
        "age": int(age),
        "workclass": workclass,
        "education": education,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": int(capital_gain),
        "capital_loss": int(capital_loss),
        "hours_per_week": int(hours_per_week),
        "native_country": native_country}
   try:
      response= requests.post(API_URL,json=payload)
      if response.status_code == 200:
         result = response.json()
         prediction = result.get('prediction')
         if prediction == 1:
                st.success("Prediction: Income > 50K")
         else:
                 st.info("Prediction: Income <= 50K")
         history_entry = payload.copy()
         history_entry['prediction'] = prediction
         st.session_state.pred_history.append(history_entry)
      
      else:
            st.error(f"API Error: Status Code {response.status_code}")
            st.write(response.text)
   except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI. Make sure the API is running!")

with tab2:
    st.write("Prediction History")
    if st.session_state.pred_history:
        df_history = pd.DataFrame(st.session_state.pred_history)
        st.dataframe(df_history)
    else:
        st.info("No predictions made yet.")

   


