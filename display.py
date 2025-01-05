import streamlit as st
import pandas as pd
import numpy as np

st.title('Prediction Credit Risk')

#-> side right input
st.sidebar.header('Input preference Creditor')
input_person_age = st.sidebar.text_input('Person Age')
input_exp_work = st.sidebar.text_input('Experience Work')
input_person_income = st.sidebar.text_input('Person Income')
input_loan_amount = st.sidebar.text_input('Loan Amount')
input_loan_int_rate = st.sidebar.text_input('Loan Interest Rate')
input_home_status = st.sidebar.text_input('Own Home')
input_loan_intent = st.sidebar.text_input('Loan Intent')

#-> body 
url = "https://www.dicoding.com/login"
url2 = "https://www.kaggle.com/datasets/laotse/credit-risk-dataset"
st.write('''This dashboard shows Predictive Analytics [(from assignment Dicoding Indonesia)](%s) 
         using Machine Learning. And then evaluate model using MSE, MAE, R2. **Goals** the model is predict the score 
         of creditor based on own preference.''' %url, '''The Dataset for training modelling from Kaggle [Credit Risk Dataset](%s).''' % url2 )