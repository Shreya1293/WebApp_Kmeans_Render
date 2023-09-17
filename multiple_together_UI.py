# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 00:38:19 2023

@author: AnS
"""

import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

#loading the model
loaded_model1=pickle.load(open('House_model.sav','rb'))
loaded_model2=pickle.load(open('cust_segm.sav','rb'))

#sidebar to navigate
with st.sidebar:
    selected = option_menu ('Prodigy Infotech ML Internship',
                            ['House Price Prediction'
                             ,'KMeans Customer Cluster Classification'
                             ],
                            icons = ['house-fill','pie-chart-fill'
                                     ], #google bootstrap icons and copy the name of the one which you want
                            default_index=0)

#house pred page
if(selected == 'House Price Prediction'):
    #page title
    st.title('House Price Prediction')
    
    #user input
    bedrooms=st.number_input("Enter the number of bedrooms:")
    bathrooms=st.number_input("Enter the number of bathrooms:")
    sqlivarea=st.number_input("Enter the living area in square feet:")
    sqlot=st.number_input("Enter the Lot size in square feet:")
    
    #code for prediction
    result=''
    
    #creating a button
    if st.button("Predict the Prize"):
        result=loaded_model1.predict([[bedrooms,bathrooms,sqlivarea,sqlot]])
    
    st.success(result)
    
if (selected == 'KMeans Customer Cluster Classification'):
    st.title('Customer Cluster Classification')
    
    user_income = st.number_input("Income:")
    user_spending_score = st.number_input("Spending Score:")
    
    # Collect user input into a DataFrame
    user_input = pd.DataFrame({
        'Income': [user_income],
        'Spending Score': [user_spending_score]
        })
    # Define features for clustering
    selected_columns = ['Income', 'Spending Score']
    result=''
    
    if st.button("Cluster Number"):
        result=loaded_model2.predict(user_input[selected_columns])[0]
    
    st.success(result)