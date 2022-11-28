# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:57:21 2022

@author: Anon
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


#loading the saved model
loaded_model = pickle.load(open('C:/Users/Anon/Desktop/PROJECT_AY_JOEL/DEPLOYMENT/parkinsons_model.sav', 'rb'))


# creating a function for prediction

def parkinsons_prediction(input_data):

    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the data
    std_data = scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)
    print(prediction)


    if (prediction[0] == 0):
      return "The Person does not have Parkinson's Disease"
    else:
      return "The Person has Parkinson's Disease"
  

def main():
    
    #giving a title
    st.title("Early Detection of Parkinson's Disease and Its Analysis Using KNN, SVM & LR")
    
    
    #getting input data from the user
    col1, col2, col3, col4 = st.columns(4)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col1:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col2:
        RAP = st.text_input('MDVP:RAP')
        
    with col3:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col4:
        DDP = st.text_input('Jitter:DDP')
        
    with col1:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col2:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col3:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col4:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col1:
        APQ = st.text_input('MDVP:APQ')
        
    with col2:
        DDA = st.text_input('Shimmer:DDA')
        
    with col3:
        NHR = st.text_input('NHR')
        
    with col4:
        HNR = st.text_input('HNR')
        
    with col1:
        RPDE = st.text_input('RPDE')
        
    with col2:
        DFA = st.text_input('DFA')
        
    with col3:
        spread1 = st.text_input('spread1')
        
    with col4:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = loaded_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)
    
    
if __name__ == 'main':
    main()