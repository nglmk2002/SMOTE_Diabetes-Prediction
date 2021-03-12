import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle 
import tensorflow as tf
import joblib

def run_ml():

    #임신횟수
    Pregnancies = st.number_input('임신횟수 입력',min_value=0,max_value=10)

    Glucose = st.number_input('공복혈당 입력',min_value=0,max_value=500)

    BloodPressure = st.number_input('혈압 입력',min_value=0,max_value=500)

    SkinThickness = st.number_input('피부두께 입력',min_value=0,max_value=500)

    Insulin	= st.number_input('인슐린 수치 입력',min_value=0,max_value=500)

    BMI = st.number_input('BMI 입력',0.0,100.0)

    DiabetesPedigreeFunction = st.number_input('당뇨병 입력',0.000,10.000)

    Age = st.number_input('나이 입력',min_value=0,max_value=100)

    model = joblib.load('data/diabetes_ai.joblib')
    
    new_data=np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
        BMI,DiabetesPedigreeFunction,Age]).reshape(1,-1)
    
    y_pred = model.predict(new_data)

    btn = st.button('결과 보기')
    if btn :
        if y_pred == 1 :
            st.write('당뇨병 당첨')
        else :
            st.write('당뇨병 꽝')


 

  
    
    
    

if __name__ == '__main__' :
    main()  
