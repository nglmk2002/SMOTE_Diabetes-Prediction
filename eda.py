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


def run_eda():
    df = pd.read_csv('data/diabetes.csv')
    corr_columns = df.columns[df.dtypes != object]
    selected_columns_corr = st.multiselect('상관계수 컬럼 선택해주세요',corr_columns)


    if len(selected_columns_corr) != 0:
        st.dataframe(df[selected_columns_corr].corr())
        if 'Outcome' in selected_columns_corr:
            fig=sns.pairplot(data=df[selected_columns_corr],hue='Outcome')
            st.pyplot(fig)
        
        else :
            fig=sns.pairplot(data=df[selected_columns_corr])
            st.pyplot(fig)


    else :
        st.write('선택한 컬럼이 없습니다.')



if __name__ == '__main__' :
    main()  
