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


def main():
    df = pd.read_csv('diabetes.csv')

    X=df.drop('Outcome',axis=1)
    y=df['Outcome']




if __name__ == '__main__' :
    main()  
