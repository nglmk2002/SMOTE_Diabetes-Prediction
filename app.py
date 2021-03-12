import streamlit as st
import numpy as np 
import pandas as pd
import seaborn as sns
from eda import run_eda
from ml import run_ml


def main():
    
    
    st.title('당뇨병 예측 모델')
    

    menu = ['Home','DataFrame','Deep Learning']
    selected_side=st.sidebar.selectbox('메뉴를 선택해주세요',menu)

    if selected_side == 'Home':
        st.write('이 모델은 당뇨병예측을 위한 모델입니다.')

    if selected_side == 'DataFrame':
        run_eda()
    
    if selected_side == 'Deep Learning':
        run_ml()

        
    

    


if __name__ == '__main__' :
    main()  
