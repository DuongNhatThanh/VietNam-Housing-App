import streamlit as st
import pandas as pd

@st.cache_resource
def get_dataset():
    df = pd.read_csv("data/processed/VN_housing_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])

    print(df.columns)
    
    return df
