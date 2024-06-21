import os
import string
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import boto3
from botocore.config import Config
# Function to preprocess data
@st.cache_data
def preprocess_data(df):
    columns_to_drop = ['Premiered', 'English name', 'Other name', 'Producers', 'Licensors', 'Studios', 'Image URL']
    df.drop(columns=columns_to_drop, inplace=True)
    df = df[(df['Rating'] != 'UNKNOWN') & (df['Episodes'] != 'UNKNOWN') & (df['Rank'] != 'UNKNOWN') & 
            (df['Type'] != 'UNKNOWN') & (df['Genres'] != 'UNKNOWN') & (df['Duration'] != 'Unknown') & 
            (df['Source'] != 'Unknown') & (df['Score'] != 'UNKNOWN')].copy()
    df['Award Winning'] = df['Genres'].apply(lambda x: "Yes" if "Award Winning" in x else "No")
    df['Genres'] = df['Genres'].str.replace(",? Award Winning,?", "", regex=True)
    df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df['Type'] = df['Type'].astype('category')
    df['Status'] = df['Status'].astype('category').str.strip().str.lower().replace({
        'finished airing': 'finished',
        'currently airing': 'ongoing',
        'not yet aired': 'upcoming'
    })
    df['Aired'] = pd.to_datetime(df['Aired'], errors='coerce')
    return df
