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
#Text preprocessing functions
def remove_punctuation_and_turn_lower(text):
    translator = str.maketrans('', '', string.punctuation)
    return (text.translate(translator)).lower()

def string_cleaning(string_list):
    strings_limpas = [remove_punctuation_and_turn_lower(string) for string in string_list]
    strings_limpas_no_numbers = [re.sub(r'\d', '', string) for string in strings_limpas]
    new_list = [item for item in strings_limpas_no_numbers if item]
    tokens_without_sw = [word for word in new_list if not word in stopwords.words('english')]
    ps = PorterStemmer()
    steemed_words = [ps.stem(w) for w in tokens_without_sw]
    return steemed_words

@st.cache_data
def preprocess_synopsis(df):
    df['Synopsis'] = df['Synopsis'].apply(lambda x: x.split() if isinstance(x, str) else [])
    df['Synopsis_cleaned'] = df['Synopsis'].apply(string_cleaning)
    df['Synopsis_cleaned_text'] = df['Synopsis_cleaned'].apply(lambda x: ' '.join(x))
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=5000)
    df['Synopsis_vectorized'] = list(vectorizer.fit_transform(df['Synopsis_cleaned_text']).toarray())
    return df
    
@st.cache_data
def calculate_similarity(df):
    similarities = cosine_similarity(df['Synopsis_vectorized'].tolist())
    return similarities
