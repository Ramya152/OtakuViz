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
from PreprocessData import preprocess_data
from TextProcessing import remove_punctuation_and_turn_lower
from TextProcessing import string_cleaning
from TextProcessing import preprocess_synopsis
from TextProcessing import calculate_similarity
# Load environment variables
load_dotenv()

# B2 class for Backblaze interaction
class B2:
    def __init__(self, endpoint, key_id, secret_key):
        self.b2 = boto3.resource(
            service_name='s3',
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4')
        )
        
    def set_bucket(self, bucket_name):
        self.bucket = self.b2.Bucket(bucket_name)
        
    def get_df(self, remote_path):
        obj = self.bucket.Object(remote_path)
        df = pd.read_csv(obj.get()['Body'])
        return df

def get_anime_recommendation(df, similarities, name):
    index = df.index[df['Name'] == name].tolist()[0]
    normal_list = similarities[index]
    ordenated_list = sorted(normal_list, reverse=True)
    ranking_list = []
    for i in range(len(ordenated_list)):
        index_new = np.where(normal_list == ordenated_list[i])[0]
        tuple_anime = (int(index_new[0]), ordenated_list[i])
        ranking_list.append(tuple_anime)
    return ranking_list[1:6]

def preprocess_duration(df):
    df['Duration'] = df['Duration'].str.replace(' min per ep', '', regex=False)
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    df.loc[df['Episodes'] != 0, 'Duration'] *= df['Episodes']
    df['Duration'].fillna(df['Duration'].median(), inplace=True)
    return df

def preprocess_for_modeling(df):
    df_model = df.copy()
    df_model['Award Winning'] = df_model['Award Winning'].apply(lambda x: 1 if x == 'Yes' else 0)
    df_model['Episodes'] = df_model['Episodes'].fillna(df_model['Episodes'].median())
    df_model = df_model.dropna(subset=['Score'])
    return df_model

class OtakuVizApp:
    REMOTE_DATA = 'anime-dataset-2023.csv'

    def __init__(self):
        self.df, self.similarities, self.model, self.rmse, self.mae = self.load_data()
        if "report_index" not in st.session_state:
            st.session_state.report_index = 0

    @staticmethod
    @st.cache_data
    def load_data():
        b2 = B2(
            endpoint=os.environ['B2_ENDPOINT'],
            key_id=os.environ['B2_KEYID'],
            secret_key=os.environ['B2_APPKEY']
        )
        b2.set_bucket(os.environ['B2_BUCKETNAME'])
        df = b2.get_df(OtakuVizApp.REMOTE_DATA)
        df = preprocess_data(df)
        df = preprocess_synopsis(df)
        df = preprocess_duration(df)
        similarities = calculate_similarity(df)
        df_model = preprocess_for_modeling(df)
        X = df_model[['Favorites', 'Members', 'Award Winning', 'Episodes', 'Duration']]
        y = df_model['Score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        return df, similarities, model, rmse, mae

    def handle_error(self, error_message):
        st.error(f"An error occurred: {error_message}")

    def run(self):
        st.title('OtakuViz: Explore Anime Insights')
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ("Data Visualization", "Additional Analysis", "Recommendation", "Text-Based Recommendation", "Score Prediction"))

        if page == "Data Visualization":
            self.data_visualization_page()
        elif page == "Additional Analysis":
            self.additional_analysis_page()
        elif page == "Recommendation":
            self.recommendation_page()
        elif page == "Text-Based Recommendation":
            self.text_based_recommendation_page()
        elif page == "Score Prediction":
            self.score_prediction_page()

        st.sidebar.text("OtakuViz Web App")

    def data_visualization_page(self):
        st.sidebar.markdown("## Select Plot Parameters")
        selected_parameter = st.sidebar.selectbox("Select Parameter", ["Genre", "Type", "Status", "Source", "Rating"])

        if selected_parameter == "Genre":
            unique_values = sorted(self.df['Genres'].str.split(', ').explode().unique())
            selected_value = st.sidebar.selectbox(f"Select {selected_parameter}", unique_values)
            filtered_df = self.df[self.df['Genres'].str.contains(selected_value, na=False)]
        elif selected_parameter == "Type":
            unique_values = sorted(self.df['Type'].unique())
            selected_value = st.sidebar.selectbox(f"Select {selected_parameter}", unique_values)
            filtered_df = self.df[self.df['Type'] == selected_value]
        elif selected_parameter == "Status":
            unique_values = sorted(self.df['Status'].unique())
            selected_value = st.sidebar.selectbox(f"Select {selected_parameter}", unique_values)
            filtered_df = self.df[self.df['Status'] == selected_value]
        elif selected_parameter == "Source":
            unique_values = sorted(self.df['Source'].unique())
            selected_value = st.sidebar.selectbox(f"Select {selected_parameter}", unique_values)
            filtered_df = self.df[self.df['Source'] == selected_value]
        elif selected_parameter == "Rating":
            unique_values = sorted(self.df['Rating'].unique())
            selected_value = st.sidebar.selectbox(f"Select {selected_parameter}", unique_values)
            filtered_df = self.df[self.df['Rating'] == selected_value]

        st.write(filtered_df)

        st.sidebar.markdown("## Plot Customization")
        plot_type = st.sidebar.selectbox("Select Plot Type", ["Bar Plot", "Line Plot", "Pie Chart"])

        if plot_type == "Bar Plot":
            plt.figure(figsize=(10, 6))
            sns.countplot(x='Type', data=filtered_df)
            plt.title(f"Number of Anime by Type for {selected_value} {selected_parameter}")
            plt.xlabel("Type")
            plt.ylabel("Count")
            st.pyplot(plt)

        elif plot_type == "Line Plot":
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=filtered_df, x='Score', y='Popularity', hue='Type')
            plt.title(f"Score vs Popularity by Type for {selected_value} {selected_parameter}")
            plt.xlabel("Score")
            plt.ylabel("Popularity")
            st.pyplot(plt)

        elif plot_type == "Pie Chart":
            plt.figure(figsize=(10, 6))
            filtered_df[selected_parameter].value_counts().plot.pie(autopct='%1.1f%%')
            plt.title(f"Distribution of {selected_parameter}")
            st.pyplot(plt)

    def recommendation_page(self):
        st.header("Anime Recommendation System")
        st.write("Get anime recommendations based on genre and type.")
    
        # Split genres into individual values and get unique genres
        genre_list = self.df['Genres'].str.split(', ').explode().unique()
    
        # Select genres
        genre_choice = st.multiselect("Select Genre(s):", genre_list)
    
        # Select type
        type_list = self.df['Type'].unique()
        type_choice = st.selectbox("Select Type:", type_list)

        if genre_choice and type_choice:
            # Filter dataframe to include anime that have all selected genres and the selected type
            recommended_anime = self.df[self.df['Genres'].apply(lambda x: all(genre in x for genre in genre_choice)) & (self.df['Type'] == type_choice)]
            st.write(recommended_anime[['Name', 'Genres', 'Type', 'Score']])


    def text_based_recommendation_page(self):
        st.header("Text-Based Recommendation")
        st.write("Get anime recommendations based on synopsis similarity.")
        anime_list = self.df['Name'].unique()
        selected_anime = st.selectbox("Select an Anime:", anime_list)

        if selected_anime:
            recommendations = get_anime_recommendation(self.df, self.similarities, selected_anime)
            recommended_anime_names = [self.df.iloc[rec[0]]['Name'] for rec in recommendations]
            st.write("Recommended Animes:")
            st.write(recommended_anime_names)

    def score_prediction_page(self):
        st.header("Anime Score Prediction")
        st.write("Predict the score of an anime based on its attributes.")
        favorites = st.number_input("Enter Favorites:", value=0)
        members = st.number_input("Enter Members:", value=0)
        award_winning = st.selectbox("Award Winning:", options=[0, 1])  # 0 for No, 1 for Yes
        episodes = st.number_input("Enter Episodes:", value=0)
        duration = st.number_input("Enter Duration (in minutes):", value=0)
        input_data = pd.DataFrame({
            'Favorites': [favorites],
            'Members': [members],
            'Award Winning': [award_winning],
            'Episodes': [episodes],
            'Duration': [duration]
        })
        predicted_score = self.model.predict(input_data)
        st.write(f"Predicted Score: {predicted_score[0]:.2f}")

if __name__ == "__main__":
    app = OtakuVizApp()
    app.run()
