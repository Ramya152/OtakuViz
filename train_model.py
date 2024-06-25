import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

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

def train_and_save_model():
    # Initialize Backblaze B2
    b2 = B2(
        endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_KEYID'],
        secret_key=os.environ['B2_APPKEY']
    )
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df = b2.get_df('anime-dataset-2023.csv')
    
    # Preprocess data
    df = preprocess_data(df)
    df = preprocess_duration(df)
    df_model = preprocess_for_modeling(df)
    #using these features instead of episodes etc(prev used features)
    X = df_model[['Favorites', 'Members', 'Popularity', 'Rank']]
    y = df_model['Score']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Train the best model on the entire dataset
    best_model.fit(X, y)

    # Save the model
    joblib.dump(best_model, 'best_random_forest_model.pkl')

    print("Best parameters found: ", grid_search.best_params_)
    print("Model saved to best_random_forest_model.pkl")

if __name__ == "__main__":
    train_and_save_model()

