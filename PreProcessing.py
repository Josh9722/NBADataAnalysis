import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def null_value_analysis(df):
    # Fill or drop null values
    df = df.dropna()
    return df

def duplicate_value_analysis(df):
    # Drop duplicate values
    df = df.drop_duplicates()
    return df

def outlier_analysis(df):
    # Remove outliers using Z-score or IQR method
    for col in df.select_dtypes(include=[np.number]).columns:
        df = df[(np.abs(df[col] - df[col].mean()) / df[col].std()) < 3]
    return df

def feature_engineering(df):
    # Add or modify features if necessary
    # Example: Creating a new feature based on existing ones
    df['Points_Per_Minute'] = df['PTS'] / df['MIN']
    df['Field_Goal_Efficiency'] = df['FGM'] / df['FGA']
    df['Points_Per_Game_Played'] = df['PTS'] / df['GP']
    df['Total_Rebounds'] = df['OREB'] + df['DREB']
    df['Efficiency_Rating'] = (df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - 
                               (df['FGA'] - df['FGM']) - (df['FTA'] - df['FTM']) - df['TOV']) / df['GP']
    return df

def determine_feature_importance(df):
    # Determine feature importance using a model
    x = df.drop(columns=['Id', 'TARGET_5Yrs'])
    y = df['TARGET_5Yrs']
    model = RandomForestClassifier(random_state=30)
    model.fit(x, y)
    importances = model.feature_importances_
    feature_importance = pd.Series(importances, index=x.columns).sort_values(ascending=False)
    return feature_importance

def perform_train_test_split(df):
    # Split the data into training and test sets
    x = df.drop(columns=['TARGET_5Yrs'])
    y = df['TARGET_5Yrs']
    x_train, x_test, y_train, y_test = sk_train_test_split(x, y, test_size=0.1, random_state=30)

    return x_train, x_test, y_train, y_test
