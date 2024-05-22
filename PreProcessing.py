import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def null_value_analysis(df):
    # Drop null values
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
    df['Points_Per_Minute'] = (df['PTS'] / df['MIN']).round(3)
    df['Field_Goal_Efficiency'] = (df['FGM'] / df['FGA']).round(3)
    df['Points_Per_Game_Played'] = (df['PTS'] / df['GP']).round(3)
    df['Total_Rebounds'] = (df['OREB'] + df['DREB']).round(3)
    df['Efficiency_Rating'] = ((df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - 
                               (df['FGA'] - df['FGM']) - (df['FTA'] - df['FTM']) - df['TOV']) / df['GP']).round(3)
    return df

def determine_feature_importance(df):
    # Determine feature importance using a model
    x = df.drop(columns=['Id', 'TARGET_5Yrs'])
    y = df['TARGET_5Yrs']
    model = RandomForestClassifier(random_state=30)
    model.fit(x, y)
    importances = model.feature_importances_
    feature_importance = pd.Series(importances, index=x.columns)

     
    return feature_importance

def perform_train_test_split(df):
    # Print the number of entries with the target value as 1 and 0\
    print("Number of entries with target value as 1: ", len(df[df['TARGET_5Yrs'] == 1]))
    print("Number of entries with target value as 0: ", len(df[df['TARGET_5Yrs'] == 0]))


    # Separate the data based on the target value
    df_class_1 = df[df['TARGET_5Yrs'] == 1]
    df_class_0 = df[df['TARGET_5Yrs'] == 0]

    # Split each class separately
    x_class_1 = df_class_1.drop(columns=['Id', 'TARGET_5Yrs'])
    y_class_1 = df_class_1['TARGET_5Yrs']
    x_class_0 = df_class_0.drop(columns=['Id', 'TARGET_5Yrs'])
    y_class_0 = df_class_0['TARGET_5Yrs']

    x_train_1, x_test_1, y_train_1, y_test_1 = sk_train_test_split(x_class_1, y_class_1, test_size=0.1)
    x_train_0, x_test_0, y_train_0, y_test_0 = sk_train_test_split(x_class_0, y_class_0, test_size=0.1)

    # Combine the splits back together
    x_train = pd.concat([x_train_1, x_train_0])
    x_test = pd.concat([x_test_1, x_test_0])
    y_train = pd.concat([y_train_1, y_train_0])
    y_test = pd.concat([y_test_1, y_test_0])

    # Print the number of entries in each class in the training and test sets
    print("Training set - Number of entries with target value as 1: ", sum(y_train == 1))
    print("Training set - Number of entries with target value as 0: ", sum(y_train == 0))
    print("Test set - Number of entries with target value as 1: ", sum(y_test == 1))
    print("Test set - Number of entries with target value as 0: ", sum(y_test == 0))

    


    return x_train, x_test, y_train, y_test
