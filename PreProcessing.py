import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
     # Initialize masks
    mask = np.ones(len(df), dtype=bool)

    # Iterate over each numerical column to update the mask
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'TARGET_5Yrs' and col != 'Id':  # Skip the target and id column
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = mask & (z_scores < 3)

    # Calculate the number of outliers
    outliers_target_1 = len(df[(~mask) & (df['TARGET_5Yrs'] == 1)])
    outliers_target_0 = len(df[(~mask) & (df['TARGET_5Yrs'] == 0)])

    # Calculate the total number of instances in each class
    total_target_1 = len(df[df['TARGET_5Yrs'] == 1])
    total_target_0 = len(df[df['TARGET_5Yrs'] == 0])

    # Calculate the percentage of outliers in each class
    percent_outliers_target_1 = (outliers_target_1 / total_target_1) * 100 if total_target_1 > 0 else 0
    percent_outliers_target_0 = (outliers_target_0 / total_target_0) * 100 if total_target_0 > 0 else 0

    # Print the number of outliers and their percentages
    print(f"Number of outliers when TARGET_5Yrs is 1: {outliers_target_1} ({percent_outliers_target_1:.2f}%)")
    print(f"Number of outliers when TARGET_5Yrs is 0: {outliers_target_0} ({percent_outliers_target_0:.2f}%)")

    # Return the DataFrame with rows removed where the mask is False
    return df[mask]

def feature_engineering(df):
    # Add or modify features if necessary
    # Example: Creating a new feature based on existing ones

    df['Points_Per_Minute'] = (df['PTS'] / df['MIN']).round(3)
    df['Field_Goal_Efficiency'] = (df['FGM'] / df['FGA']).round(3)
    df['Points_Per_Game_Played'] = (df['PTS'] / df['GP']).round(3)
    df['Total_Rebounds'] = (df['OREB'] + df['DREB']).round(3)
    df['DREB_Per_Game'] = (df['DREB'] / df['GP']).round(3)
    df['STL_Per_Game'] = (df['STL'] / df['GP']).round(3)
    df['BLK_Per_Game'] = (df['BLK'] / df['GP']).round(3)
    df['TOV_Per_Game'] = (df['TOV'] / df['GP']).round(3)
    df['Points_Per_FGA'] = (df['PTS'] / df['FGA']).round(3)
    df['Points_Per_REB'] = (df['PTS'] / df['REB']).round(3)
    df['TOV_Per_Minute'] = (df['TOV'] / df['MIN']).round(3)
    df['TOV_Per_Game'] = (df['TOV'] / df['GP']).round(3)
    df['FGM_Missed_Per_Game'] = ((df['FGA'] - df['FGM']) / df['GP']).round(3)
    df['FT_Missed_Per_Game'] = ((df['FTA'] - df['FTM']) / df['GP']).round(3)
    df['FG_Missed_Percentage'] = (1 - df['FG%']).round(3)

    # More complicated feature engineering stats
    df['Offensive_Rating'] = ((df['PTS'] / df['MIN']) * 100).round(3)
    df['Defensive_Rating'] = ((df['DREB'] / df['MIN']) * 100).round(3)
    df['Usage_Rate'] = (((df['FGA'] + df['TOV'] + (0.44 * df['FTA'])) * (df['GP'] / 5)) / df['MIN']).round(3) 
    df['True_Shooting_Percentage'] = (df['PTS'] / (2 * (df['FGA'] + (0.44 * df['FTA'])))).round(3)
    df['Effective_Field_Goal_Percentage'] = (((df['FGM'] + (0.5 * df['3P Made'])) / df['FGA'])).round(3)
    df['Turnover_Percentage'] = ((df['TOV'] / (df['FGA'] + (0.44 * df['FTA']) + df['TOV'])) * 100).round(3)
    df['Offensive_Rebound_Percentage'] = ((df['OREB'] / (df['OREB'] + df['DREB'])) * 100).round(3)
    df['Defensive_Rebound_Percentage'] = ((df['DREB'] / (df['DREB'] + df['OREB'])) * 100).round(3)

    # Non linear features
    df['PTS_Cubed'] = df['PTS'] ** 3
    df['Usage_Rate_Power4'] = df['Usage_Rate'] ** 4 
    df['PTS_MIN_Poly'] = (df['PTS'] ** 2) * (df['MIN'] ** 2)
    df['FT_FGA_Poly'] = (df['FTM'] ** 2) * (df['FGA'] ** 3)
    df['Efficiency_Usage_Interaction'] = (df['True_Shooting_Percentage'] ** 2) * (df['Usage_Rate'] ** 3)
    df['Defensive_Complex'] = (df['Defensive_Rating'] ** 2) * (df['STL_Per_Game'] + df['BLK_Per_Game']) ** 2
    df['PTS_FG%_Fractional'] = (df['PTS'] ** (2/3)) * (df['FG%'] ** (1/3))
    df['Usage_TOV_Fractional'] = (df['Usage_Rate'] ** (1/2)) / (df['TOV_Per_Minute'] ** (1/4))
    df['Log_Exp_Cross'] = np.log(df['PTS'] + 1) * np.exp(df['MIN'] / 100)
    df['Sin_MIN'] = np.sin(df['MIN'])
    df['Cos_MIN'] = np.cos(df['MIN'])

    # Scaling
    scaler = StandardScaler()  # Or MinMaxScaler()
    features = ['MIN', 'PTS', 'FGM', 'FGA', '3PA', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV']  # List all features needing scaling
    df[features] = scaler.fit_transform(df[features])

    # Drop columns that have a low feature importance
    columns_to_drop = [
    'FG_Missed_Percentage', 'FG%', 'Offensive_Rebound_Percentage', 'PTS_MIN_Poly',
    'Points_Per_Minute', 'FT_FGA_Poly', 'Defensive_Rebound_Percentage', '3PA',
    'TOV_Per_Minute', 'FTM', 'MIN', 'Log_Exp_Cross', 'AST', 'Points_Per_Game_Played',
    'FGM_Missed_Per_Game', 'DREB_Per_Game', 'TOV_Per_Game', 'REB', 'FT_Missed_Per_Game',
    'FTA', 'Total_Rebounds', 'BLK_Per_Game', 'PTS', 'OREB', 'STL_Per_Game', 'DREB',
    '3P Made', 'FGM', 'BLK', 'FGA', 'PTS_Cubed', 'TOV', 'STL']

    df = df.drop(columns=columns_to_drop)

    # Drop the 'Id' column
    df = df.drop(columns=['Id'])
    df.to_csv('new_data.csv', index=False)
    return df

def determine_feature_importance(df):
    # Determine feature importance using a model
    x = df.drop(columns=['TARGET_5Yrs'])
    y = df['TARGET_5Yrs']
    model = RandomForestClassifier(class_weight = "balanced")
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
    x_class_1 = df_class_1.drop(columns=['TARGET_5Yrs'])
    y_class_1 = df_class_1['TARGET_5Yrs']
    x_class_0 = df_class_0.drop(columns=['TARGET_5Yrs'])
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
