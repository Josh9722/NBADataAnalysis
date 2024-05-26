import PreProcessing as preprocess 
import Modelling as modelling 
from ModelEvaluation import ModelEvaluation
import numpy as np
import pandas as pd
import pickle 


def main():
    analyse_train_data()
    analyse_test_data() 

def analyse_test_data():
    # Take Test Data and Provide TARGET_5Yrs Predictions and Confidence Scores
    test_data_original = pd.read_csv('nba_test.csv')
    test_data_copy = test_data_original.copy()
    
    # Add the extra features to the test data
    test_data_copy = preprocess.feature_engineering(test_data_copy)
    
    # Load the pre-trained Random Forest Model
    rf_model = load_model('random_forest_model.pkl')
    
    # Make predictions
    predictions = rf_model.predict(test_data_copy)
    prediction_probabilities = rf_model.predict_proba(test_data_copy)[:, 1]  # Probability of class 1

    # Add predictions to the test data
    test_data_original['TARGET_5Yrs'] = predictions
    test_data_original['Prediction_Probability'] = prediction_probabilities

    # Save the result to a new CSV file
    test_data_original.to_csv('nba_test_with_predictions.csv', index=False)


def analyse_train_data():
    train_data = pd.read_csv('nba_train.csv')
    # STEP 1: PRE PROCESSING
    # Training Data
    print("--------------------------------------------------")
    print("Data Pre-processing Started...")
    train_data = preprocess.null_value_analysis(train_data)
    train_data = preprocess.duplicate_value_analysis(train_data)
    train_data = preprocess.outlier_analysis(train_data)
    train_data = preprocess.feature_engineering(train_data)
    feature_importance = preprocess.determine_feature_importance(train_data)
    sorted_feature_importance = feature_importance.sort_values(ascending=False)
    print("Feature Importances:\n", sorted_feature_importance)

    # Splitting the training data into training and testing sets
    x_train, x_test, y_train, y_test = preprocess.perform_train_test_split(train_data)


    print("Data Pre-processing Completed.")
    print("--------------------------------------------------")

    
    # STEP 2: MODELLING
    print("--------------------------------------------------")
    print("Modelling Started...")
    model_instance = modelling.Modelling(feature_importances = feature_importance)

    # Define hyperparameters for each model
    knn_params = {'n_neighbors': [2, 3, 5, 7, 10]}
    dt_params = {'criterion': ['gini', 'entropy', 'log_loss'], 'max_depth': [None, 2, 4, 8, 10], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2, 4]}
    
    rf_params = {
        #'n_estimators': [50, 100, 200, 400], Commented out for faster execution but n = 100 has been determined as most optimal
        'n_estimators': [100],
        'max_depth': [None, 5,],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
        'criterion': ['gini', 'entropy', 'log_loss']
    }

    # Train and evaluate KNN
    print("Beginning KNN Model Training and Evaluation...")
    knn_model = model_instance.train_model(x_train, y_train, 'knn', param_grid=knn_params)
    knn_accuracy = model_instance.evaluate_model(knn_model, x_test, y_test)
    print("KNN Model Accuracy On Testing Split (Overall, When 5yrs=1, When 5yrs=0):", knn_accuracy)
    knn_accuracy_train = model_instance.evaluate_model(knn_model, x_train, y_train)
    print("KNN Model Accuracy On Training Split (Overall, When 5yrs=1, When 5yrs=0):", knn_accuracy_train)
    print("--------------------------------------------------")

    # Train and evaluate Decision Tree
    print("Beginning Decision Tree Model Training and Evaluation...")
    dt_model = model_instance.train_model(x_train, y_train, 'decision_tree', param_grid=dt_params)
    dt_accuracy = model_instance.evaluate_model(dt_model, x_test, y_test)
    print("Decision Tree Model Accuracy On Testing Split (Overall, When 5yrs=1, When 5yrs=0):", dt_accuracy)
    dt_accuracy_train = model_instance.evaluate_model(dt_model, x_train, y_train)
    print("Decision Tree Model Accuracy On Training Split (Overall, When 5yrs=1, When 5yrs=0):", dt_accuracy_train)
    print("--------------------------------------------------")

    # Train and evaluate Random Forest
    print("Beginning Random Forest Model Training and Evaluation...")
    rf_model = model_instance.train_model(x_train, y_train, 'random_forest', param_grid=rf_params)
    rf_accuracy = model_instance.evaluate_model(rf_model, x_test, y_test)
    print("Random Forest Model Accuracy On Testing Split (Overall, When 5yrs=1, When 5yrs=0):", rf_accuracy)
    rf_accuracy_train = model_instance.evaluate_model(rf_model, x_train, y_train)
    print("Random Forest Model Accuracy On Training Split (Overall, When 5yrs=1, When 5yrs=0):", rf_accuracy_train)
    print("--------------------------------------------------")

    # Perform KMeans clustering on the entire dataset
    print ("Performing KMeans Clustering...")
    kmeans_params = {'n_clusters': [2, 3, 4]}
    kmeans_model, cluster_labels = model_instance.perform_clustering(train_data, 'kmeans', param_grid=kmeans_params)
    print("KMeans Cluster Labels:", cluster_labels)
    print("KMeans Cluster Centers:", kmeans_model.cluster_centers_)
    print("--------------------------------------------------")
    
    # Print chosen hyperparameters for each model
    print("Best Hyperparameters:")
    print("KNN:", model_instance.best_params['knn'])
    print("Decision Tree:", model_instance.best_params['decision_tree'])
    print("Random Forest:", model_instance.best_params['random_forest'])
    print("K Means:", model_instance.best_params['kmeans'])

    print("Modelling Completed.")
    print("--------------------------------------------------")

    # STEP 4: MODEL EVALUATION
    eval_instance = ModelEvaluation()
    
    # Evaluate KNN
    knn_auc = eval_instance.evaluate_roc(knn_model, x_test, y_test, 'KNN')
    print("KNN Model AUC:", knn_auc)
    
    # Evaluate Decision Tree
    dt_auc = eval_instance.evaluate_roc(dt_model, x_test, y_test, 'Decision Tree')
    print("Decision Tree Model AUC:", dt_auc)
    
    # Evaluate Random Forest
    rf_auc = eval_instance.evaluate_roc(rf_model, x_test, y_test, 'Random Forest')
    print("Random Forest Model AUC:", rf_auc)
    
    # Plot ROC Curves
    eval_instance.plot_roc_curve()


    # Saving the chosen model (Random Forest) to a file
    save_model(rf_model, 'random_forest_model.pkl')


def load_model(filename):
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")
    
    

if __name__ == "__main__":
    main()
