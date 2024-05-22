import PreProcessing as preprocess 
import Modelling as modelling 
from ModelEvaluation import ModelEvaluation
import numpy as np
import pandas as pd

def main():
    # Load the dataset
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
    dt_params = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 2, 4, 8, 10], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2, 4]}
    
    rf_params = {
        'n_estimators': [100],
        'max_depth': [None, 5,],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
    }

    # Train and evaluate KNN
    knn_model = model_instance.train_model(x_train, y_train, 'knn', param_grid=knn_params)
    knn_accuracy = model_instance.evaluate_model(knn_model, x_test, y_test)
    print("KNN Model Accuracy (Overall, When 5rs=1, When 5rs=0):", knn_accuracy)

    # Train and evaluate Decision Tree
    dt_model = model_instance.train_model(x_train, y_train, 'decision_tree', param_grid=dt_params)
    dt_accuracy = model_instance.evaluate_model(dt_model, x_test, y_test)
    print("Decision Tree Model Accuracy (Overall, When 5rs=1, When 5rs=0):", dt_accuracy)

    # Train and evaluate Random Forest
    rf_model = model_instance.train_model(x_train, y_train, 'random_forest', param_grid=rf_params)
    rf_accuracy = model_instance.evaluate_model(rf_model, x_test, y_test)
    print("Random Forest Model Accuracy (Overall, When 5rs=1, When 5rs=0):", rf_accuracy)

    # Perform KMeans clustering on the entire dataset (example)
    kmeans_params = {'n_clusters': [2, 3, 4]}
    kmeans_model, cluster_labels = model_instance.perform_clustering(train_data, 'kmeans', param_grid=kmeans_params)
    #print("KMeans Cluster Centers:", kmeans_model.cluster_centers_)
    
    # Print chosen hyperparameters for each model
    print("Best Hyperparameters:")
    print("KNN:", model_instance.best_params['knn'])
    print("Decision Tree:", model_instance.best_params['decision_tree'])
    print("Random Forest:", model_instance.best_params['random_forest'])

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



if __name__ == "__main__":
    main()
