import PreProcessing as preprocess 
import Modelling as modelling 
from ModelEvaluation import ModelEvaluation

import pandas as pd

def main():
    # Load the dataset
    train_data = pd.read_csv('nba_train.csv')
    test_data = pd.read_csv('nba_test.csv')

    # STEP 1: PRE PROCESSING
    # Training Data
    print("--------------------------------------------------")
    print("Data Pre-processing Started...")
    train_data = preprocess.null_value_analysis(train_data)
    train_data = preprocess.duplicate_value_analysis(train_data)
    train_data = preprocess.outlier_analysis(train_data)
    train_data = preprocess.feature_engineering(train_data)
    print("Feature Importances:\n", preprocess.determine_feature_importance(train_data))

    # Testing Data
    test_data = preprocess.null_value_analysis(test_data)
    test_data = preprocess.duplicate_value_analysis(test_data)
    test_data = preprocess.outlier_analysis(test_data)
    test_data = preprocess.feature_engineering(test_data)

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = preprocess.perform_train_test_split(train_data)

    print("Data Pre-processing Completed.")
    print("--------------------------------------------------")

    
    # STEP 2: MODELLING
    model_instance = modelling.Modelling()
    # Train and evaluate KNN
    knn_params = {'n_neighbors': [3, 5, 7]}
    knn_model = model_instance.train_model(x_train, y_train, 'knn', param_grid=knn_params)
    knn_accuracy = model_instance.evaluate_model(knn_model, x_test, y_test)
    print("KNN Model Accuracy:", knn_accuracy)

    # Train and evaluate Decision Tree
    dt_params = {'max_depth': [5, 10, 15]}
    dt_model = model_instance.train_model(x_train, y_train, 'decision_tree', param_grid=dt_params)
    dt_accuracy = model_instance.evaluate_model(dt_model, x_test, y_test)
    print("Decision Tree Model Accuracy:", dt_accuracy)

    # Train and evaluate Random Forest
    rf_model = model_instance.train_model(x_train, y_train, 'random_forest')
    rf_accuracy = model_instance.evaluate_model(rf_model, x_test, y_test)
    print("Random Forest Model Accuracy:", rf_accuracy)

    # Perform KMeans clustering on the entire dataset (example)
    kmeans_params = {'n_clusters': [2, 3, 4]}
    kmeans_model, cluster_labels = model_instance.perform_clustering(train_data, 'kmeans', param_grid=kmeans_params)
    # print("KMeans Cluster Centers:", kmeans_model.cluster_centers_)

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
