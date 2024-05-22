import PreProcessing as preprocess  # Ensure the module name is correct
import Modelling as modelling 
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
    feature_importance = preprocess.determine_feature_importance(train_data)

    # Testing Data
    test_data = preprocess.null_value_analysis(test_data)
    test_data = preprocess.duplicate_value_analysis(test_data)
    test_data = preprocess.outlier_analysis(test_data)
    test_data = preprocess.feature_engineering(test_data)

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = preprocess.perform_train_test_split(train_data)

    print("Data Pre-processing Completed.")
    print("--------------------------------------------------")

    return 

    # STEP 2: MODELLING
    
    model = modelling.train_model(x_train, y_train)
    accuracy = preprocess.evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
