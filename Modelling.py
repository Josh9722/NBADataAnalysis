import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def train_model(x_train, y_train):

    # Split train data into train and validation sets
    train_features, val_features, train_target, val_target = train_test_split(train_features, train_target, test_size=0.2, random_state=42)

    # Train and test KNN model
    knn_predictions = knn_model(train_features, train_target, val_features)
    knn_accuracy = evaluate_accuracy(val_target, knn_predictions)
    print("KNN Accuracy:", knn_accuracy)

    # Train and test Decision Tree model
    dt_predictions = decision_tree_model(train_features, train_target, val_features)
    dt_accuracy = evaluate_accuracy(val_target, dt_predictions)
    print("Decision Tree Accuracy:", dt_accuracy)

    # Train and test K-Means model
    kmeans_predictions = k_means_model(train_features, val_features)
    kmeans_accuracy = evaluate_accuracy(val_target, kmeans_predictions)
    print("K-Means Accuracy:", kmeans_accuracy)

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

# def preprocess_data(df):
#     """
#     Preprocess the dataset: remove unnecessary columns and split into features and target.
#     """
#     features = df.drop(['Id', 'TARGET_5Yrs'], axis=1)
#     target = df['TARGET_5Yrs']
#     return features, target

def knn_model(train_features, train_target, test_features):
    """
    Train and test a KNN model.
    """
    knn = KNeighborsClassifier()
    knn.fit(train_features, train_target)
    predictions = knn.predict(test_features)
    return predictions

def decision_tree_model(train_features, train_target, test_features):
    """
    Train and test a Decision Tree model.
    """
    dt = DecisionTreeClassifier()
    dt.fit(train_features, train_target)
    predictions = dt.predict(test_features)
    return predictions

def k_means_model(train_features, test_features):
    """
    Train and test a K-Means clustering model.
    """
    kmeans = KMeans(n_clusters=2)  # assuming binary classification
    kmeans.fit(train_features)
    predictions = kmeans.predict(test_features)
    return predictions

def evaluate_accuracy(true_labels, predicted_labels):
    """
    Evaluate the accuracy of the model.
    """
    return accuracy_score(true_labels, predicted_labels)