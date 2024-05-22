import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

class Modelling:
    def __init__(self, feature_importances=None):
        self.models = {
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier(),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }
        self.best_params = {}
        self.trained_models = {}

        if feature_importances is not None:
            print("Feature importances provided.")
            self.feature_importances = feature_importances
        else:
            print("Feature importances not provided. Setting to 1 for all features.")
            self.feature_importances = 1
    
    def train_model(self, x_train, y_train, model_name, param_grid=None):
        model = self.models[model_name]

        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=4, scoring='roc_auc')
            grid_search.fit(x_train, y_train)
            model = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
        else:
            model.fit(x_train, y_train)
        
        self.trained_models[model_name] = model
        return model

    def evaluate_model(self, model, x_test, y_test):
        # Make predictions
        predictions = model.predict(x_test)

        # Calculate overall accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Calculate the number of correct predictions for each class
        correct_1 = sum((predictions == 1) & (y_test == 1))
        correct_0 = sum((predictions == 0) & (y_test == 0))

        # Calculate the total number of predictions for each class
        total_1 = sum(predictions == 1)
        total_0 = sum(predictions == 0)

        # Calculate the percentage of correct predictions for each class
        percentage_correct_1 = (correct_1 / total_1) * 100 if total_1 > 0 else 0
        percentage_correct_0 = (correct_0 / total_0) * 100 if total_0 > 0 else 0

        return accuracy, percentage_correct_1, percentage_correct_0

    def perform_clustering(self, data, model_name, param_grid=None):
        if model_name == 'kmeans':
            model = KMeans()
            if param_grid:
                grid_search = GridSearchCV(model, param_grid, cv=5)
                grid_search.fit(data)
                model = grid_search.best_estimator_
                self.best_params[model_name] = grid_search.best_params_
            else:
                model.fit(data)
            
            self.trained_models[model_name] = model
            labels = model.predict(data)
            return model, labels
          