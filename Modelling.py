import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

class Modelling:
    def __init__(self):
        self.models = {
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier(),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.best_params = {}
        self.trained_models = {}

    def train_model(self, x_train, y_train, model_name, param_grid=None):
        model = self.models[model_name]
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(x_train, y_train)
            model = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
        else:
            model.fit(x_train, y_train)
        
        self.trained_models[model_name] = model
        return model

    def evaluate_model(self, model, x_test, y_test):
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

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
        