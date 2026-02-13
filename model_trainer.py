import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.models = {'LinearRegression': LinearRegression(),
                       'RandomForest': RandomForestRegressor(),
                       'GradientBoosting': GradientBoostingRegressor(),
                       'SVM': SVR()}
        self.train_results = {}

    def prepare_data(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_models(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            self.train_results[model_name] = {'MSE': mse, 'R2': r2}
        return self.train_results

    def get_results(self):
        return self.train_results

# Example usage:
# data = pd.read_csv('your_dataset.csv')  # Load your dataset here
# trainer = ModelTrainer(data, 'target_column_name')
# results = trainer.train_models()
# print(results)