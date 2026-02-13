# Advanced Data Preprocessing Module
# Handles data cleaning, transformation, and feature engineering

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for ML pipelines.
    
    Features:
    - Missing value imputation
    - Feature scaling
    - Categorical encoding
    - Outlier detection and removal
    - Feature engineering
    """
    
    def __init__(self, scaler_type='standard', missing_strategy='mean'):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type (str): 'standard' or 'minmax'
            missing_strategy (str): 'mean', 'median', or 'drop'
        """
        self.scaler_type = scaler_type
        self.missing_strategy = missing_strategy
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        
        logger.info(f"DataPreprocessor initialized with scaler: {scaler_type}")
    
    def load_data(self, filepath):
        """Load data from CSV file."""
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
    
    def handle_missing_values(self, X, y=None):
        """Handle missing values in the dataset."""
        logger.info(f"Handling missing values with strategy: {self.missing_strategy}")
        
        if self.missing_strategy == 'drop':
            X = X.dropna()
            if y is not None:
                y = y.loc[X.index]
        else:
            imputer = SimpleImputer(strategy=self.missing_strategy)
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        logger.info(f"After missing value handling - Shape: {X.shape}")
        return X, y
    
    def remove_outliers(self, X, threshold=3):
        """Remove outliers using z-score method."""
        logger.info(f"Removing outliers with threshold: {threshold}")
        
        z_scores = np.abs((X - X.mean()) / X.std())
        mask = (z_scores < threshold).all(axis=1)
        
        logger.info(f"Removed {(~mask).sum()} outliers")
        return X[mask]
    
    def encode_categorical(self, X, fit=True):
        """Encode categorical features."""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        logger.info(f"Encoding categorical features: {list(categorical_cols)}")
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        return X
    
    def scale_features(self, X, fit=True):
        """Scale numerical features."""
        logger.info(f"Scaling features with {self.scaler_type} scaler")
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def feature_engineering(self, X):
        """Create new features from existing ones."""
        logger.info("Performing feature engineering")
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        # Create polynomial features for numerical columns
        for col in numerical_cols:
            X[f'{col}_squared'] = X[col] ** 2
            X[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
        
        logger.info(f"New features created. Total features: {X.shape[1]}")
        return X
    
    def preprocess(self, X, y=None, fit=True):
        """
        Complete preprocessing pipeline.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            fit (bool): Whether to fit scalers/encoders
        
        Returns:
            tuple: Preprocessed X and y
        """
        logger.info("Starting preprocessing pipeline")
        
        # Handle missing values
        X, y = self.handle_missing_values(X, y)
        
        # Remove outliers
        if fit:
            X = self.remove_outliers(X)
            if y is not None:
                y = y.loc[X.index]
        
        # Encode categorical features
        X = self.encode_categorical(X, fit=fit)
        
        # Feature engineering
        X = self.feature_engineering(X)
        
        # Scale features
        X = self.scale_features(X, fit=fit)
        
        logger.info("Preprocessing completed")
        return X, y
    
    def load_and_preprocess(self, filepath, target_col='target', test_size=0.2, 
                           random_state=42):
        """Complete pipeline: load data, separate features and target, preprocess."""
        # Load data
        data = self.load_data(filepath)
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Preprocess
        X_train, y_train = self.preprocess(X_train, y_train, fit=True)
        X_test, y_test = self.preprocess(X_test, y_test, fit=False)
        
        logger.info(f"Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(scaler_type='standard', missing_strategy='mean')
    
    # Example: Load and preprocess data
    # X_train, X_test, y_train, y_test = preprocessor.load_and_preprocess('data.csv')