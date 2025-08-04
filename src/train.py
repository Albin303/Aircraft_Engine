import pandas as pd
import joblib
from model import get_model

def load_data(features_path, labels_path):
    """
    Load features (X) and target (y) from CSV files.
    """
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    return X, y.values.ravel()  # Ensure y is 1D array

if __name__ == "__main__":
    # Paths to train datasets
    X_train_path = 'data/X_train.csv'
    y_train_path = 'data/y_train.csv'

    # Load training data
    X_train, y_train = load_data(X_train_path, y_train_path)

    # Initialize model
    model = get_model()

    # Train model on training data
    model.fit(X_train, y_train)
    print("Model training completed successfully.")

    # Save trained model to disk
    model_path = 'models/rf_model.pkl'
    joblib.dump(model, model_path)
    print(f"Trained model saved at {model_path}.")
