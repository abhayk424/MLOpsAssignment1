# test_model.py
import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def test_model_training():
    # Load data
    boston = fetch_california_housing()
    X, y = boston.data, boston.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Check if the model is trained successfully
    assert model is not None, "Model training failed"

    # Save model
    joblib.dump(model, 'test_model.joblib')

    # Check if the model file is created
    assert os.path.exists('test_model.joblib'), \
           "Model file was not created"


def test_model_prediction():
    # Load the saved model
    model = joblib.load('test_model.joblib')

    # Load data
    boston = fetch_california_housing()
    X, _ = boston.data, boston.target

    # Make predictions
    predictions = model.predict(X[:5])

    # Check if predictions are made successfully
    assert len(predictions) == 5, "Model did not make predictions correctly"

    # Clean up the saved model file
    os.remove('test_model.joblib')
