# Import necessary libraries
import joblib  # For loading and saving the model
import os  # For file operations
from sklearn.datasets import fetch_california_housing  # Import dataset
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.linear_model import LinearRegression  # Linear Regression model

def test_model_training():
    """
    Test the training of the Linear Regression model and the saving of the trained model.
    """

    # Load the California housing dataset
    california = fetch_california_housing()
    X, y = california.data, california.target  # Features and target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Check if the model is trained successfully
    assert model is not None, "Model training failed"

    # Save the trained model to a file
    joblib.dump(model, 'test_model.joblib')

    # Check if the model file is created successfully
    assert os.path.exists('test_model.joblib'), "Model file was not created"

def test_model_prediction():
    """
    Test the prediction capability of the saved Linear Regression model.
    """

    # Load the saved model from the file
    model = joblib.load('test_model.joblib')

    # Load the California housing dataset
    california = fetch_california_housing()
    X, _ = california.data, california.target  # Only features are needed for prediction

    # Make predictions on the first 5 samples of the dataset
    predictions = model.predict(X[:5])

    # Check if predictions are made successfully
    assert len(predictions) == 5, "Model did not make predictions correctly"

    # Clean up the saved model file
    os.remove('test_model.joblib')
