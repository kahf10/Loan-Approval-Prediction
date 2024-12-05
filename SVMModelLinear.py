import numpy as np
import pandas as pd


class LinearSVM:
    def __init__(self, datasetPath, targetVariable, learning_rate, lambda_param, n_iters):
        """
        Initializes the Custom SVM Model.

        Parameters:
        - datasetPath (str): Path to the preprocessed dataset.
        - targetVariable (str): The target variable of the dataset.
        - learning_rate (float): Learning rate for weight updates.
        - lambda_param (float): Regularization parameter.
        - n_iters (int): Number of iterations for training.
        """
        self.datasetPath = datasetPath
        self.targetVariable = targetVariable
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Fits the SVM model using the dataset.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target labels (expected to be -1 and 1).
        """
        n_samples, n_features = X.shape

        # Convert labels to -1 and 1 if not already
        y = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        """
        Predicts labels for the dataset.

        Parameters:
        - X (np.ndarray): Feature matrix.

        Returns:
        - np.ndarray: Predicted labels.
        """
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def evaluateMetrics(self, y_true, y_pred):
        """
        Calculates classification metrics: accuracy, precision, recall, and F1-score.

        Parameters:
        - y_true (np.ndarray): True labels.
        - y_pred (np.ndarray): Predicted labels.

        Returns:
        - dict: Metrics including accuracy, precision, recall, and F1-score.
        """
        accuracy = np.mean(y_true == y_pred)
        precision = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) else 0
        recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}

    def trainAndEvaluate(self):
        """
        Trains and evaluates the SVM model on the loan approval dataset.
        """
        # Load the dataset
        df = pd.read_csv(self.datasetPath)
        X = df.drop(self.targetVariable, axis=1).to_numpy()
        y = df[self.targetVariable].to_numpy()

        # Convert labels to -1 and 1
        y = np.where(y == 0, -1, 1)

        # Split the dataset into training and validation sets
        split_index = int(0.67 * len(X))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        # Train the SVM
        print("Training SVM...")
        self.fit(X_train, y_train)

        # Predict on training and validation data
        y_pred_train = self.predict(X_train)
        y_pred_val = self.predict(X_val)

        # Evaluate metrics
        print("Evaluating metrics...")
        train_metrics = self.evaluateMetrics(y_train, y_pred_train)
        val_metrics = self.evaluateMetrics(y_val, y_pred_val)

        # Print metrics
        print("-" * 150)
        print("Training Metrics:")
        for metric, value in train_metrics.items():
            print(f"    {metric.capitalize()}: {value:.4f}")
        print("-" * 150)
        print("Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"    {metric.capitalize()}: {value:.4f}")
        print("-" * 150)
