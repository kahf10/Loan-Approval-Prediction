import numpy as np
import pandas as pd


class PolynomialSVM:
    def __init__(self, datasetPath, targetVariable, learning_rate, lambda_param, n_iters, degree=2, coef0=1):
        """
        Initializes the Custom Polynomial SVM Model.

        Parameters:
        - datasetPath (str): Path to the preprocessed dataset.
        - targetVariable (str): The target variable of the dataset.
        - learning_rate (float): Learning rate for weight updates.
        - lambda_param (float): Regularization parameter.
        - n_iters (int): Number of iterations for training.
        - degree (int): Degree of the polynomial kernel.
        - coef0 (float): Independent term in the kernel function.
        """
        self.datasetPath = datasetPath
        self.targetVariable = targetVariable
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.degree = degree
        self.coef0 = coef0
        self.alpha = None
        self.b = 0

    def polynomial_kernel(self, x, y):
        """
        Computes the polynomial kernel between two vectors.

        Parameters:
        - x (np.ndarray): First vector.
        - y (np.ndarray): Second vector.

        Returns:
        - float: Polynomial kernel result.
        """
        return (np.dot(x, y) + self.coef0) ** self.degree

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

        # Initialize alpha values (dual coefficients) and bias
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # Training loop
        for _ in range(self.n_iters):
            for i in range(n_samples):
                condition = y[i] * (self._decision_function(X[i], X, y)) >= 1
                if condition:
                    self.alpha[i] -= self.lr * (2 * self.lambda_param * self.alpha[i])
                else:
                    self.alpha[i] -= self.lr * (2 * self.lambda_param * self.alpha[i] - y[i] * self.polynomial_kernel(X[i], X[i]))
                    self.b -= self.lr * y[i]

    def _decision_function(self, x_i, X, y):
        """
        Computes the decision function for a given sample.

        Parameters:
        - x_i (np.ndarray): Input vector.
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target labels.

        Returns:
        - float: Decision function value.
        """
        result = 0
        for alpha_j, x_j, y_j in zip(self.alpha, X, y):
            result += alpha_j * y_j * self.polynomial_kernel(x_i, x_j)
        return result - self.b

    def predict(self, X):
        """
        Predicts labels for the dataset.

        Parameters:
        - X (np.ndarray): Feature matrix.

        Returns:
        - np.ndarray: Predicted labels.
        """
        predictions = []
        for x_i in X:
            prediction = np.sign(self._decision_function(x_i, X, np.where(self.alpha > 0, 1, -1)))
            predictions.append(prediction)
        return np.array(predictions)

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
        print("Training SVM with Polynomial Kernel...")
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
