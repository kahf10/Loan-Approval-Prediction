import numpy as np
import pandas as pd


class PolynomialSVM:
    def __init__(self, datasetPath, targetVariable, n_components=0.95, C=1.0):
        """
        Initializes the SVM Model with PCA.

        Parameters:
        datasetPath (str): Path to the preprocessed dataset.
        targetVariable (str): The target variable of the dataset.
        n_components (float or int): Number of principal components or percentage of variance to retain.
        C (float): Regularization parameter for the SVM.
        """
        self.datasetPath = datasetPath
        self.targetVariable = targetVariable
        self.n_components = n_components
        self.C = C

    def performPCA(self, X):
        """
        Performs PCA on the dataset to reduce dimensionality.

        Parameters:
        - X: Feature matrix.

        Returns:
        - X_reduced: Feature matrix with reduced dimensions.
        - explained_variance_ratio: Percentage of variance explained by each component.
        """
        print("Performing PCA...")
        X_centered  = X

        # Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Determine the number of components to retain
        if isinstance(self.n_components, float):  # Retain by variance ratio
            total_variance = np.sum(eigenvalues)
            cumulative_variance = np.cumsum(eigenvalues) / total_variance
            n_components = np.searchsorted(cumulative_variance, self.n_components) + 1
        else:
            n_components = self.n_components

        print(f"Number of components retained: {n_components}")

        # Select top components
        principal_components = eigenvectors[:, :n_components]

        # Project the data onto the principal components
        X_reduced = X_centered @ principal_components

        return X_reduced, eigenvalues[:n_components] / np.sum(eigenvalues)

    @staticmethod
    def kernelFunction(X1, X2):
        """
        Computes the linear kernel between two datasets.

        Parameters:
        - X1: First dataset
        - X2: Second dataset

        Returns:
        - Kernel matrix
        """
        print("Computing kernel function...")
        degree = 2  # Choose the degree of the polynomial
        return (X1 @ X2.T + 1) ** degree

    def trainSVM(self, X_train, y_train):
        """
        Train the SVM using direct computation of alpha values.

        Parameters:
        - X_train: Training features.
        - y_train: Training labels.

        Returns:
        - alpha: Computed alpha values.
        """
        print("Training SVM...")

        # Convert labels to diagonal matrix
        y_train_diag = np.diag(y_train)

        # Compute the kernel matrix
        kernel_matrix = self.kernelFunction(X_train, X_train)

        # Solve for alpha
        alpha = np.linalg.pinv(y_train_diag @ kernel_matrix @ y_train_diag) @ np.ones((y_train_diag.shape[0], 1))
        return alpha

    def predict(self, X_train, X_val, y_train, alpha):
        """
        Predicts labels for validation data.

        Parameters:
        - X_train: Training features
        - X_val: Validation features
        - y_train: Training labels
        - alpha: Computed alpha values

        Returns:
        - Predicted labels for validation data
        """
        print("Predicting labels...")

        # Compute the kernel matrix for validation
        kernelVal = self.kernelFunction(X_val, X_train)

        # Compute predictions
        predictions = kernelVal @ (np.diag(y_train) @ alpha)
        predictions = np.array(predictions).flatten()  # Flatten predictions

        # Convert to binary labels
        return np.where(predictions >= 0, 1, -1)

    @staticmethod
    def evaluateMetrics(y_true, y_pred):
        """
        Calculates classification metrics: accuracy, precision, recall, and F1-score.

        Parameters:
        - y_true: True labels
        - y_pred: Predicted labels.

        Returns:
        - Dictionary of metrics
        """
        print("Evaluating metrics...")
        accuracy = np.mean(y_true == y_pred)
        precision = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1)
        recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}

    def trainAndEvaluate(self):
        """
        Trains and evaluates the SVM model with PCA.
        """
        # Load the dataset
        df = pd.read_csv(self.datasetPath)
        X = df.drop(self.targetVariable, axis=1).to_numpy()
        y = df[self.targetVariable].to_numpy()

        # Transform labels from {0, 1} to {-1, 1}
        y = np.where(y == 0, -1, 1)

        # Perform PCA
        X_reduced, explained_variance = self.performPCA(X)

        # Split data
        split_index = int(0.67 * len(X_reduced))
        X_train, X_val = X_reduced[:split_index], X_reduced[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        # Train the SVM
        alpha = self.trainSVM(X_train, y_train)

        # Predict on training and validation data
        y_pred_train = self.predict(X_train, X_train, y_train, alpha)
        y_pred_val = self.predict(X_train, X_val, y_train, alpha)

        # Evaluate metrics
        train_metrics = self.evaluateMetrics(y_train, y_pred_train)
        val_metrics = self.evaluateMetrics(y_val, y_pred_val)

        print("_" * 150)
        print("Training Metrics:")
        for metric, value in train_metrics.items():
            print(f"    {metric.capitalize()}: {value:.4f}")
        print("_" * 150)
        print("Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"    {metric.capitalize()}: {value:.4f}")
        print("_" * 150)


if __name__ == "__main__":
    datasetPath = "./PreprocessedDataset.csv"
    targetVariable = "LoanApproved"
    svmModel = PolynomialSVM(datasetPath, targetVariable)
    svmModel.trainAndEvaluate()