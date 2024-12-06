import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, datasetPath, targetVariable):
        """
        Initializes the Logistic Regression model.

        Parameters:
        - datasetPath (str): Path to the preprocessed dataset.
        - targetVariable (str): The target variable for training.
        """
        self.datasetPath = datasetPath
        self.targetVariable = targetVariable
        self.dataset = pd.read_csv(datasetPath)

    def splitData(self, randomSeed=42, testSize=1/3):
        """
        Splits the dataset into training and validation sets manually.

        Parameters:
        - randomSeed (int): Seed for random number generator for reproducibility.
        - testSize (float): Fraction of the dataset to be used as validation set.

        Returns:
        - X_train, y_train, X_val, y_val: Split feature and target arrays.
        """
        print("Splitting dataset into training and validation sets...")

        # Set random seed for reproducibility
        np.random.seed(randomSeed)

        # Shuffle the dataset indices
        shuffledIndices = np.random.permutation(len(self.dataset))

        # Determine split index
        splitIndex = int(len(self.dataset) * (1 - testSize))

        # Split indices for training and validation sets
        trainIndices = shuffledIndices[:splitIndex]
        valIndices = shuffledIndices[splitIndex:]

        # Create training and validation datasets
        trainSet = self.dataset.iloc[trainIndices]
        valSet = self.dataset.iloc[valIndices]

        # Debug: Confirm target column exists
        if self.targetVariable not in self.dataset.columns:
            raise ValueError(f"Target variable '{self.targetVariable}' not found!")

        # Separate features and target
        X_train = trainSet.drop(columns=[self.targetVariable], errors='ignore').to_numpy()
        y_train = trainSet[self.targetVariable].to_numpy()
        X_val = valSet.drop(columns=[self.targetVariable], errors='ignore').to_numpy()
        y_val = valSet[self.targetVariable].to_numpy()

        return X_train, y_train, X_val, y_val

    @staticmethod
    def computeYHat(z):
        """
        Logistic function to compute predictions.
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def calculateMetrics(y_true, y_pred, threshold=0.5):
        """
        Calculates classification metrics: accuracy, precision, recall, and F1-score.
        """
        y_pred_binary = (y_pred >= threshold).astype(int)
        TP = np.sum((y_true == 1) & (y_pred_binary == 1))
        TN = np.sum((y_true == 0) & (y_pred_binary == 0))
        FP = np.sum((y_true == 0) & (y_pred_binary == 1))
        FN = np.sum((y_true == 1) & (y_pred_binary == 0))

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

        return accuracy, precision, recall, f1

    def trainModel(self, X_train, y_train, X_val, y_val, learning_rate, epochs):
        """
        Trains the logistic regression model using gradient descent.
        """
        weights = np.zeros(X_train.shape[1])
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            # Predictions and loss for training data
            y_pred_train = self.computeYHat(np.dot(X_train, weights))
            log_loss_train = -np.mean(
                y_train * np.log(y_pred_train + 1e-14) + (1 - y_train) * np.log(1 - y_pred_train + 1e-14)
            )
            train_losses.append(log_loss_train)

            # Gradient descent
            dw = np.dot(X_train.T, (y_pred_train - y_train)) / len(y_train)
            weights -= learning_rate * dw

            # Validation loss
            y_pred_val = self.computeYHat(np.dot(X_val, weights))
            log_loss_val = -np.mean(
                y_val * np.log(y_pred_val + 1e-14) + (1 - y_val) * np.log(1 - y_pred_val + 1e-14)
            )
            val_losses.append(log_loss_val)

        return weights, train_losses, val_losses, y_pred_train, y_pred_val

    def printRelevantFeatures(self, weights, featureNames, topN=5):
        """
        Prints the most relevant features based on their weights.

        Parameters:
        - weights (np.ndarray): The weights of the model (excluding the bias term).
        - featureNames (list): List of feature names corresponding to the dataset columns.
        - topN (int): The number of top features to print.
        """
        # Exclude the bias term (first weight)
        featureWeights = weights[1:]  # Exclude the bias term
        featureImportance = abs(featureWeights)

        # Sort features by their importance
        sortedIndices = np.argsort(featureImportance)[::-1]
        topFeatures = [(featureNames[i], featureWeights[i]) for i in sortedIndices[:topN]]

        print("-" * 150)
        print(f"Top {topN} Relevant Features:")
        for feature, weight in topFeatures:
            print(f"    Feature: {feature}, Weight: {weight:.4f}")
        print("-" * 150)

    def trainAndEvaluate(self, learning_rate=0.1, epochs=3000, topNFeatures = 10):
        """
        Trains and evaluates the Logistic Regression model.
        """
        # Split dataset
        X_train, y_train, X_val, y_val = self.splitData(randomSeed=42)

        # Add bias term to features
        X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
        X_val = np.column_stack((np.ones(X_val.shape[0]), X_val))

        # Train the model
        weights, train_losses, val_losses, y_pred_train, y_pred_val = self.trainModel(
            X_train, y_train, X_val, y_val, learning_rate, epochs
        )

        # Calculate and print metrics
        train_accuracy, train_precision, train_recall, train_f1 = self.calculateMetrics(y_train, y_pred_train)
        val_accuracy, val_precision, val_recall, val_f1 = self.calculateMetrics(y_val, y_pred_val)

        print("-" * 150)
        print("Final Training Metrics:")
        print(f"    Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print("-" * 150)
        print("Final Validation Metrics:")
        print(f"    Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print("-" * 150)

        # Print relevant features
        featureNames = list(self.dataset.drop(columns=[self.targetVariable], errors='ignore').columns)
        self.printRelevantFeatures(weights, featureNames, topN=topNFeatures)

        # Loss plot
        plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
        plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.title('Epoch vs Log-Loss')
        plt.show()

if __name__ == "__main__":
    datasetPath = "./PreprocessedDataset.csv"
    targetVariable = "LoanApproved"
    logisticModel = LogisticRegression(datasetPath, targetVariable)
    logisticModel.trainAndEvaluate()
