import numpy as np
from DataSplitter import DataSplitter
import matplotlib.pyplot as plt

class LogisticRegressionModel:
    def __init__(self, datasetPath, targetVariable):
        """
        Initializes the Logistic Regression model.

        Parameters:
        - datasetPath (str): Path to the preprocessed dataset.
        - targetVariable (str): The target variable for training.
        """
        self.datasetPath = datasetPath
        self.targetVariable = targetVariable

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

    def trainAndEvaluate(self, learning_rate=0.1, epochs=3000):
        """
        Trains and evaluates the Logistic Regression model.
        """
        # Split dataset using DataSplitter
        splitter = DataSplitter(self.datasetPath, self.targetVariable)
        X_train, y_train, X_val, y_val = splitter.splitData(testSize=0.2, randomSeed=42)

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

        print("Final Training Metrics:")
        print(f"Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print("_" * 80)
        print("Final Validation Metrics:")
        print(f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print("_" * 80)

        # Loss plot
        plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
        plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.title('Epoch vs Log-Loss')
        plt.show()
