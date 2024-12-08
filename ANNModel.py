import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ANNModel:
    def __init__(self, datasetPath, targetVariable, learning_rate, epochs, patience, batch_size):
        """
        Initializes the ANN Model.

        Parameters:
        - datasetPath (str): Path to the preprocessed dataset.
        - targetVariable (str): The target variable of the dataset.
        - learning_rate (float): Learning rate for backpropagation.
        - epochs (int): Number of training epochs.
        - patience (int): Patience for early stopping.
        - batch_size (int): Batch size for training.
        """
        self.datasetPath = datasetPath
        self.targetVariable = targetVariable
        self.lr = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.model = None
        self.best_model_path = "best_ann_model.pth"
        self.train_losses = []  # Store training losses
        self.val_losses = []    # Store validation losses

    class NeuralNet(nn.Module):
        def __init__(self, input_dim):
            """
            Initializes the ANN structure.
            Parameters:
            - input_dim (int): Number of input features.
            """
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(64, 32)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            return self.sigmoid(x)

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the ANN model.

        Parameters:
        - X_train, y_train: Training data and labels.
        - X_val, y_val: Validation data and labels.
        """
        input_dim = X_train.shape[1]
        self.model = self.NeuralNet(input_dim)

        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float("inf")
        no_improve_epochs = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = self.model(X_batch).squeeze()
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()

            # Store losses for plotting
            self.train_losses.append(train_loss / len(train_loader))
            self.val_losses.append(val_loss / len(val_loader))

            print(
                f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {self.train_losses[-1]:.4f}, "
                f"Val Loss: {self.val_losses[-1]:.4f}"
            )

            # Early Stopping
            if val_loss / len(val_loader) < best_val_loss:
                best_val_loss = val_loss / len(val_loader)
                no_improve_epochs = 0
                torch.save(self.model.state_dict(), self.best_model_path)  # Save best model
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= self.patience:
                print("Early stopping triggered.")
                break

    def predict(self, X):
        """
        Makes predictions using the ANN model.

        Parameters:
        - X: Input features.

        Returns:
        - y_pred: Predicted labels.
        """
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).squeeze()
            y_pred_labels = (y_pred >= 0.5).float()
        return y_pred_labels

    def evaluateMetrics(self, y_true, y_pred):
        """
        Calculates classification metrics: accuracy, precision, recall, and F1-score.

        Parameters:
        - y_true: True labels.
        - y_pred: Predicted labels.

        Returns:
        - dict: Metrics including accuracy, precision, recall, and F1-score.
        """
        accuracy = accuracy_score(y_true.numpy(), y_pred.numpy())
        precision = precision_score(y_true.numpy(), y_pred.numpy())
        recall = recall_score(y_true.numpy(), y_pred.numpy())
        f1 = f1_score(y_true.numpy(), y_pred.numpy())
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    def plot_loss(self):
        """
        Plots the training and validation loss.
        """
        plt.plot(range(len(self.train_losses)), self.train_losses, label='Training Loss')
        plt.plot(range(len(self.val_losses)), self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross-Entropy Loss')
        plt.legend()
        plt.title('Epochs vs Loss')
        plt.grid(True)
        plt.show()

    def printRelevantFeatures(self, featureNames, topN):
        """
        Prints the most relevant features based on weights in the first hidden layer.

        Parameters:
        - featureNames (list): List of feature names.
        - topN (int): Number of top features to print.
        """
        fc1_weights = self.model.fc1.weight.detach().numpy()
        feature_importance = np.sum(np.abs(fc1_weights), axis=0)
        sorted_indices = np.argsort(feature_importance)[::-1]
        top_features = [(featureNames[i], feature_importance[i]) for i in sorted_indices[:topN]]

        print("-" * 150)
        print(f"Top {topN} Relevant Features:")
        for feature, importance in top_features:
            print(f"    Feature: {feature}, Importance: {importance:.4f}")
        print("-" * 150)

    def trainAndEvaluate(self):
        """
        Trains and evaluates the ANN model on the dataset.
        """
        # Load dataset
        df = pd.read_csv(self.datasetPath)
        X = df.drop(self.targetVariable, axis=1).values
        y = df[self.targetVariable].values

        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        # Train the model
        print("Training ANN...")
        self.fit(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

        # Evaluate model
        print("Evaluating metrics...")
        train_pred = self.predict(X_train_tensor)
        val_pred = self.predict(X_val_tensor)
        train_metrics = self.evaluateMetrics(y_train_tensor, train_pred)
        val_metrics = self.evaluateMetrics(y_val_tensor, val_pred)

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

        # Plot losses
        self.plot_loss()

        # Print relevant features
        featureNames = list(df.drop(columns=[self.targetVariable]).columns)
        self.printRelevantFeatures(featureNames, topN=10)


if __name__ == "__main__":
    datasetPath = "./PreprocessedDataset.csv"
    targetVariable = "LoanApproved"
    annModel = ANNModel(datasetPath, targetVariable, learning_rate=0.001, epochs=100, patience=5, batch_size=32)
    annModel.trainAndEvaluate()
