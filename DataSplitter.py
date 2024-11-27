import pandas as pd
import numpy as np

class DataSplitter:
    def __init__(self, datasetPath, targetVariable):
        """
        Initializes the DataSplitter with the dataset and target variable.

        Parameters:
        - datasetPath (str): Path to the preprocessed dataset.
        - targetVariable (str): The target variable for splitting.
        """
        self.dataset = pd.read_csv(datasetPath)
        self.targetVariable = targetVariable

    def splitData(self, testSize=1 / 3, randomSeed=42):
        """
        Splits the dataset into training and validation sets manually.
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
        X_train = trainSet.drop(columns=[self.targetVariable], errors='ignore')
        y_train = trainSet[self.targetVariable].values.ravel()  # Ensure 1D array
        X_val = valSet.drop(columns=[self.targetVariable], errors='ignore')
        y_val = valSet[self.targetVariable].values.ravel()  # Ensure 1D array

        return X_train, y_train, X_val, y_val
