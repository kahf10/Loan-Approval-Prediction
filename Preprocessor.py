import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Preprocessor:
    def __init__(self, datasetPath):
        """
        Initialize the Preprocessor class and load the dataset.
        params:
        datasetPath: Path to the dataset.
        """
        self.datasetPath = datasetPath
        self.data = pd.read_csv(self.datasetPath)

    def dropUnnecessaryColumns(self):
        """
        Drops unnecessary columns to simplify the dataset.
        """
        print("Dropping unnecessary columns...")
        columnsToDrop = ['ApplicationDate', 'RiskScore']
        self.data.drop(columns=columnsToDrop, inplace=True, errors='ignore')
        print(f"Columns dropped: {columnsToDrop}")

    def summarizeDataset(self):
        """
        Summarizes numerical and categorical features in the dataset.
        """
        print("Summarizing dataset...")
        numericalFeatures = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categoricalFeatures = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        print("\nNumerical features:")
        for feature in numericalFeatures:
            print(f"{feature} (Unique: {self.data[feature].nunique()})")

        print("\nCategorical features:")
        for feature in categoricalFeatures:
            print(f"{feature} (Unique: {self.data[feature].nunique()})")

    def printCategoricalFeatureValues(self):
        """
        Prints unique values for specified categorical features.
        """
        print("Analyzing categorical features...")
        categoricalFeatures = ["EmploymentStatus", "EducationLevel", "MaritalStatus", "HomeOwnershipStatus",
                               "LoanPurpose"]
        for feature in categoricalFeatures:
            if feature in self.data.columns:
                print(f"\nFeature: {feature}")
                print(f"Unique Values: {self.data[feature].unique()}")

    def ordinalEncode(self):
        """
        Ordinal encodes a single categorical feature.
        """
        ordinalMappings = {
            "EmploymentStatus": {'Unemployed': 1, 'Self-Employed': 2, 'Employed': 3},
            "EducationLevel": {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5},
            "MaritalStatus": {'Single': 1, 'Divorced': 2, 'Widowed': 3, 'Married': 4},
            "HomeOwnershipStatus": {'Other': 1, 'Rent': 2, 'Mortgage': 3, 'Own': 4}
        }

        for feature, mapping in ordinalMappings.items():
            print(f"Ordinal encoding feature: {feature}")
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].map(mapping)
                print(f"Applied mapping for {feature}: {mapping}")
            else:
                print(f"Feature {feature} not found in the dataset.")

    def oneHotEncode(self):
        """
        One-hot encodes specified categorical features.
        """
        print("One-hot encoding features...")
        featuresToEncode = ['LoanPurpose']
        self.data = pd.get_dummies(self.data, columns=featuresToEncode, drop_first=True)

        #Convert all new one-hot encoded columns to integers (1/0)
        for feature in featuresToEncode:
            oneHotCols = [col for col in self.data.columns if feature in col]
            self.data[oneHotCols] = self.data[oneHotCols].astype(int)

        print(f"One-hot encoded features: {featuresToEncode}")

    def normalizeNumericalFeatures(self):
        """
        Normalizes numerical features (excluding binary features) using standard scaling.
        """
        # Identify binary and ordinal features to exclude from normalization
        binaryFeatures = [col for col in self.data.columns if self.data[col].nunique() == 2]
        ordinalFeatures = ["EmploymentStatus", "EducationLevel", "MaritalStatus", "HomeOwnershipStatus"]
        featuresToExclude = set(binaryFeatures + ordinalFeatures)

        # Identify numerical features to normalize
        numericalFeatures = [
            col for col in self.data.select_dtypes(include=['int64', 'float64']).columns
            if col not in featuresToExclude
        ]

        print(f"Features excluded from normalization: {featuresToExclude}")
        print(f"Features to be normalized: {numericalFeatures}")

        for feature in numericalFeatures:
            mean = self.data[feature].mean()
            std = self.data[feature].std()
            self.data[feature] = (self.data[feature] - mean) / std

        print("Numerical features normalized successfully.")

    def plotCorrelationMatrix(self):
        """
        Plots a heatmap of the correlation matrix for numerical features.
        """
        print("Plotting correlation matrix heatmap...")
        plt.figure(figsize=(21, 21))
        correlationMatrix = self.data.corr()
        sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix Heatmap")
        plt.show()

    def printStrongCorrelations(self):
        """
        Prints feature pairs with strong correlations above the threshold.
        """
        threshold = 0.7
        print(f"Finding feature pairs with correlation above {threshold}...")
        correlationMatrix = self.data.corr()
        for i in range(len(correlationMatrix.columns)):
            for j in range(i + 1, len(correlationMatrix.columns)):
                corr = correlationMatrix.iloc[i, j]
                if abs(corr) > threshold:
                    print(f"{correlationMatrix.columns[i]} and {correlationMatrix.columns[j]}: {corr:.2f}")

    def dropCollinearColumns(self):
        """
        Drops collinear columns to avoid redundancy.
        """
        print("Dropping collinear columns...")
        columnsToDrop = ['Experience', 'AnnualIncome', 'TotalAssets', 'LoanAmount', 'BaseInterestRate']
        self.data.drop(columns=columnsToDrop, inplace=True, errors='ignore')
        print(f"Columns dropped: {columnsToDrop}")

    def savePreprocessedData(self):
        """
        Saves the preprocessed dataset to a specified file path.
        """
        outputPath = './PreprocessedDataset.csv'
        print(f"Saving preprocessed dataset to {outputPath}...")
        self.data.to_csv(outputPath, index=False)
        print("Dataset saved successfully.")

    def runPipeline(self):
        """
        Executes the entire preprocessing pipeline in sequence.
        """
        lineLength = 150
        print("Running the preprocessing pipeline...")

        # Step 1: Drop unnecessary columns
        self.dropUnnecessaryColumns()
        print("-" * lineLength)

        # Step 2: Summarize the dataset
        self.summarizeDataset()
        print("-" * lineLength)

        # Step 3: Categorical feature analysis and encoding
        self.printCategoricalFeatureValues()
        print("-" * lineLength)

        # Step 3a: Ordinal encoding
        self.ordinalEncode()
        print("-" * lineLength)

        # Step 3b: One-hot encoding
        self.oneHotEncode()
        print("-" * lineLength)

        # Step 4: Normalize numerical features
        self.normalizeNumericalFeatures()
        print("-" * lineLength)

        # Step 5: Plot correlation matrix and analyze multi-collinearity
        self.plotCorrelationMatrix()
        print("-" * lineLength)
        self.printStrongCorrelations()
        print("-" * lineLength)

        # Step 6: Drop collinear columns
        self.dropCollinearColumns()
        print("-" * lineLength)

        # Step 7: Saving the Preprocessed dataset in a different file
        self.savePreprocessedData()
        print("-" * lineLength)

        print("Preprocessing pipeline completed.")
        return self.data