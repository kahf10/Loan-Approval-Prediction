from Preprocessor import Preprocessor
from LogisticRegressionModel import LogisticRegressionModel


class MasterControl:
    def __init__(self, datasetPath, preprocessedPath):
        self.datasetPath = datasetPath
        self.preprocessedPath = preprocessedPath

    def runPreprocessing(self):
        """
        Executes the preprocessing pipeline and saves the clean dataset.
        """
        print("-" * 150)
        print("-" * 150)

        print("Starting the preprocessing pipeline...")
        preprocessor = Preprocessor(self.datasetPath)
        preprocessor.runPipeline()

        print("-" * 150)

    def runModels(self):
        """
        Executes model training and evaluation.
        """
        print("Starting model training and evaluation...")

        # Example: Logistic Regression Model
        logisticModel = LogisticRegressionModel(self.preprocessedPath, "LoanApproved")
        logisticModel.trainAndEvaluate()

        # Add other models here as needed
        print("-" * 150)

    def run(self):
        """
        Main entry point for the pipeline.
        """
        self.runPreprocessing()
        self.runModels()


if __name__ == "__main__":
    datasetPath = "./dataset.csv"  # Path to the raw dataset
    preprocessedPath = "./PreprocessedDataset.csv"  # Path for the preprocessed dataset
    masterControl = MasterControl(datasetPath, preprocessedPath)
    masterControl.run()
