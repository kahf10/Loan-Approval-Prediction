from Preprocessor import Preprocessor
from LogisticRegressionModel import LogisticRegressionModel
from SVMModelLinear import LinearSVM
from SVMModelPolynomial import PolynomialSVM


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
        print("-" * 150)

        # Logistic Regression Model

        print("\nRunning Logistic Regression Model...")
        print("-" * 150)
        logisticModel = LogisticRegressionModel(self.preprocessedPath, "LoanApproved")
        logisticModel.trainAndEvaluate()

        # print("\nRunning Linear SVM Model...")
        # print("-" * 150)
        # svmModel = LinearSVM("./PreprocessedDataset.csv", "LoanApproved", learning_rate=0.0001, lambda_param=0.005, n_iters=1200)
        # svmModel.trainAndEvaluate()

        # print("\n\nRunning Polynomial SVM Model...")
        # print("-" * 150)
        # svmModel = PolynomialSVM("./PreprocessedDataset.csv", "LoanApproved", learning_rate=0.0001, lambda_param=0.005, n_iters=1000)
        # svmModel.trainAndEvaluate()


        # Add other models here as needed
        print("-" * 150)

    def run(self):
        """
        Main entry point for the pipeline.
        """
        #self.runPreprocessing()
        self.runModels()


if __name__ == "__main__":
    datasetPath = "./dataset.csv"  # Path to the raw dataset
    preprocessedPath = "./PreprocessedDataset.csv"  # Path for the preprocessed dataset
    masterControl = MasterControl(datasetPath, preprocessedPath)
    masterControl.run()
