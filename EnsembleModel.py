import numpy as np
from LogisticRegressionModel import LogisticRegression
from LinearSVMModel import LinearSVM


class Ensemble:
    def __init__(self, logit_model, svm_model):
        """
        Initializes the Ensemble model.

        Parameters:
        - logit_model: Trained instance of LogisticRegressionModel.
        - svm_model: Trained instance of LinearSVM.
        """
        self.logit_model = logit_model
        self.svm_model = svm_model

    def hardVote(self, logit_pred, svm_pred):
        """
        Performs hard voting by majority rule.

        Parameters:
        - logit_pred (np.ndarray): Predicted labels from Logistic Regression (0 or 1).
        - svm_pred (np.ndarray): Predicted labels from SVM (-1 or 1).

        Returns:
        - np.ndarray: Final ensemble prediction labels (0 or 1).
        """
        # Convert SVM predictions from -1/1 to 0/1 for consistency
        svm_pred_binary = (svm_pred == 1).astype(int)

        # Majority voting
        votes = np.stack((logit_pred, svm_pred_binary), axis=1)
        ensemble_pred = (np.sum(votes, axis=1) > 1).astype(int)
        return ensemble_pred

    def softVote(self, logit_probs, svm_decision):
        """
        Performs soft voting using weighted probabilities.

        Parameters:
        - logit_probs (np.ndarray): Predicted probabilities from Logistic Regression (0 to 1).
        - svm_decision (np.ndarray): Decision function values from SVM.

        Returns:
        - np.ndarray: Final ensemble prediction labels (0 or 1).
        """

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        # Convert SVM decision values to probabilities using the hardcoded sigmoid function
        svm_probs = sigmoid(svm_decision)

        # Average the probabilities
        combined_probs = (logit_probs + svm_probs) / 2

        # Convert averaged probabilities to binary predictions
        ensemble_pred = (combined_probs >= 0.5).astype(int)
        return ensemble_pred

    def evaluateEnsemble(self, y_true, ensemble_pred):
        """
        Evaluates the ensemble model.

        Parameters:
        - y_true (np.ndarray): True labels.
        - ensemble_pred (np.ndarray): Predicted labels from the ensemble.

        Returns:
        - dict: Metrics including accuracy, precision, recall, and F1-score.
        """
        accuracy = np.mean(y_true == ensemble_pred)
        precision = np.sum((ensemble_pred == 1) & (y_true == 1)) / np.sum(ensemble_pred == 1) if np.sum(
            ensemble_pred == 1) else 0
        recall = np.sum((ensemble_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}

    def trainAndEvaluate(self, X_train, y_train, X_val, y_val):
        """
        Trains the ensemble model and evaluates its performance.

        Parameters:
        - X_train, y_train: Training features and labels.
        - X_val, y_val: Validation features and labels.
        """
        # Train Logistic Regression and SVM models
        logit_weights, _, _, logit_train_probs, logit_val_probs = self.logit_model.trainModel(X_train, y_train, X_val,
                                                                                              y_val, learning_rate=0.1,
                                                                                              epochs=3000)
        self.svm_model.fit(X_train, y_train)

        # Get predictions for validation data
        logit_val_pred = (logit_val_probs >= 0.5).astype(int)
        svm_val_decision = np.dot(X_val, self.svm_model.w) - self.svm_model.b
        svm_val_pred = np.sign(svm_val_decision)

        # Hard voting ensembl
        hard_ensemble_pred = self.hardVote(logit_val_pred, svm_val_pred)
        hard_metrics = self.evaluateEnsemble(y_val, hard_ensemble_pred)

        # Soft voting ensemble
        soft_ensemble_pred = self.softVote(logit_val_probs, svm_val_decision)
        soft_metrics = self.evaluateEnsemble(y_val, soft_ensemble_pred)

        # Print metrics
        print("-" * 150)
        print("Hard Voting Ensemble Metrics:")
        for metric, value in hard_metrics.items():
            print(f"    {metric.capitalize()}: {value:.4f}")
        print("-" * 150)
        print("Soft Voting Ensemble Metrics:")
        for metric, value in soft_metrics.items():
            print(f"    {metric.capitalize()}: {value:.4f}")
        print("-" * 150)


if __name__ == "__main__":
    datasetPath = "./PreprocessedDataset.csv"
    targetVariable = "LoanApproved"

    # Instantiate models
    logisticModel = LogisticRegression(datasetPath, targetVariable)
    svmModel = LinearSVM(datasetPath, targetVariable, learning_rate=0.0005, lambda_param=0.01, n_iters=1000)

    # Split data
    X_train, y_train, X_val, y_val = logisticModel.splitData()

    # Initialize and train ensemble model
    ensemble = Ensemble(logisticModel, svmModel)
    ensemble.trainAndEvaluate(X_train, y_train, X_val, y_val)