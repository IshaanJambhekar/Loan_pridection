import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class LoanDefaultModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_model(self, X_train, y_train):
        """Train the Random Forest model."""
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained Random Forest model on test data."""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        return results

    def save_model(self, file_path):
        """Save the trained Random Forest model to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Random Forest model saved to {file_path}.")

    def load_model(self, file_path):
        """Load a trained Random Forest model from a file."""
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Random Forest model loaded from {file_path}.")

# Example Usage:
# loan_model = LoanDefaultModel()
# loan_model.train_model(X_train, y_train)
# results = loan_model.evaluate_model(X_test, y_test)
# loan_model.save_model('random_forest_model.pkl')
# loan_model.load_model('random_forest_model.pkl')
