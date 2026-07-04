import os
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'artifacts'))

def evaluate_model():
    """
    Loads test artifacts and the tuned model, then prints a complete 
    classification evaluation report to the screen.
    """
    print("Loading test data and champion model...")
    X_test = np.load(os.path.join(ARTIFACTS_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(ARTIFACTS_DIR, 'y_test.npy'))
    
    model_path = os.path.join(ARTIFACTS_DIR, 'fraud_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Champion model not found at {model_path}. Please train a model first.")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # Get model predictions and probability scores
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate confusion matrix & precision-recall metrics
    cm = confusion_matrix(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_prob)
    
    # Print a beautiful evaluation metrics printout
    print("\n" + "="*50)
    print("         FINAL MODEL EVALUATION REPORT")
    print("="*50)
    print("\n--- Confusion Matrix ---")
    print(f"True Negatives (Legitimate Approved):   {cm[0][0]:,}")
    print(f"False Positives (Legitimate Flagged):  {cm[0][1]:,}")
    print(f"False Negatives (Fraudulent Missed):    {cm[1][0]:,}")
    print(f"True Positives (Fraudulent Caught):     {cm[1][1]:,}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"]))
    
    print(f"Precision-Recall AUC (PR-AUC): {pr_auc:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_model()
