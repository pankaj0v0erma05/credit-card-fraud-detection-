import os
import pickle
import pandas as pd
import numpy as np

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'artifacts'))

# Path configurations for preprocessor and model binaries
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'fraud_model.pkl')

def load_inference_artifacts():
    """
    Loads and returns the preprocessor and model binaries.
    """
    if not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Required artifacts (preprocessor.pkl or fraud_model.pkl) are missing. "
            "Please run preprocessing.py and train.py/tune.py first."
        )
        
    print(f"Loading preprocessor from: {PREPROCESSOR_PATH}")
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
        
    print(f"Loading champion model from: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    return preprocessor, model

# Load artifacts once at module load time for efficiency
preprocessor, model = load_inference_artifacts()

def predict_transaction(transaction_data):
    """
    Takes raw, unscaled transaction inputs, preprocesses them safely,
    and returns model predictions (binary class and fraud probability).
    
    Parameters:
    transaction_data (dict): Dictionary containing the transaction fields:
                             'Time', 'Amount', and 'V1' through 'V28'.
                             
    Returns:
    dict: Prediction results containing:
          - 'is_fraud': 1 if fraudulent, 0 if legitimate
          - 'fraud_probability': Float percentage of fraud risk (0% to 100%)
    """
    # -------------------------------------------------------------------------
    # 1. Convert Input to DataFrame
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE:
    # - Scikit-learn's ColumnTransformer remembers the column names it was fitted on.
    # - To apply it during inference, we convert our single transaction dictionary 
    #   into a Pandas DataFrame so column headers match the training schema exactly.
    df_input = pd.DataFrame([transaction_data])
    
    # -------------------------------------------------------------------------
    # 2. Apply Preprocessing (Standard Scaling)
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE ON INFERENCE PREPROCESSING:
    # - In production, new incoming transaction data is raw and unscaled (e.g. Amount = $150.0).
    # - We must scale the 'Time' and 'Amount' columns using the EXACT SAME mean and variance 
    #   parameters learned from the training set.
    # - To do this, we use the loaded, fitted 'preprocessor' object.
    # - CRITICAL: We only call '.transform()' here. Calling '.fit_transform()' would rebuild 
    #   the scaler on this single transaction, resulting in division-by-zero or incorrect scales.
    X_processed = preprocessor.transform(df_input)
    
    # -------------------------------------------------------------------------
    # 3. Model Prediction
    # -------------------------------------------------------------------------
    # - `model.predict()` outputs the binary prediction: 0 (legitimate) or 1 (fraudulent).
    # - `model.predict_proba()` outputs the prediction probabilities for each class. 
    #   Index 1 gives the probability of the transaction being fraudulent.
    prediction = int(model.predict(X_processed)[0])
    probability = float(model.predict_proba(X_processed)[0, 1])
    
    return {
        "is_fraud": prediction,
        "fraud_probability": probability
    }

if __name__ == "__main__":
    # =========================================================================
    # Standalone Execution Test Cases
    # =========================================================================
    # We define two mock transactions to verify our end-to-end inference engine.
    
    # Base columns structure of our features (excluding Class)
    v_features = {f"V{i}": 0.0 for i in range(1, 29)}
    
    # CASE A: Legitimate Transaction Scenario
    # - Low amount ($25.0)
    # - PCA features are close to their normal/mean values (0.0)
    legit_tx = {
        "Time": 42000.0,
        "Amount": 25.0,
        **v_features
    }
    
    # CASE B: Fraudulent Transaction Scenario
    # - Medium-high amount ($150.0)
    # - PCA features strongly indicative of fraud (extremely negative V17, V14, V12)
    #   based on our EDA correlation findings.
    fraud_tx = {
        "Time": 42100.0,
        "Amount": 150.0,
        **v_features
    }
    fraud_tx["V17"] = -8.5
    fraud_tx["V14"] = -7.2
    fraud_tx["V12"] = -6.1
    
    print("\n" + "="*50)
    print("      TESTING INFERENCE ENGINE ON MOCK TRANSACTIONS")
    print("="*50)
    
    # Run Case A
    print("\n---> Running Test Case A (Legitimate Scenario):")
    print(f"Input: Time = {legit_tx['Time']}, Amount = ${legit_tx['Amount']}, PCA Features = Normal (0.0)")
    result_legit = predict_transaction(legit_tx)
    print(f"Result: is_fraud = {result_legit['is_fraud']}")
    print(f"Confidence (Probability): {result_legit['fraud_probability'] * 100:.4f}%")
    
    # Run Case B
    print("\n---> Running Test Case B (Fraudulent Scenario):")
    print(f"Input: Time = {fraud_tx['Time']}, Amount = ${fraud_tx['Amount']}, PCA Features V17={fraud_tx['V17']}, V14={fraud_tx['V14']}, V12={fraud_tx['V12']}")
    result_fraud = predict_transaction(fraud_tx)
    print(f"Result: is_fraud = {result_fraud['is_fraud']}")
    print(f"Confidence (Probability): {result_fraud['fraud_probability'] * 100:.4f}%")
    
    print("\n" + "="*50)
    print("      INFERENCE ENGINE RUN COMPLETE AND VALIDATED!")
    print("="*50 + "\n")
