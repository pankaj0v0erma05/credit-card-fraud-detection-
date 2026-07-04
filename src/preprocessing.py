import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
ARTIFACTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'artifacts'))

def run_preprocessing():
    # -------------------------------------------------------------------------
    # 1. Data Loading & Understanding PCA Features
    # -------------------------------------------------------------------------
    # The primary dataset is 'raw_transactions_ulb.csv' (ULB European Bank PCA data).
    #
    # EDUCATIONAL NOTE ON PCA FEATURES (V1-V28):
    # - In financial fraud detection, sharing raw transaction features (like card numbers, 
    #   merchant IDs, location, or names) violates user privacy and financial compliance rules.
    # - To protect privacy, the original features were transformed using Principal Component 
    #   Analysis (PCA), which is a dimensionality reduction technique.
    # - PCA projects the original features onto orthogonal directions of maximum variance.
    # - The resulting features V1, V2, ..., V28 are the principal components. They are 
    #   numerical, uncorrelated, and anonymous.
    # - Only 'Time' and 'Amount' were left in their original, non-transformed format.
    
    csv_path = os.path.join(DATA_DIR, 'raw_transactions_ulb.csv')
    print(f"Loading raw transaction data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Expected dataset not found at {csv_path}. Please run data_ingestion.py first."
        )
        
    df = pd.read_csv(csv_path)
    
    # Separate features and target label
    # 'Class' is our binary target column: 0 represents valid transactions, 1 represents fraud.
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # -------------------------------------------------------------------------
    # 2. Stratified Train-Test Split
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE ON STRATIFIED SPLITS:
    # - Our fraud dataset has an extreme class imbalance: only 492 out of 284,807 transactions 
    #   are fraudulent (~0.17%).
    # - If we perform a standard random train-test split, there is a high probability that the 
    #   rare fraud cases will be unevenly distributed. In the worst-case scenario, the test set 
    #   might contain zero fraud examples, making it impossible to evaluate our model's performance 
    #   on fraud detection.
    # - Specifying 'stratify=y' forces the train_test_split function to maintain the exact same 
    #   class proportion (0.17% fraud, 99.83% non-fraud) in both the training set and the test set.
    # - This ensures that both training and evaluation datasets are representative of the true 
    #   distribution of fraud.
    
    print("Performing stratified train-test split (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Train set shape: {X_train.shape} (Fraud: {y_train.sum()} / {len(y_train)})")
    print(f"Test set shape: {X_test.shape} (Fraud: {y_test.sum()} / {len(y_test)})")
    
    # -------------------------------------------------------------------------
    # 3. ColumnTransformer Configuration
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE ON COLUMNTRANSFORMER:
    # - A ColumnTransformer allows us to apply specific preprocessing steps to subset columns 
    #   of our dataset, while keeping other columns unchanged.
    # - In our case:
    #   1. The 'Time' and 'Amount' columns are in their raw format, with widely different ranges 
    #      (e.g., Amount can range from 0 to thousands, while Time ranges from 0 to 172,792 seconds). 
    #      We need to apply StandardScaler to these columns so that distance-based or gradient-descent 
    #      models don't get biased by the absolute magnitude of these values.
    #   2. The features 'V1' through 'V28' are already PCA-transformed and standard-scaled. We must 
    #      leave them completely untouched ('passthrough') to preserve their pre-scaled variance structure.
    # - ColumnTransformer helps us declare this mapping cleanly in a single production-grade pipeline object.
    
    # Identify features to scale vs. features to leave untouched
    columns_to_scale = ['Time', 'Amount']
    
    # All other columns are V1-V28
    columns_to_pass = [col for col in X.columns if col not in columns_to_scale]
    
    # Setup ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), columns_to_scale),
            ('passthrough', 'passthrough', columns_to_pass)
        ]
    )
    
    # -------------------------------------------------------------------------
    # 4. Data Leakage Prevention (Fitting vs. Transforming)
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE ON DATA LEAKAGE PREVENTION:
    # - Data Leakage occurs when information from outside the training dataset is used to train 
    #   the model. This leads to overly optimistic performance during testing, but poor generalization 
    #   in production.
    # - When scaling features:
    #   - StandardScaler computes the mean and standard deviation of the column to scale it.
    #   - We must compute (fit) these metrics ONLY on the training set: preprocessor.fit_transform(X_train).
    #   - We then apply the EXACT SAME scaling parameters (mean and std dev) to transform the test set: 
    #     preprocessor.transform(X_test).
    #   - Crucially: We DO NOT call '.fit()' or '.fit_transform()' on the test set! If we did, the scaling 
    #     of the test set would be influenced by the test set's own mean and standard deviation, leaking 
    #     global evaluation statistics into the preprocessing phase.
    
    print("Fitting and applying ColumnTransformer on training data (Data Leakage Prevention)...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print("Applying the fitted ColumnTransformer on test data...")
    X_test_processed = preprocessor.transform(X_test)
    
    # -------------------------------------------------------------------------
    # 5. Exporting Split Data & Preprocessor Artifacts
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE ON EXPORTING ARTIFACTS:
    # - In a professional ML pipeline, we decouple preprocessing from training. 
    # - We save the processed arrays (X_train, X_test, etc.) as numpy binaries (.npy) so that the 
    #   training script can load them instantly without re-running the preprocessing steps.
    # - We freeze (serialize) the fitted 'preprocessor' object using pickle. This is critical: 
    #   when deploying our model to production to serve real-time predictions, we must apply the 
    #   exact same preprocessor configuration (with the training set's mean and std dev) to any new, 
    #   incoming transaction data.
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print(f"Ensuring artifacts directory exists at: {ARTIFACTS_DIR}")
    
    # Save the numpy arrays
    np.save(os.path.join(ARTIFACTS_DIR, 'X_train.npy'), X_train_processed)
    np.save(os.path.join(ARTIFACTS_DIR, 'X_test.npy'), X_test_processed)
    np.save(os.path.join(ARTIFACTS_DIR, 'y_train.npy'), y_train.to_numpy())
    np.save(os.path.join(ARTIFACTS_DIR, 'y_test.npy'), y_test.to_numpy())
    print("Saved split numpy arrays (X_train, X_test, y_train, y_test) to artifacts directory.")
    
    # Save the fitted preprocessor object
    preprocessor_pkl_path = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
    with open(preprocessor_pkl_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Successfully froze and saved fitted preprocessor to: {preprocessor_pkl_path}")
    print("Preprocessing pipeline run complete!")

if __name__ == '__main__':
    run_preprocessing()
