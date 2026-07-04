import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from xgboost import XGBClassifier

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'artifacts'))

def run_tuning():
    # -------------------------------------------------------------------------
    # 1. Load Preprocessed Data
    # -------------------------------------------------------------------------
    print("Loading preprocessed numpy arrays from artifacts...")
    X_train = np.load(os.path.join(ARTIFACTS_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(ARTIFACTS_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(ARTIFACTS_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(ARTIFACTS_DIR, 'y_test.npy'))
    
    # Calculate scale_pos_weight dynamically based on training labels
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    imbalance_ratio = neg_count / pos_count
    
    print(f"Loaded train shape: {X_train.shape}, test shape: {X_test.shape}")
    print(f"Majority-to-minority ratio (scale_pos_weight): {imbalance_ratio:.2f}")

    # -------------------------------------------------------------------------
    # 2. Hyperparameter Grid Setup & Overfitting Explanation
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE ON OVERFITTING CONTROL VIA TREE DEPTH:
    # - In decision tree-based algorithms like XGBoost, 'max_depth' controls how deep 
    #   individual trees can grow.
    # - Deep trees (e.g., max_depth >= 6) have a high capacity to split data down to 
    #   very small subsets, which leads to memorizing individual training transactions (overfitting).
    # - Shorter trees (e.g., max_depth 3, 4, or 5) have high "bias" but low "variance". 
    #   They are forced to look only at major, highly generalized signals (such as "is Amount large 
    #   AND V17 very low?"), rather than fine-tuning to specific outlier noise.
    # - Restricting 'max_depth' is one of the most effective parameters to force 
    #   XGBoost to generalize better and bridge the Train-Test F1-score gap.
    # - 'learning_rate' (shrinkage) scales the step size of boosting updates. Slower rates 
    #   (0.01 or 0.05) require more estimators but prevent the model from overshooting the optimal pattern.
    
    # Define our hyperparameter search grid
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 150]
    }
    
    # Instantiate the base XGBoost model with dynamic pos weight weighting
    xgb_base = XGBClassifier(
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )

    # -------------------------------------------------------------------------
    # 3. GridSearchCV Execution
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE ON SCORING METRICS:
    # - We use 'average_precision' (PR-AUC) as our cross-validation scoring target.
    # - Since fraudulent transactions constitute only 0.17% of the dataset, standard accuracy 
    #   or ROC-AUC will score close to 99.9% even if the model fails to predict a single fraud case.
    # - PR-AUC directly measures precision/recall curves, forcing cross-validation to select 
    #   parameters that maximize fraud detection precision and recall.
    
    print("\nInitializing GridSearchCV with 3-Fold Stratified Cross-Validation...")
    print(f"Parameter grid to search: {param_grid}")
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='average_precision',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    print("Executing hyperparameter search (this may take a moment)...")
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and estimator
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print("\n" + "="*50)
    print(f"Optimal Hyperparameters Identified:")
    for param, val in best_params.items():
        print(f"  -> {param}: {val}")
    print("="*50 + "\n")

    # -------------------------------------------------------------------------
    # 4. Overfitting Verification & Evaluation
    # -------------------------------------------------------------------------
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)
    
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    train_prauc = average_precision_score(y_train, y_train_prob)
    test_prauc = average_precision_score(y_test, y_test_prob)
    
    # Print direct comparison to check for overfitting
    f1_diff = train_f1 - test_f1
    overfit_status = "WARNING (Still Overfitted)" if f1_diff > 0.15 else "PASS (Successfully Generalized)"
    
    print("="*70)
    print("                   FINAL OVERFITTING VERIFICATION AUDIT")
    print("="*70)
    print(f"Metric          | Train Set | Test Set  | Difference")
    print(f"----------------|-----------|-----------|-----------")
    print(f"F1-Score        |   {train_f1:.4f}  |   {test_f1:.4f}  |   {f1_diff:+.4f} ({overfit_status})")
    print(f"Recall          |   {train_recall:.4f}  |   {test_recall:.4f}  |   {(train_recall - test_recall):+.4f}")
    print(f"Precision       |   {train_precision:.4f}  |   {test_precision:.4f}  |   {(train_precision - test_precision):+.4f}")
    print(f"PR-AUC (AP)     |   {train_prauc:.4f}  |   {test_prauc:.4f}  |   {(train_prauc - test_prauc):+.4f}")
    print("="*70 + "\n")

    # -------------------------------------------------------------------------
    # 5. Overwrite/Serialize Optimized Model
    # -------------------------------------------------------------------------
    best_model_path = os.path.join(ARTIFACTS_DIR, 'fraud_model.pkl')
    print(f"Overwriting serialized model binary at: {best_model_path}")
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
        
    print("Hyperparameter tuning execution and model update complete!")

if __name__ == "__main__":
    run_tuning()
