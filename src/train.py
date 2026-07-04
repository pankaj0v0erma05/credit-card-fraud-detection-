import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'artifacts'))

def train_and_benchmark():
    # -------------------------------------------------------------------------
    # 1. Loading Preprocessed Artifacts
    # -------------------------------------------------------------------------
    print("Loading preprocessed numpy arrays from artifacts...")
    X_train = np.load(os.path.join(ARTIFACTS_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(ARTIFACTS_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(ARTIFACTS_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(ARTIFACTS_DIR, 'y_test.npy'))
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # -------------------------------------------------------------------------
    # 2. Dynamic Imbalance Handling
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE ON DYNAMIC POSITIVE CLASS WEIGHTING:
    # - Boosting algorithms like XGBoost and LightGBM do not have a simple 'balanced' 
    #   class weight switch. Instead, they accept a parameter called 'scale_pos_weight'.
    # - 'scale_pos_weight' acts as a multiplier for the loss incurred on positive (fraud) cases.
    # - We calculate this dynamically as: ratio = count(negative class) / count(positive class).
    # - This scales the gradient updates for the minority class, ensuring that the model 
    #   doesn't ignore fraud transactions in favor of always predicting legitimate transactions.
    
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    imbalance_ratio = neg_count / pos_count
    
    print(f"\n--- Class Imbalance Analysis (Training Set) ---")
    print(f"Legitimate Transactions (Class 0): {neg_count:,}")
    print(f"Fraudulent Transactions (Class 1): {pos_count:,}")
    print(f"Imbalance Ratio (Majority-to-Minority): {imbalance_ratio:.2f}")
    print(f"Suggested scale_pos_weight for Boosting Models: {imbalance_ratio:.2f}")

    # Define the 6 models to benchmark (5 requested + 1 Data Scientist addition: CatBoost)
    models = {
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight='balanced', 
            random_state=42, 
            max_iter=1000
        ),
        "Standard Random Forest (Balanced)": RandomForestClassifier(
            class_weight='balanced', 
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        ),
        "Balanced Random Forest": BalancedRandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            sampling_strategy='all',
            replacement=True
        ),
        "XGBoost (Weighted)": XGBClassifier(
            scale_pos_weight=imbalance_ratio,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        "LightGBM (Weighted)": LGBMClassifier(
            scale_pos_weight=imbalance_ratio,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        # DATA SCIENTIST ADDITION: CatBoost Classifier
        # CatBoost is an advanced gradient boosting algorithm that is highly robust 
        # to class imbalances and performs exceptionally well on tabular datasets.
        "CatBoost (Balanced)": CatBoostClassifier(
            auto_class_weights='Balanced',
            iterations=200,
            depth=6,
            random_seed=42,
            verbose=0
        )
    }

    results = []
    trained_models = {}

    print("\nStarting model training and benchmarking...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on Train & Test
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Compute probabilities for PR-AUC (Average Precision)
        # Handle models that output decision_function instead of predict_proba
        if hasattr(model, "predict_proba"):
            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_train_prob = model.decision_function(X_train)
            y_test_prob = model.decision_function(X_test)

        # -------------------------------------------------------------------------
        # 3. Overfitting Detection Engine & Metric Computations
        # -------------------------------------------------------------------------
        # EDUCATIONAL NOTE ON OVERFITTING & METRICS:
        # - Overfitting occurs when a machine learning model learns the training data too 
        #   well—memorizing noise and random fluctuations rather than the underlying pattern.
        # - MASSIVE RED FLAG IN FRAUD DETECTION: If a model achieves 99% F1-score on the training 
        #   set but drops to 60% on the test set, it has overfitted. In production, this model will 
        #   fail to catch new, unseen fraudulent patterns.
        # - Why evaluate both Precision and Recall?
        #   - Recall (Sensitivity): Out of all true fraud cases, how many did we catch? High recall 
        #     minimizes financial losses from missed fraud.
        #   - Precision: Out of all transactions flagged as fraud, how many were actually fraud? 
        #     High precision minimizes customer frustration (false positives blocking valid cards).
        #   - F1-Score: Harmonic mean of Precision and Recall. It is our single best metric for 
        #     balancing both goals.
        #   - PR-AUC (Average Precision): Evaluates precision/recall trade-offs at all thresholds. 
        #     Unlike ROC-AUC, PR-AUC is highly sensitive to the minority class and is the industry-standard 
        #     validation metric for severe imbalances.

        metrics = {
            "Model": name,
            "Train Precision": precision_score(y_train, y_train_pred),
            "Test Precision": precision_score(y_test, y_test_pred),
            "Train Recall": recall_score(y_train, y_train_pred),
            "Test Recall": recall_score(y_test, y_test_pred),
            "Train F1": f1_score(y_train, y_train_pred),
            "Test F1": f1_score(y_test, y_test_pred),
            "Train PR-AUC": average_precision_score(y_train, y_train_prob),
            "Test PR-AUC": average_precision_score(y_test, y_test_prob)
        }
        results.append(metrics)
        
        # Log quick overview for overfitting audit
        f1_diff = metrics["Train F1"] - metrics["Test F1"]
        overfit_status = "WARNING (Overfitted)" if f1_diff > 0.15 else "PASS (Generalized)"
        print(f"  -> Train F1: {metrics['Train F1']:.4f} | Test F1: {metrics['Test F1']:.4f} | Diff: {f1_diff:.4f} ({overfit_status})")

    # -------------------------------------------------------------------------
    # 4. Print Comparison Summary Table
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(results)
    
    # Sort columns for clean visual comparison: Group Train vs Test side-by-side
    column_order = [
        "Model", 
        "Train F1", "Test F1", 
        "Train Recall", "Test Recall", 
        "Train Precision", "Test Precision",
        "Train PR-AUC", "Test PR-AUC"
    ]
    df_results = df_results[column_order]
    
    print("\n" + "="*95)
    print("                      MODEL BENCHMARKING SUMMARY TABLE")
    print("="*95)
    print(df_results.to_string(index=False, formatters={
        "Train F1": "{:.4f}".format, "Test F1": "{:.4f}".format,
        "Train Recall": "{:.4f}".format, "Test Recall": "{:.4f}".format,
        "Train Precision": "{:.4f}".format, "Test Precision": "{:.4f}".format,
        "Train PR-AUC": "{:.4f}".format, "Test PR-AUC": "{:.4f}".format
    }))
    print("="*95 + "\n")

    # -------------------------------------------------------------------------
    # 5. Save the Best Model
    # -------------------------------------------------------------------------
    # We select the best model based on the highest Test F1-Score (balances Precision/Recall).
    # If there is a tie, we look at Test PR-AUC.
    best_row = df_results.sort_values(by=["Test F1", "Test PR-AUC"], ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model_object = trained_models[best_model_name]
    
    print(f"Best Model Selected: {best_model_name}")
    print(f"  -> Test F1-Score: {best_row['Test F1']:.4f}")
    print(f"  -> Test Recall:   {best_row['Test Recall']:.4f}")
    print(f"  -> Test PR-AUC:   {best_row['Test PR-AUC']:.4f}")
    
    best_model_path = os.path.join(ARTIFACTS_DIR, 'fraud_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model_object, f)
        
    print(f"Successfully saved and serialized best model to: {best_model_path}")
    print("Benchmarking framework run completed!")

if __name__ == "__main__":
    train_and_benchmark()
