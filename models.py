import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from io import StringIO

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# --- NEW IMPORTS FOR SMOTE ---
# We use a try/except block in case imbalanced-learn is not installed
try:
    from imblearn.over_sampling import SMOTE
    # We must use the imblearn pipeline to correctly apply SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    # Create a placeholder class if imblearn is not available
    # This prevents the app from crashing if SMOTE is selected
    class SMOTE:
        def __init__(self, **kwargs):
            pass
    class ImbPipeline:
        def __init__(self, steps):
            pass
    print("Warning: imbalanced-learn not found. SMOTE functionality will be disabled.")


# Import the metrics functions from your metrics.py file
try:
    from metrics import compute_metrics
except ImportError:
    print("Warning: metrics.py not found. Using placeholder metrics.")
    # Define a placeholder if metrics.py is missing, to avoid crashing
    def compute_metrics(y_tr, y_te, p_te, **kwargs):
        return {"AUC": 0.5, "PCC": 0.5, "BS": 0.25, "KS": 0.0, "PG": 0.0, "H": 0.0}

def _preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
        ]
    )

# ---------- Individual Classifiers ----------

# ---------- Logistic Regression ----------

def build_lr_lbfgs():
    base = LogisticRegression(
        penalty="l2",
        C=1e6,
        solver="lbfgs", # Corrected from 'lfgs'
        max_iter=1000,
        random_state=42,
        n_jobs=None,     # supports parallelization via joblib env vars
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_lr_saga():
    base = LogisticRegression(
        penalty="l2",
        C=1e6,
        solver="saga",
        max_iter=2000,   # saga often needs more iterations
        random_state=42,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_lr_newton_cg():
    base = LogisticRegression(
        penalty="l2",
        C=1e6,
        solver="newton-cg",
        max_iter=1000,
        random_state=42,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

# ---------- Regularized Logistic Regression ----------

def build_lr_reg_saga():
    """Regularised LogisticRegression using saga (supports elasticnet)."""
    base = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        C=1.0,
        solver="saga",
        max_iter=2000,
        random_state=42,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_lr_reg_lbfgs():
    """Regularised LogisticRegression using lbfgs (L2 penalty)."""
    base = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        n_jobs=None,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_lr_reg_liblinear():
    """Regularised LogisticRegression using liblinear (supports L1/L2, ovR)."""
    base = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))


# --- NEW: AdaBoost (Alternating Decision Tree) Group ---

def build_adaboost(n_estimators):
    """
    Helper function to build an AdaBoost classifier with a Decision Tree stump.
    This is scikit-learn's equivalent of an alternating decision tree.
    """
    base = AdaBoostClassifier(
        # Use a "stump" (max_depth=1) as the weak learner
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_estimators,
        random_state=42,
        algorithm='SAMME.R' # Default, supports predict_proba
    )
    # We calibrate it for consistency with other models
    return make_pipeline(_preprocessor(), CalibratedClassifierCV(base, method="sigmoid", cv=3))

def build_adaboost_10():
    return build_adaboost(n_estimators=10)

def build_adaboost_20():
    return build_adaboost(n_estimators=20)

def build_adaboost_30():
    return build_adaboost(n_estimators=30)


# Dictionary of model groups for the Streamlit app
MODELS = {
    "lr": {
        "lr_lbfgs": build_lr_lbfgs,
        "lr_saga": build_lr_saga,
        "lr_newton_cg": build_lr_newton_cg,
    },
    "lr_reg": {
        "lr_reg_saga": build_lr_reg_saga,
        "lr_reg_lbfgs": build_lr_reg_lbfgs,
        "lr_reg_liblinear": build_lr_reg_liblinear,
    },
    # --- NEWLY ADDED GROUP ---
    "adaboost": {
        "adaboost_10": build_adaboost_10,
        "adaboost_20": build_adaboost_20,
        "adaboost_30": build_adaboost_30,
    },
}


# --- MODIFIED EXECUTION FUNCTION ---

def run_experiment(uploaded_files, target_column, selected_model_groups_dict, use_smote):
    """
    Runs the full experiment on all uploaded datasets and selected models.

    Args:
        uploaded_files (list): List of Streamlit UploadedFile objects.
        target_column (str): The name of the target variable.
        selected_model_groups_dict (dict): A dictionary of the selected model groups.
        use_smote (bool): Whether to apply SMOTE to the training data.
    """
    results = {}

    if use_smote and not IMBLEARN_AVAILABLE:
        return {"error": "SMOTE was selected, but the 'imbalanced-learn' library is not installed."}

    for file in uploaded_files:
        dataset_name = file.name
        results[dataset_name] = {}
        
        try:
            # Read the uploaded file into a pandas DataFrame
            # We use StringIO to read the file buffer
            stringio = StringIO(file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)

            # Check if target column exists
            if target_column not in df.columns:
                results[dataset_name] = {"error": f"Target column '{target_column}' not found."}
                continue

            # --- Data Preparation ---
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Simple train-test split (you could replace this with cross-validation)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # --- Model Training and Evaluation ---
            for group_name, models in selected_model_groups_dict.items():
                results[dataset_name][group_name] = {}
                
                for model_name, model_builder in models.items():
                    try:
                        # Build the original pipeline (preprocessor + classifier)
                        original_pipeline = model_builder()
                        
                        # --- NEW SMOTE LOGIC ---
                        if use_smote:
                            # If SMOTE is selected, we rebuild the pipeline
                            # using imblearn's Pipeline to inject SMOTE
                            
                            # Extract the steps from the original pipeline
                            original_steps = original_pipeline.steps
                            preprocessor = original_steps[0] # ('columntransformer', ...)
                            classifier = original_steps[1]   # ('calibratedclassifiercv', ...)

                            # Create a new pipeline with SMOTE
                            pipeline = ImbPipeline([
                                preprocessor,
                                ('smote', SMOTE(random_state=42)),
                                classifier
                            ])
                        else:
                            # If SMOTE is not selected, use the original pipeline
                            pipeline = original_pipeline
                        # --- END NEW SMOTE LOGIC ---

                        # Train the model (this will now use SMOTE if enabled)
                        pipeline.fit(X_train, y_train)
                        
                        # Get predicted probabilities for the positive class
                        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                        
                        # Compute all metrics
                        model_metrics = compute_metrics(y_train, y_test, y_pred_proba)
                        
                        # Store the metrics
                        results[dataset_name][group_name][model_name] = model_metrics

                    except Exception as e:
                        # Store any error that occurs during model training/prediction
                        results[dataset_name][group_name][model_name] = {"error": str(e)}

        except Exception as e:
            # Store any error that occurs during file reading/processing
            results[dataset_name] = {"error": f"Failed to process file: {str(e)}"}

    return results


