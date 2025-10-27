# shap_analysis.py
import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.pipeline import Pipeline
import numpy as np # Added numpy

# Set SHAP's visualize_feature to True to use matplotlib
shap.initjs()

def get_shap_values(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Calculates GLOBAL SHAP values for multiple instances (the test set).
    This function is NOT cached to avoid using old, broken results.
    
    Returns:
        shap_values (np.array): The 1D array of SHAP values.
        explain_data_sample_df (pd.DataFrame): The data sample used for explanation.
    """
    
    feature_names = X_train.columns.tolist()
    
    # 1. Sample the data
    background_data_df = shap.sample(X_train, min(100, X_train.shape[0]), random_state=42)
    explain_data_df = shap.sample(X_test, min(500, X_test.shape[0]), random_state=42)
    
    # 2. Wrapper function that returns ONLY P(class 1)
    def predict_proba_p1(data):
        if not isinstance(data, pd.DataFrame):
            data_df = pd.DataFrame(data, columns=feature_names)
        else:
            data_df = data
        return model.predict_proba(data_df)[:, 1]

    # 3. Create Explainer
    explainer = shap.KernelExplainer(predict_proba_p1, background_data_df)
    
    # 4. Calculate SHAP values
    shap_values = explainer.shap_values(explain_data_df)
    
    # 5. Return the single SHAP array and the DataFrame used for explanation
    return shap_values, explain_data_df


# --- NEW FUNCTION FOR STEP 6 ---
def get_local_shap_explanation(model: Pipeline, X_train: pd.DataFrame, instance_df: pd.DataFrame):
    """
    Calculates LOCAL SHAP values for a SINGLE instance.
    
    Returns:
        shap.Explanation: A SHAP Explanation object for the single instance.
    """
    
    feature_names = X_train.columns.tolist()
    
    # 1. Background data for the explainer
    #    We use a smaller sample for local explanation as it can be faster
    background_data_df = shap.sample(X_train, min(50, X_train.shape[0]), random_state=42)
    
    # 2. Wrapper function that returns ONLY P(class 1)
    def predict_proba_p1(data):
        if not isinstance(data, pd.DataFrame):
            data_df = pd.DataFrame(data, columns=feature_names)
        else:
            data_df = data
        return model.predict_proba(data_df)[:, 1]

    # 3. Create Explainer
    explainer = shap.KernelExplainer(predict_proba_p1, background_data_df)
    
    # 4. Calculate SHAP values for the single instance
    #    This returns a 1D array
    shap_values_instance = explainer.shap_values(instance_df)[0] 
    
    # 5. Get the explainer's base value
    expected_value = explainer.expected_value
    
    # 6. Create and return a SHAP Explanation object
    #    This bundles values, base_value, data, and feature names
    explanation_object = shap.Explanation(
        values=shap_values_instance,
        base_values=expected_value,
        data=instance_df.iloc[0], # Pass the 1D Series of data
        feature_names=feature_names
    )
    
    return explanation_object
