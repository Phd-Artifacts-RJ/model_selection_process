import streamlit as st
import pandas as pd
import numpy as np  # Added for benchmark calculations
from typing import List, Dict, Any

# Import the MODELS dictionary and the new run_experiment function
try:
    from models import MODELS, run_experiment, IMBLEARN_AVAILABLE
except ImportError:
    st.error("Could not find 'MODELS' dictionary, 'run_experiment' function, or 'IMBLEARN_AVAILABLE' in models.py. Please ensure they are defined.")
    MODELS = {}
    IMBLEARN_AVAILABLE = False
    def run_experiment(files, target, models, use_smote): # Added use_smote
        return {"error": "models.py not found"}

class ExperimentSetupApp:
    """
    A class to encapsulate the Streamlit experiment setup wizard.
    """
    
    def __init__(self):
        """
        Initialize the app and set the page title.
        """
        st.title("Experiment Setup Wizard")
        
        # Initialize session state if it doesn't exist
        if 'uploaded_files_map' not in st.session_state:
            st.session_state.uploaded_files_map: Dict[str, Any] = {} # Stores the actual UploadedFile objects
        if 'selected_datasets' not in st.session_state:
            st.session_state.selected_datasets: List[str] = [] # Stores just the names
        
        # --- NEW: Session state for Step 1.5 ---
        if 'use_smote' not in st.session_state:
            st.session_state.use_smote: bool = False
            
        if 'target_column' not in st.session_state:
            st.session_state.target_column: str = "target"
        if 'selected_model_groups' not in st.session_state:
            st.session_state.selected_model_groups: List[str] = []
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models: List[str] = []
        if 'results' not in st.session_state:
            st.session_state.results: Dict[str, Any] = {} # To store the final results
        
        if 'benchmark_results_df' not in st.session_state:
            st.session_state.benchmark_results_df = None # Will store a DataFrame
        if 'benchmark_auc_comparison' not in st.session_state:
            # This will store the avg. AUC tables, e.g., {group: DataFrame}
            st.session_state.benchmark_auc_comparison = None 

    # --- NEW: Callback to clear results if preprocessing options change ---
    def _on_preprocessing_change(self):
        """Resets results if preprocessing options change."""
        st.session_state.results = {}
        st.session_state.benchmark_results_df = None
        st.session_state.benchmark_auc_comparison = None

    def _render_step_1_dataset_selection(self):
        """
        Renders the UI for dataset selection (Step 1).
        """
        st.header("Step 1: Select Datasets")
        
        with st.expander("Upload datasets in experiment", expanded=True):
            uploads = st.file_uploader(
                "Upload CSV files:", type="csv", accept_multiple_files=True
            )

            if uploads:
                new_file_names = [f.name for f in uploads]
                # Check if files have actually changed before clearing results
                if set(new_file_names) != set(st.session_state.selected_datasets):
                    # Use the callback to clear all results
                    self._on_preprocessing_change()

                # Store the actual file objects in a map
                st.session_state.uploaded_files_map = {f.name: f for f in uploads}
                # Store just the names for display and selection
                st.session_state.selected_datasets = new_file_names
            
            else:
                # Clear state if all files are removed
                if st.session_state.selected_datasets: # Only clear if there *were* files
                    self._on_preprocessing_change()
                st.session_state.uploaded_files_map = {}
                st.session_state.selected_datasets = []


    def _display_step_1_results(self):
        """
        Displays the results from Step 1 based on session state.
        """
        if st.session_state.selected_datasets:
            st.success(f"Datasets selected: {', '.join(st.session_state.selected_datasets)}")
        else:
            st.info("No datasets selected.")

    # --- NEW: Step 1.5 for Preprocessing Options ---
    def _render_step_1_5_preprocessing_options(self):
        """
        Renders the UI for preprocessing options (Step 1.5).
        """
        st.header("Step 1.5: Preprocessing Options")
        
        # Disable checkbox if imblearn is not installed
        smote_disabled = not IMBLEARN_AVAILABLE
        
        st.session_state.use_smote = st.checkbox(
            "Apply SMOTE (Synthetic Minority Over-sampling TEchnique)",
            value=st.session_state.use_smote,
            on_change=self._on_preprocessing_change,
            disabled=smote_disabled,
            help="If checked, SMOTE will be applied to the *training data* to handle class imbalance before model fitting. Requires 'imbalanced-learn' to be installed."
        )
        
        if smote_disabled:
            st.warning("SMOTE is disabled because the 'imbalanced-learn' library was not found. Please install it to enable this feature.")

    # --- NEW: Display for Step 1.5 ---
    def _display_step_1_5_results(self):
        """
        Displays the results from Step 1.5 based on session state.
        """
        if st.session_state.use_smote:
            st.info("SMOTE (Oversampling) is **Enabled**.")
        else:
            st.info("SMOTE (Oversampling) is **Disabled**.")

            
    def _render_step_2_model_selection(self):
        """
        Renders the UI for model and target selection (Step 2).
        """
        st.header("Step 2: Select Models & Target")
        
        available_model_groups = list(MODELS.keys())
        
        # Use the same callback for all settings changes
        def on_step_2_change():
            self._on_preprocessing_change() # Use the main reset callback

        st.session_state.target_column = st.text_input(
            "Target Column Name:",
            value=st.session_state.target_column,
            on_change=on_step_2_change
        )
        
        st.session_state.selected_model_groups = st.multiselect(
            "Select model groups to run:",
            options=available_model_groups,
            default=st.session_state.selected_model_groups,
            on_change=on_step_2_change
        )
        
        # Flatten the selected groups into a list of individual models for display
        flat_model_list = []
        for group_name in st.session_state.selected_model_groups:
            if group_name in MODELS:
                flat_model_list.extend(list(MODELS[group_name].keys()))
        
        st.session_state.selected_models = flat_model_list

    def _display_step_2_results(self):
        """
        Displays the results from Step 2 based on session state.
        """
        if not st.session_state.target_column:
            st.warning("Please enter a target column name.")
        else:
            st.success(f"Target column: '{st.session_state.target_column}'")
        
        if st.session_state.selected_models:
            st.success(f"Models to run: {', '.join(st.session_state.selected_models)}")
        else:
            st.info("No models selected.")

    def _render_step_3_run_experiment(self):
        """
        Renders the "Run Experiment" button ONLY if results do not exist.
        The button's logic is contained here.
        """
        st.header("Step 3: Run Experiment")
        
        # Only show the button if the experiment hasn't been run yet
        if not st.session_state.results:
            if st.button("Run Experiment"):
                # Clear any old benchmark results
                st.session_state.benchmark_results_df = None
                st.session_state.benchmark_auc_comparison = None
                with st.spinner("Running models on all datasets... This may take a moment."):
                    try:
                        # Get the list of actual UploadedFile objects to run on
                        files_to_run = [
                            st.session_state.uploaded_files_map[name] 
                            for name in st.session_state.selected_datasets
                        ]
                        
                        # Create the dictionary of selected model groups to pass
                        selected_groups_dict = {
                            group: MODELS[group] 
                            for group in st.session_state.selected_model_groups
                            if group in MODELS
                        }
                        
                        # --- MODIFIED: Pass the use_smote state ---
                        st.session_state.results = run_experiment(
                            files_to_run,
                            st.session_state.target_column,
                            selected_groups_dict,
                            st.session_state.use_smote # <-- NEW ARGUMENT
                        )
                        
                        # Check for global errors (e.g., SMOTE library missing)
                        if "error" in st.session_state.results:
                            st.error(st.session_state.results["error"])
                            st.session_state.results = {} # Clear the error
                        else:
                            st.success("Experiment complete!")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"An error occurred during the experiment: {e}")
                        st.session_state.results = {} # Clear partial results on failure
        else:
            # Results exist, so don't show the "Run" button.
            pass

    def _display_step_3_results(self):
        """
        Displays the results from the experiment run, grouped by dataset and model group.
        """
        if not st.session_state.results:
            return
            
        st.subheader("Experiment Results")

        for dataset_name, dataset_results in st.session_state.results.items():
            st.markdown(f"### Results for: `{dataset_name}`")
            
            if "error" in dataset_results:
                st.error(f"Error processing this dataset: {dataset_results['error']}")
                continue

            if not dataset_results:
                st.warning("No results were generated for this dataset.")
                continue

            for group_name, group_results in dataset_results.items():
                st.markdown(f"#### Model Group: {group_name}")
                
                try:
                    df = pd.DataFrame.from_dict(group_results, orient="index")
                    
                    if "error" in df.columns and len(df.columns) == 1:
                        st.dataframe(df) # Show the error DataFrame
                    else:
                        st.dataframe(
                            df.style.format(
                                {
                                    "AUC": "{:.4f}",
                                    "PCC": "{:.4f}",
                                    "BS": "{:.4f}",
                                    "KS": "{:.4f}",
                                    "PG": "{:.4f}",
                                    "H": "{:.4f}",
                                },
                                na_rep="Error" 
                            )
                        )
                except Exception as e:
                    st.error(f"Could not display results for {group_name}: {e}")
                    st.json(group_results) 

    def _calculate_benchmarks(self):
        """
        Calculates benchmark models and average AUC comparison tables.
        Populates session state with two DataFrames.
        """
        results = st.session_state.results
        if not results:
            st.warning("No results found. Please run Step 3 first.")
            return

        # 1. Aggregate all AUC scores for each model
        model_scores: Dict[str, Dict[str, List[float]]] = {} # {group: {model: [auc1, auc2, ...]}}
        
        for dataset_name, dataset_results in results.items():
            if "error" in dataset_results:
                continue
            
            for group_name, group_results in dataset_results.items():
                if group_name not in model_scores:
                    model_scores[group_name] = {}
                
                for model_name, metrics in group_results.items():
                    if model_name not in model_scores[group_name]:
                        model_scores[group_name][model_name] = []
                    
                    if "AUC" in metrics and pd.notna(metrics["AUC"]): # Check for errors/NaN
                        model_scores[group_name][model_name].append(metrics["AUC"])

        # 2. Find best model AND build average AUC comparison tables
        benchmark_models: Dict[str, str] = {} # {group: 'best_model_name'}
        auc_comparison_tables: Dict[str, pd.DataFrame] = {} 
        
        for group_name, models in model_scores.items():
            avg_aucs = {}
            for model_name, auc_list in models.items():
                if auc_list: # Only consider models that ran successfully
                    avg_aucs[model_name] = np.mean(auc_list)
            
            if avg_aucs: 
                avg_auc_df = pd.DataFrame.from_dict(avg_aucs, orient='index', columns=['Average AUC'])
                avg_auc_df = avg_auc_df.sort_values(by='Average AUC', ascending=False)
                auc_comparison_tables[group_name] = avg_auc_df

                best_model = max(avg_aucs, key=avg_aucs.get)
                benchmark_models[group_name] = best_model
        
        st.session_state.benchmark_auc_comparison = auc_comparison_tables

        if not benchmark_models:
            st.error("Could not determine benchmark models. No successful runs found.")
            return

        # 3. Build the final benchmark summary table data
        final_table_data = []
        
        for dataset_name, dataset_results in results.items():
            if "error" in dataset_results:
                continue
            
            for group_name, best_model_name in benchmark_models.items():
                if group_name in dataset_results and best_model_name in dataset_results[group_name]:
                    metrics = dataset_results[group_name][best_model_name]
                    
                    if "error" not in metrics:
                        row = {
                            'Dataset': dataset_name,
                            'Model Group': group_name,
                            'Benchmark Model': best_model_name,
                            **metrics 
                        }
                        final_table_data.append(row)
        
        if not final_table_data:
            st.error("Failed to build benchmark table. No valid metrics found.")
            return
            
        # 4. Create and store the final summary DataFrame
        df = pd.DataFrame(final_table_data)
        all_cols = ['Dataset', 'Model Group', 'Benchmark Model', 'AUC', 'PCC', 'BS', 'KS', 'PG', 'H']
        final_cols = [col for col in all_cols if col in df.columns]
        st.session_state.benchmark_results_df = df[final_cols]

    def _render_step_4_benchmark_analysis(self):
        """
        Renders the button to trigger benchmark analysis.
        """
        st.header("Step 4: Benchmark Analysis")
        
        def on_benchmark_click():
            # Clear previous benchmark results before recalculating
            st.session_state.benchmark_results_df = None
            st.session_state.benchmark_auc_comparison = None 

        if st.button("Find Benchmark Models"): # Removed on_click, logic is simple
            with st.spinner("Calculating benchmark models..."):
                self._calculate_benchmarks()
                if st.session_state.benchmark_results_df is not None:
                    st.success("Benchmark analysis complete!")
                # Errors are handled inside _calculate_benchmarks

    def _display_step_4_results(self):
        """
        Displays the avg. AUC comparison tables and the final benchmark results.
        """
        
        if st.session_state.benchmark_auc_comparison:
            st.subheader("Average AUC Comparison (by Group)")
            st.markdown("This table shows the average AUC for all models across all datasets, which is used to select the benchmark model for each group.")
            
            for group_name, auc_df in st.session_state.benchmark_auc_comparison.items():
                st.markdown(f"#### Group: {group_name}")
                st.dataframe(auc_df.style.format("{:.4f}"))
            
            st.markdown("---") 
        
        if st.session_state.benchmark_results_df is not None:
            st.subheader("Benchmark Model Summary")
            st.markdown("This table shows the full performance metrics for *only* the best model from each group on each dataset.")
            
            df = st.session_state.benchmark_results_df
            st.dataframe(
                df.style.format(
                    {
                        "AUC": "{:.4f}",
                        "PCC": "{:.4f}",
                        "BS": "{:.4f}",
                        "KS": "{:.4f}",
                        "PG": "{:.4f}",
                        "H": "{:.4f}",
                    },
                    na_rep="N/A"
                )
            )
        elif not st.session_state.benchmark_auc_comparison:
            st.info("Run benchmark analysis to see the final summary table here.")

    # --- MODIFIED: Main run() method logic ---
    def run(self):
        """
        Run the main application logic and render the UI.
        """
        
        # --- Step 1: Datasets ---
        self._render_step_1_dataset_selection()
        self._display_step_1_results()
        
        st.markdown("---")

        # --- Steps 1.5, 2, 3, 4 (Conditional) ---
        if st.session_state.selected_datasets:
            
            # --- NEW: Render Step 1.5 ---
            self._render_step_1_5_preprocessing_options()
            self._display_step_1_5_results()
            st.markdown("---")
            # --- END NEW STEP ---

            self._render_step_2_model_selection()
            self._display_step_2_results()
            st.markdown("---")
            
            # --- Step 3: Run & View Results (Conditional) ---
            if st.session_state.target_column and st.session_state.selected_models:
                
                self._render_step_3_run_experiment()
                self._display_step_3_results()
                
                # --- Step 4 (Conditional) ---
                if st.session_state.results: # Only show if Step 3 has run
                    st.markdown("---")
                    self._render_step_4_benchmark_analysis()
                    self._display_step_4_results()
                    
            else:
                st.info("Complete Step 2 (select target and models) to run the experiment.")
                
        else:
            st.info("Complete Step 1 (upload datasets) to proceed.")


def main() -> None:
    """
    Main function to instantiate and run the Streamlit app.
    """
    app = ExperimentSetupApp()
    app.run()

if __name__ == "__main__":
    main()