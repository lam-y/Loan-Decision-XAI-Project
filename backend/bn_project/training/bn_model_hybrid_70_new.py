# training/bn_model_hybrid.py
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, ExpertKnowledge

class BNModelBuilder:
    def __init__(self):
        self.model = None
        self.bins = {}       
        self.encoders = {}   
        self.selected_features = []
        
    def build_and_train(self, data_filename, top_features, test_size=0.30, random_state=42):
        print("Starting Hierarchical Expert-Driven Bayesian Network Build Process...")
        self.selected_features = top_features
        
        # 1. Load Data
        BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        full_path = os.path.join(BASE_DIR, "data", data_filename)
        
        print(f"Loading dataset from: {full_path}")
        try:
            data = pd.read_csv(full_path)
        except FileNotFoundError:
            full_path = os.path.join("data", data_filename) if os.path.exists(os.path.join("data", data_filename)) else data_filename
            data = pd.read_csv(full_path)

        data.columns = data.columns.str.strip()
        data['Loan'] = data['loan_approved'].astype(int)
        
        # 2. Train / Test Split (70% Train - 30% Test)
        print(f"Splitting Dataset: {int((1-test_size)*100)}% Training | {int(test_size*100)}% Testing...")
        train_raw, test_raw = train_test_split(
            data, test_size=test_size, random_state=random_state, stratify=data['Loan']
        )
        
        # حفظ عينة الاختبار (30%) لاستخدامها في التقييم الموحد للشبكة البايزية وللنموذج اللغوي
        # Define the project data directory
        DATA_DIR = os.path.join(BASE_DIR, "data")

        # Create the directory if it does not exist
        os.makedirs(DATA_DIR, exist_ok=True)

        # Save the test dataset
        test_csv_path = os.path.join(DATA_DIR, "test_dataset_new_30.csv")
        test_raw.to_csv(test_csv_path, index=False)

        print(f"Test dataset saved to: {test_csv_path}")
        print(f"Exported 30% Test Dataset to: '{test_csv_path}' ({len(test_raw)} samples)")

        # 3. Preprocessing (Fit on Train only)
        print("Applying tailored binning and encoding per feature...")
        df_train = pd.DataFrame()
        df_train['Loan'] = train_raw['Loan']

        for col in top_features:
            if col in ['credit_score', 'annual_income']:
                _, bins = pd.qcut(train_raw[col], q=4, retbins=True, duplicates='drop')
                bins[0] = -np.inf
                bins[-1] = np.inf
                self.bins[col] = bins
                df_train[col] = pd.cut(train_raw[col], bins=self.bins[col], labels=False)
                print(f" {col} -> Split into 4 logical Quantile Bins.")
                
            elif train_raw[col].dtype == 'object' or col == 'employment_status':
                le = LabelEncoder()
                df_train[col] = le.fit_transform(train_raw[col])
                self.encoders[col] = le
                print(f" {col} -> Encoded using LabelEncoder.")
                
            else:
                df_train[col] = train_raw[col].copy()
                print(f" {col} -> Kept as native Discrete Integers.")

        # Export the exact bin boundaries used for preprocessing
        self.export_bins_to_excel(
            filename=os.path.join(
                DATA_DIR,
                "Loan_Binning_Boundaries_New.xlsx"
            )
        )

        # Domain-Expert Intermediate Latent Variables
        print("Injecting Domain-Expert Intermediate Latent Variables...")
        
        def determine_financial_capacity(row):
            if row['employment_status'] in [0, 'No', 'unemployed']:
                return 0 if row['annual_income'] < 1 else 1
            if row['annual_income'] >= 2 and row['num_dependents'] < 2:
                return 2
            elif row['annual_income'] < 1:
                return 0
            else:
                return 1

        def determine_credit_worthiness(row):
            if row['credit_score'] < 1:
                return 0
            if 1 <= row['credit_score'] < 3:
                return 1 if row['existing_loans_count'] <= 1 else 0
            if row['credit_score'] >= 3:
                return 2 if row['existing_loans_count'] == 0 else 1

        df_train['financial_capacity'] = df_train.apply(determine_financial_capacity, axis=1)
        df_train['credit_worthiness'] = df_train.apply(determine_credit_worthiness, axis=1)

        df_train = df_train.dropna().astype(str)

        # 4. Hierarchical Structure Learning
        print("\nConstructing Hierarchical Causal Chain using Hybrid Knowledge...")
        required_edges = [
            ('annual_income', 'financial_capacity'),
            ('employment_status', 'financial_capacity'),
            ('num_dependents', 'financial_capacity'),
            ('credit_score', 'credit_worthiness'),
            ('existing_loans_count', 'credit_worthiness'),
            ('financial_capacity', 'Loan'),
            ('credit_worthiness', 'Loan')
        ]
        
        forbidden_edges = [
            ('Loan', 'financial_capacity'),
            ('Loan', 'credit_worthiness'),
            ('financial_capacity', 'annual_income'),
            ('credit_worthiness', 'credit_score')
        ]
        
        expert_knowledge = ExpertKnowledge(required_edges=required_edges, forbidden_edges=forbidden_edges)
        
        hc = HillClimbSearch(df_train)
        best_model_dag = hc.estimate(scoring_method='k2', expert_knowledge=expert_knowledge)
        
        self.model = DiscreteBayesianNetwork(best_model_dag.edges())

        # 5. Parameter Learning
        print("\nComputing Conditional Probability Tables...")
        estimator = BayesianEstimator(self.model, df_train)
        cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=10)
        
        for cpd in cpds:
            self.model.add_cpds(cpd)
            
        self.model.check_model()
        print("Ultimate Hierarchical Bayesian Network successfully trained and validated on 70% data!")

        important_nodes = ['financial_capacity', 'credit_worthiness', 'Loan']
        self.export_cpts_to_excel(
            filename="Bayesian_Network_CPTs_New.xlsx",
            nodes_to_export=important_nodes
        )

    
    def export_bins_to_excel(self, filename="Loan_Binning_Boundaries_New.xlsx"):
        """Export the exact bin boundaries used by the trained model."""

        rows = []

        for feature, bins in self.bins.items():
            for i in range(len(bins) - 1):
                lower = float(bins[i])
                upper = float(bins[i + 1])

                # Keep exact boundary representations as text,
                # including -inf and inf.
                lower_exact = repr(lower)
                upper_exact = repr(upper)

                rows.append({
                    "Feature": feature,
                    "Bin": i,
                    "Lower_Bound": lower if np.isfinite(lower) else lower_exact,
                    "Upper_Bound": upper if np.isfinite(upper) else upper_exact,
                    "Lower_Bound_Exact": lower_exact,
                    "Upper_Bound_Exact": upper_exact,
                    "Interval": f"({lower_exact}, {upper_exact}]"
                })

        if not rows:
            print("No bin boundaries found to export.")
            return

        df_bins = pd.DataFrame(rows)

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df_bins.to_excel(writer, sheet_name="Bin_Boundaries", index=False)

        print(f"Bin boundaries exported to: {os.path.abspath(filename)}")


    def export_cpts_to_excel(self, filename="Bayesian_Network_CPTs_Full.xlsx", nodes_to_export=None):
        import itertools
        print(f"\nExporting all CPT values to Excel: '{filename}'...")
        nodes = nodes_to_export if nodes_to_export else self.model.nodes()
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_welcome = pd.DataFrame({
                    "Information": ["Bayesian Network Conditional Probability Tables (Full Export)"], 
                    "Status": ["All possible states and combinations are included"]
                })
                df_welcome.to_excel(writer, sheet_name="Overview", index=False)
                
                sheets_counter = 0
                for node in nodes:
                    if node not in self.model.nodes():
                        continue
                    
                    cpd = self.model.get_cpds(node)
                    if cpd is None:
                        continue
                    
                    variables = cpd.variables
                    state_names = cpd.state_names
                    iterator = itertools.product(*[state_names[var] for var in variables])
                    rows = []
                    
                    for combo in iterator:
                        values_dict = {var: combo[i] for i, var in enumerate(variables)}
                        prob = cpd.get_value(**values_dict)
                        row = {var: combo[i] for i, var in enumerate(variables)}
                        row['Probability'] = prob
                        rows.append(row)
                        
                    grid = pd.DataFrame(rows)
                    columns_order = [node] + [v for v in variables if v != node] + ['Probability']
                    grid = grid[columns_order]
                    
                    sheet_name = node[:31] 
                    grid.to_excel(writer, sheet_name=sheet_name, index=False)
                    sheets_counter += 1
                    
            print(f"Successfully exported all {sheets_counter} CPTs completely into '{filename}'!")
        except Exception as e:
            print(f"Failed to write to Excel. Error: {e}")

    def save_model(self, save_path="loan_bn_model_new.pkl"):
        joblib.dump(self, save_path)
        print(f"Model saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    DATASET_FILENAME = "loanapproval.csv"
    MODEL_FILENAME = "loan_bn_model_hybrid.pkl"
    
    top_5_features = ['credit_score', 'annual_income', 'existing_loans_count', 'num_dependents', 'employment_status']
    
    builder = BNModelBuilder()
    builder.build_and_train(DATASET_FILENAME, top_5_features, test_size=0.30)
    builder.save_model(MODEL_FILENAME)