# training/bn_model_hybrid.py
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import itertools
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, ExpertKnowledge

class BNModelBuilder:
    def __init__(self):
        self.model = None
        self.bins = {}       
        self.encoders = {}   
        self.selected_features = []
        
    def build_and_train(self, data_filename, top_features):
        print("Starting Hierarchical Expert-Driven Bayesian Network Build Process...")
        self.selected_features = top_features
        
        # 1. Load Data
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else '.'
        full_path = os.path.join(BASE_DIR, "data", data_filename)
        
        print(f"Loading dataset from: {full_path}")
        try:
            data = pd.read_csv(full_path)
        except FileNotFoundError:
            full_path = os.path.join("data", data_filename) if os.path.exists(os.path.join("data", data_filename)) else data_filename
            data = pd.read_csv(full_path)

        data.columns = data.columns.str.strip()
        data['Loan'] = data['loan_approved'].astype(int)
        
        # 2. Preprocessing
        print("Applying tailored binning and encoding per feature...")
        df_train = pd.DataFrame()
        df_train['Loan'] = data['Loan']

        for col in top_features:
            if col in ['credit_score', 'annual_income']:
                _, bins = pd.qcut(data[col], q=4, retbins=True, duplicates='drop')
                bins[0] = -np.inf
                bins[-1] = np.inf
                self.bins[col] = bins
                df_train[col] = pd.cut(data[col], bins=self.bins[col], labels=False)
                print(f" {col} -> Split into 4 logical Quantile Bins.")
                
            elif data[col].dtype == 'object' or col == 'employment_status':
                le = LabelEncoder()
                df_train[col] = le.fit_transform(data[col])
                self.encoders[col] = le
                print(f" {col} -> Encoded using LabelEncoder.")
                
            else:
                df_train[col] = data[col].copy()
                print(f" {col} -> Kept as native Discrete Integers.")

        # هندسة الميزات الوسيطة
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

        # 3. Hierarchical Structure Learning
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

        # 4. Parameter Learning
        print("\nComputing Conditional Probability Tables...")
        estimator = BayesianEstimator(self.model, df_train)
        cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=10)
        
        for cpd in cpds:
            self.model.add_cpds(cpd)
            
        self.model.check_model()
        print("Ultimate Hierarchical Bayesian Network successfully trained and validated!")

        important_nodes = ['financial_capacity', 'credit_worthiness', 'Loan']
        self.export_cpts_to_excel(nodes_to_export=important_nodes)

    def _decode_value(self, var_name, val):
        try:
            val_int = int(float(val))
        except (ValueError, TypeError):
            return str(val)

        # 1. فك ترميز العقد الاستثنائية والتابعة
        if var_name == 'Loan':
            return "Approved (1)" if val_int == 1 else "Rejected (0)"
        elif var_name in ['financial_capacity', 'credit_worthiness']:
            capacity_map = {0: "Low (0)", 1: "Medium (1)", 2: "High (2)"}
            return capacity_map.get(val_int, str(val))

        # 2. فك ترميز المتغيرات المعالجة بـ LabelEncoder
        if var_name in self.encoders:
            try:
                return str(self.encoders[var_name].inverse_transform([val_int])[0])
            except Exception:
                pass

        # 3. فك ترميز المتغيرات المقسمة بـ Quantile Bins
        if var_name in self.bins:
            b = self.bins[var_name]
            if 0 <= val_int < len(b) - 1:
                low = "-∞" if np.isneginf(b[val_int]) else f"{b[val_int]:.1f}"
                high = "+∞" if np.isposinf(b[val_int+1]) else f"{b[val_int+1]:.1f}"
                return f"Bin {val_int} [{low} to {high}]"

        return str(val)

    def export_cpts_to_excel(self, filename="Bayesian_Network_CPTs_Full_2.xlsx", nodes_to_export=None):
        """تستخرج جداول الـ CPTs بالكامل مع فك الترميز وتصديرها إلى ملف إكسل واضح ومقروء"""
        print(f"\nExporting all CPT values to Excel: '{filename}'...")
        
        nodes = nodes_to_export if nodes_to_export else self.model.nodes()
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # صفحة ملخص واضحة
                df_welcome = pd.DataFrame({
                    "Information": ["Bayesian Network Conditional Probability Tables (Decoded & Formatted Export)"], 
                    "Status": ["All possible states are included with clear human-readable labels"]
                })
                df_welcome.to_excel(writer, sheet_name="Overview", index=False)
                
                sheets_counter = 0
                for node in nodes:
                    if node not in self.model.nodes():
                        print(f"  Warning: Node '{node}' not found in the model. Skipping...")
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
                        
                        # إنشاء الصف وفك ترميز القيم للحصول على قراءة بشرية واضحة
                        row = {}
                        for i, var in enumerate(variables):
                            raw_val = combo[i]
                            row[var] = self._decode_value(var, raw_val)
                            
                        row['Probability'] = round(float(prob), 4)
                        rows.append(row)
                        
                    grid = pd.DataFrame(rows)
                    
                    # ترتيب الأعمدة: المتغير التابع ثم الآباء ثم الاحتمالية
                    columns_order = [node] + [v for v in variables if v != node] + ['Probability']
                    grid = grid[columns_order]
                    
                    print(f"\n--- Decoded CPT Generated for Node: {node} (Total Rows: {len(grid)}) ---")
                    
                    sheet_name = node[:31] 
                    grid.to_excel(writer, sheet_name=sheet_name, index=False)
                    sheets_counter += 1
                    
            print(f"Successfully exported all {sheets_counter} CPTs completely into '{filename}'!")
        except Exception as e:
            print(f"Failed to write to Excel. Error: {e}")

    def save_model(self, save_path="loan_bn_model_hybrid.pkl"):
        joblib.dump(self, save_path)
        print(f"Model completely saved to: '{save_path}'")

if __name__ == "__main__":
    DATASET_FILENAME = "loanapproval.csv"
    MODEL_FILENAME = "loan_bn_model_hybrid.pkl"
    
    top_5_features = ['credit_score', 'annual_income', 'existing_loans_count', 'num_dependents', 'employment_status']
    
    builder = BNModelBuilder()
    builder.build_and_train(DATASET_FILENAME, top_5_features)
    # builder.save_model(MODEL_FILENAME)