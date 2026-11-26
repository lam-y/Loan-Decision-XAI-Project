# bn_model.py
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import DiscreteBayesianNetwork
# تنظيف الاستيرادات والاعتماد على العناصر الأساسية المستقرة فقط
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, ExpertKnowledge

class BNModelBuilder:
    def __init__(self):
        self.model = None
        self.bins = {}       
        self.encoders = {}   
        self.selected_features = []
        
    def build_and_train(self, data_filename, top_features):
        print("Starting Hybrid Bayesian Network Build Process...")
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
        
        # 2. Logical & Customized Preprocessing
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
                print(f" {col} -> Kept as native Discrete Integers (No Binning).")
                
        # تحويل الجدول بالكامل لنصوص لتهيئة البيئة المتقطعة (Discrete Environment)
        df_train = df_train.dropna().astype(str)

        # 3. Hybrid Structure Learning (Domain Knowledge + Data Discovery)
        print("\nDiscovering deep causal chain using Hybrid Knowledge...")
        
        required_edges = [(feature, 'Loan') for feature in top_features]
        forbidden_edges = [('Loan', feature) for feature in top_features]
        
        expert_knowledge = ExpertKnowledge(
            required_edges=required_edges,
            forbidden_edges=forbidden_edges
        )
        
        # استخدام الاسم النصي القياسي والمقبول عالمياً في pgmpy للبيانات المتقطعة وهو 'k2'
        hc = HillClimbSearch(df_train)
        best_model_dag = hc.estimate(
            scoring_method='k2',   
            expert_knowledge=expert_knowledge
        )
        
        self.model = DiscreteBayesianNetwork(best_model_dag.edges())
        
        print("\nFinal Discovered Causal Network Structure:")
        for edge in self.model.edges():
            is_internal = "[Discovered Interaction]" if edge[1] != 'Loan' else "🧠 [Domain Required]"
            print(f"   {edge[0]:<20} ---> {edge[1]:<10} {is_internal}")

        # 4. Parameter Learning (CPTs Calculation)
        print("\nComputing Conditional Probability Tables (Scientific Weighting)...")
        estimator = BayesianEstimator(self.model, df_train)
        cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=5)
        
        for cpd in cpds:
            self.model.add_cpds(cpd)
            
        self.model.check_model()
        print("Ultimate Hybrid Bayesian Network successfully trained and validated!")

    def save_model(self, save_path="loan_bn_model.pkl"):
        joblib.dump(self, save_path)
        print(f"Model completely saved to: '{save_path}'")


if __name__ == "__main__":
    DATASET_FILENAME = "loanapproval.csv"
    MODEL_FILENAME = "loan_bn_model.pkl"
    
    top_5_features = [
        'credit_score',
        'annual_income',
        'existing_loans_count',
        'num_dependents',
        'employment_status'
    ]
    
    builder = BNModelBuilder()
    builder.build_and_train(DATASET_FILENAME, top_5_features)
    builder.save_model(MODEL_FILENAME)