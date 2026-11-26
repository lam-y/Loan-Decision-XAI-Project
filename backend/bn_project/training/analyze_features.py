# analyze_features.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import os

def analyze_features_for_bn(file_name):
    # Setup paths and read data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else '.'
    data_path = os.path.join(BASE_DIR, "data", file_name)
    
    if not os.path.exists(data_path):
        print(f"File not found at {data_path}, trying local path.")
        data_path = file_name

    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()

    # Define target variable
    target_col = 'loan_approved'
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Remove identifiers if they exist
    if 'applicant_id' in data.columns:
        data = data.drop(columns=['applicant_id'])

    df_discrete = data.copy()

    # ---------------------------------------------------------
    # 1. Continuous to Categorical (Discretization)
    # ---------------------------------------------------------
    numeric_cols = df_discrete.select_dtypes(include=['int64', 'float64']).columns.drop(target_col, errors='ignore')
    
    for col in numeric_cols:
        if df_discrete[col].nunique() > 10:
            df_discrete[col] = pd.qcut(df_discrete[col], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
        df_discrete[col] = df_discrete[col].astype(str)

    # ---------------------------------------------------------
    # 2. Data Encoding 
    # ---------------------------------------------------------
    df_encoded = df_discrete.copy()
    for col in df_encoded.columns:
        # Golden Check: If the column is not numeric, encode it immediately
        if not pd.api.types.is_numeric_dtype(df_encoded[col]):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    # ---------------------------------------------------------
    # 3. Calculate importance using Decision Tree
    # ---------------------------------------------------------
    tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_model.fit(X, y)
    importance = pd.Series(tree_model.feature_importances_, index=X.columns)

    # ---------------------------------------------------------
    # 4. Calculate Mutual Information
    # ---------------------------------------------------------
    mi = mutual_info_classif(X, y, discrete_features=True, random_state=42)
    mi = pd.Series(mi, index=X.columns)

    # ---------------------------------------------------------
    # 5. Merge results and calculate final score
    # ---------------------------------------------------------
    results = pd.DataFrame({
        'tree_importance': importance,
        'mutual_information': mi
    })

    results['final_score'] = (results['mutual_information'] * 0.7) + (results['tree_importance'] * 0.3)
    results = results.sort_values(by='final_score', ascending=False)

    print("\n=== Final Ranking of Features Influencing the Loan Decision ===")
    print(results.round(4))

    # Taking the top 5 features + the target variable implicitly
    suggested_features = results.head(5).index.tolist()
    print("\nSuggested features for building the Bayesian Network structure:", suggested_features)

    return suggested_features, df_discrete

if __name__ == "__main__":
    selected_features, processed_data = analyze_features_for_bn("loanapproval.csv")