# training/evaluate_bn_performance.py
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from pgmpy.inference import VariableElimination

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from bn_model_hybrid import BNModelBuilder
except ImportError:
    raise ImportError("لم يتم العثور على ملف bn_model_hybrid.py! تأكد من وجوده في نفس المجلد.")

def evaluate_bayesian_network():
    print("Starting Bayesian Network Evaluation on 30% Test Set...")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else '.'
    model_path = os.path.join(BASE_DIR, "loan_bn_model_hybrid.pkl")
    test_csv_path = os.path.join(BASE_DIR, "data", "test_dataset_30.csv")
    
    if not os.path.exists(model_path) or not os.path.exists(test_csv_path):
        print("Error: Model file or test dataset not found! Please run bn_model_hybrid.py first.")
        return

    # 1. تحميل النموذج المكتمل والبيانات
    builder = joblib.load(model_path)
    test_df = pd.read_csv(test_csv_path)
    infer = VariableElimination(builder.model)
    
    y_true = test_df['Loan'].astype(str).tolist()
    y_pred = []
    
    print(f"Evaluating over {len(test_df)} test samples...")
    
    # 2. إجراء الاستدلال لكل حالة اختبار
    for idx, row in test_df.iterrows():
        evidence = {}
        for col in builder.selected_features:
            val = row[col]
            if col in builder.bins:
                binned_val = pd.cut([val], bins=builder.bins[col], labels=False)[0]
                evidence[col] = str(binned_val)
            elif col in builder.encoders:
                encoded_val = builder.encoders[col].transform([val])[0]
                evidence[col] = str(encoded_val)
            else:
                evidence[col] = str(int(val))
        
        emp = int(evidence['employment_status'])
        inc = int(evidence['annual_income'])
        dep = int(evidence['num_dependents'])
        cred = int(evidence['credit_score'])
        loans = int(evidence['existing_loans_count'])
        
        # Financial Capacity Latent Logic
        if emp in [0, 'No', 'unemployed']:
            fin_cap = 0 if inc < 1 else 1
        elif inc >= 2 and dep < 2:
            fin_cap = 2
        elif inc < 1:
            fin_cap = 0
        else:
            fin_cap = 1

        # Credit Worthiness Latent Logic
        if cred < 1:
            cred_worth = 0
        elif 1 <= cred < 3:
            cred_worth = 1 if loans <= 1 else 0
        else:
            cred_worth = 2 if loans == 0 else 1

        evidence['financial_capacity'] = str(fin_cap)
        evidence['credit_worthiness'] = str(cred_worth)
        
        # الاستدلال البايزي
        query_res = infer.query(variables=['Loan'], evidence=evidence)
        predicted_state = str(np.argmax(query_res.values))
        y_pred.append(predicted_state)

    # 3. حساب المقاييس القياسية
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print("\n" + "="*50)
    print("BAYESIAN NETWORK PERFORMANCE RESULTS (30% TEST SET)")
    print("="*50)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")
    print("="*50)
    
    # 4. حفظ النتائج وتصديرها بملف Excel منظم
    metrics_summary = pd.DataFrame([
        {'Metric': 'Accuracy', 'Score (%)': round(acc * 100, 2)},
        {'Metric': 'Precision (Weighted)', 'Score (%)': round(precision * 100, 2)},
        {'Metric': 'Recall (Weighted)', 'Score (%)': round(recall * 100, 2)},
        {'Metric': 'F1-Score (Weighted)', 'Score (%)': round(f1 * 100, 2)}
    ])
    
    report_dict = classification_report(y_true, y_pred, target_names=['Rejected (0)', 'Approved (1)'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df.rename(columns={'index': 'Class / Metric'}, inplace=True)
    
    output_excel = os.path.join(BASE_DIR, "Bayesian_Network_Evaluation_Metrics.xlsx")
    
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        metrics_summary.to_excel(writer, sheet_name='Summary Metrics', index=False)
        report_df.to_excel(writer, sheet_name='Classification Report', index=False)
        
    print(f"All evaluation metrics saved successfully to Excel: '{output_excel}'")

if __name__ == "__main__":
    evaluate_bayesian_network()