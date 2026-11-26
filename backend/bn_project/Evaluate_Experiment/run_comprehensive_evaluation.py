import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import joblib
from pgmpy.inference import VariableElimination
import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. تحديد المسارات وهيكلة مجلد النتائج
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) in ['training', 'Evaluate_Experiment', 'api', 'data', 'results'] else CURRENT_DIR

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")
sys.path.append(PROJECT_ROOT)
sys.path.append(TRAINING_DIR)

try:
    from bn_model import BNModelBuilder
    sys.modules['__main__'].BNModelBuilder = BNModelBuilder
except ImportError:
    pass

model_path = None
for root, dirs, files in os.walk(PROJECT_ROOT):
    for f in files:
        if f.endswith('.pkl') and ('model' in f.lower() or 'bn' in f.lower()):
            model_path = os.path.join(root, f)
            break
    if model_path:
        break

test_csv_path = os.path.join(DATA_DIR, "test_dataset_new_30.csv")
if not os.path.exists(test_csv_path):
    test_csv_path = os.path.join(PROJECT_ROOT, "test_dataset_new_30.csv")

print(f" Loading Bayesian Network Model from: {model_path}")
builder = joblib.load(model_path)
network_model = builder.model if hasattr(builder, 'model') else builder
infer = VariableElimination(network_model)

print(f" Loading test dataset from: {test_csv_path}")
df_test = pd.read_csv(test_csv_path)
df_test.columns = df_test.columns.str.strip()

df_test_sample = df_test.head(3)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LLAMA_MODEL_NAME = "llama3.2"

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smoothing = SmoothingFunction().method1

def map_value_to_bin(val, feature_type):
    try:
        val = float(val)
    except:
        return str(val)
    
    if feature_type == 'credit_score':
        if val <= 445: return '0'
        elif val <= 575: return '1'
        elif val <= 719: return '2'
        else: return '3'
    elif feature_type == 'annual_income':
        if val <= 49266.25: return '0'
        elif val <= 82303.5: return '1'
        elif val <= 113904.75: return '2'
        else: return '3'
    elif feature_type in ['existing_loans_count', 'num_dependents']:
        return str(int(min(val, 3)))
    return str(val)

def generate_expert_ground_truth(row, decision, prob):
    c_score = row.get('credit_score', 0)
    income = row.get('annual_income', 0)
    loans = row.get('existing_loans_count', 0)
    
    alignment = "Aligned with banking logic" if (decision == "Approved" and c_score > 575) or (decision == "Rejected" and c_score <= 575) else "Borderline evaluation"
    
    gt = f"""- **Decision Status**: State decision as {decision}, exact baseline probability {prob:.2f}%, and {alignment}.
- **Risk Evaluation**: The profile reflects risk factors associated with a credit score of {c_score} and {loans} existing liabilities affecting debt-to-income ratio.
- **Mitigating Strengths**: The annual income level of {income} provides a baseline financial capacity supporting repayment capability."""
    return gt

def query_llama3_2(profile_text, decision, prob):
    prompt = f"""
You are an expert Credit Risk Analyst and Model Auditor evaluating a Hierarchical Bayesian Network decision.

STRICT OPERATIONAL LAWS:
1. NO FLUFF: Start directly with the 3 required bullets. No introductory or concluding remarks.
2. STRICT NUMERICAL ALIGNMENT:
   - IF Probability > 75% AND profile metrics are mostly Good/Excellent: State "Aligned with banking logic (Strong Profile)".
   - IF Probability > 75% AND profile metrics are mostly Bad/Poor: State "CRITICAL ANOMALY: The model over-approved a subprime profile".
   - IF Probability <= 50% AND profile metrics are mostly Bad/Poor: State "Aligned with banking logic (Risk Correctly Captured)".
3. ZERO HALLUCINATION OF NUMBERS: The ONLY allowed number/percentage in your entire output is the exact baseline probability ({prob:.2f}%) provided below.

DATA TO ANALYZE:
- Model Decision: {decision} | Probability: {prob:.2f}%
- Profile Metrics:
{profile_text}

REQUIRED OUTPUT FORMAT (Strictly 3 Bullets):
- **Decision Status**: State decision, exact baseline probability ({prob:.2f}%), and alignment phrase.
- **Risk Evaluation**: Identify specific negative inputs. Max 2 sentences.
- **Mitigating Strengths**: List high-performing assets. Max 2 sentences.
"""
    payload = {"model": LLAMA_MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get('response', '').strip()
    except Exception:
        pass
    return "Error generating explanation."

print(f"\nبدء الاختبار السريع على 3 حالات مع إضافة تقييم الشبكة لكل حالة...")

results_details = []
bn_true_labels, bn_pred_labels = [], []
r1_list, r2_list, rl_list, bleu_list = [], [], [], []

for idx, row in df_test_sample.iterrows():
    c_score_val = row.get('credit_score', 0)
    income_val = row.get('annual_income', 0)
    loans_val = row.get('existing_loans_count', 0)
    deps_val = row.get('num_dependents', 0)
    emp_raw = row.get('employment_status', 'Employed')
    emp_val = 1 if str(emp_raw).lower() in ['employed', 'self-employed', '1', 'yes'] else 0
    
    # القيمة الحقيقية للقرارات من الداتا سيت
    true_loan_raw = row.get('loan_approved', row.get('Loan', 1))
    true_loan = 1 if str(true_loan_raw) in ['1', 'True', 'Approved'] else 0
    true_decision_str = "Approved" if true_loan == 1 else "Rejected"

    evidence = {
        'credit_score': map_value_to_bin(c_score_val, 'credit_score'),
        'annual_income': map_value_to_bin(income_val, 'annual_income'),
        'existing_loans_count': map_value_to_bin(loans_val, 'existing_loans_count'),
        'num_dependents': map_value_to_bin(deps_val, 'num_dependents'),
        'employment_status': str(emp_val)
    }

    t0_bn = time.time()
    query_res = infer.query(variables=['Loan'], evidence=evidence)
    t_bn = (time.time() - t0_bn) * 1000
    probs = query_res.values
    approved_idx = 1 if len(probs) > 1 else 0
    prob_approved = probs[approved_idx] * 100
    decision = "Approved" if prob_approved >= 50 else "Rejected"
    pred_loan = 1 if decision == "Approved" else 0

    bn_true_labels.append(true_loan)
    bn_pred_labels.append(pred_loan)

    bn_match = "Yes" if decision == true_decision_str else "No"

    profile_desc = f"Credit Score: {c_score_val}, Income: {income_val}, Loans: {loans_val}, Dependents: {deps_val}, Employment: {emp_raw}"
    ground_truth = generate_expert_ground_truth(row, decision, prob_approved)

    t0_llm = time.time()
    explanation = query_llama3_2(profile_desc, decision, prob_approved)
    t_llm = time.time() - t0_llm

    scores = scorer.score(ground_truth, explanation)
    r1 = round(scores['rouge1'].fmeasure, 4)
    r2 = round(scores['rouge2'].fmeasure, 4)
    rl = round(scores['rougeL'].fmeasure, 4)
    bleu = round(sentence_bleu([ground_truth.lower().split()], explanation.lower().split(), smoothing_function=smoothing), 4)

    r1_list.append(r1); r2_list.append(r2); rl_list.append(rl); bleu_list.append(bleu)

    results_details.append({
        "Scenario_ID": idx + 1,
        "Credit_Score_Value": c_score_val,
        "Annual_Income_Value": income_val,
        "Existing_Loans_Count": loans_val,
        "Dependents_Count": deps_val,
        "Employment_Status": str(emp_raw),
        "True_Dataset_Decision": true_decision_str,
        "BN_Decision": decision,
        "BN_Decision_Match": bn_match,
        "BN_Prob_Approved_%": round(prob_approved, 2),
        "ROUGE-1": r1,
        "ROUGE-2": r2,
        "ROUGE-L": rl,
        "BLEU": bleu,
        "LLM_Latency_s": round(t_llm, 2),
        "Ground_Truth_Baseline": ground_truth,
        "Generated_Explanation": explanation
    })

    print(f"Sample {idx + 1} processed | BN Decision: {decision} | True: {true_decision_str} | Match: {bn_match}")


bn_acc = accuracy_score(bn_true_labels, bn_pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(bn_true_labels, bn_pred_labels, average='binary', zero_division=0)

df_details = pd.DataFrame(results_details)

summary_data = {
    "Metric_Name": [
        "Total Samples Evaluated",
        "Bayesian Network Accuracy",
        "Bayesian Network Precision",
        "Bayesian Network Recall",
        "Bayesian Network F1-Score",
        "ROUGE-1 Average",
        "ROUGE-2 Average",
        "ROUGE-L Average",
        "BLEU Average",
        "Avg LLM Latency (s)"
    ],
    "Score_Value": [
        len(results_details),
        round(bn_acc, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(sum(r1_list) / len(r1_list), 4) if r1_list else 0,
        round(sum(r2_list) / len(r2_list), 4) if r2_list else 0,
        round(sum(rl_list) / len(rl_list), 4) if rl_list else 0,
        round(sum(bleu_list) / len(bleu_list), 4) if bleu_list else 0,
        round(df_details["LLM_Latency_s"].mean(), 2)
    ]
}
df_summary = pd.DataFrame(summary_data)

# 6. التصدير لملف إكسل بصفحتين
excel_output_path = os.path.join(RESULTS_DIR, "Test_3_Samples_With_BN_Match.xlsx")
json_output_path = os.path.join(RESULTS_DIR, "Test_3_Samples_With_BN_Match.json")

with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump({"summary": summary_data, "details": results_details}, f, indent=4, ensure_ascii=False)

with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='Summary Metrics', index=False)
    df_details.to_excel(writer, sheet_name='Detailed Per-Scenario', index=False)

print("\n" + "="*60)
print(f"تم الانتهاء بنجاح وحفظ التقرير في:\n    {excel_output_path}")
print("="*60)