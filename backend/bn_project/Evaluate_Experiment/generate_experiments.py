import os
import sys
import time
import random
import json
import pandas as pd
import joblib
from pgmpy.inference import VariableElimination
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))               # Evaluate_Experiment
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                             # bn_project
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")                   # training folder

sys.path.append(PROJECT_ROOT)
sys.path.append(TRAINING_DIR)

from bn_model import BNModelBuilder
sys.modules['__main__'].BNModelBuilder = BNModelBuilder

print("🔮 Loading Bayesian Network Model...")

model_pkl_path = os.path.join(PROJECT_ROOT, "loan_bn_model.pkl")
builder = joblib.load(model_pkl_path)
network_model = builder.model
infer = VariableElimination(network_model)

def analyze_factors(evidence):
    """
    Dynamically identifies positive and negative risk factors from the evidence.
    """
    credit_map = {'0': 'Very Poor', '1': 'Low', '2': 'Fair', '3': 'Excellent'}
    income_map = {'0': 'Very Low', '1': 'Low', '2': 'Moderate', '3': 'High'}
    loans_map = {'0': 'None', '1': '1 Loan', '2': '2 Loans', '3': '3+ Loans'}
    deps_map = {'0': 'None', '1': '1 Dependent', '2': '2 Dependents', '3': '3+ Dependents'}
    emp_map = {'0': 'Unemployed', '1': 'Employed'}

    positives = []
    negatives = []

    # Credit Score analysis
    if evidence['credit_score'] in ['2', '3']:
        positives.append(f"Credit Score ({credit_map[evidence['credit_score']]})")
    else:
        negatives.append(f"Credit Score ({credit_map[evidence['credit_score']]})")

    # Income analysis
    if evidence['annual_income'] in ['2', '3']:
        positives.append(f"Annual Income ({income_map[evidence['annual_income']]})")
    else:
        negatives.append(f"Annual Income ({income_map[evidence['annual_income']]})")

    # Employment analysis
    if evidence['employment_status'] == '1':
        positives.append("Employment Status (Employed)")
    else:
        negatives.append("Employment Status (Unemployed)")

    # Liabilities analysis
    if evidence['existing_loans_count'] in ['2', '3']:
        negatives.append(f"Existing Liabilities ({loans_map[evidence['existing_loans_count']]})")
    if evidence['num_dependents'] in ['2', '3']:
        negatives.append(f"High Dependents ({deps_map[evidence['num_dependents']]})")

    return positives, negatives

def generate_dynamic_ground_truth(evidence, decision, prob_approved):
    positives, negatives = analyze_factors(evidence)
    
    if decision == 'Approved':
        main_reasons = ", ".join(positives) if positives else "overall balanced profile"
        gt = (f"Based on the Bayesian Network inference, the loan approval probability is {prob_approved:.1f}% (Approved). "
              f"The primary positive contributing factors are: {main_reasons}.")
        if negatives:
            gt += f" Minor risk factors considered include: {', '.join(negatives)}."
    else:
        main_reasons = ", ".join(negatives) if negatives else "accumulated borderline factors"
        gt = (f"Based on the Bayesian Network inference, the loan approval probability is {prob_approved:.1f}% (Rejected). "
              f"The decision is primarily driven by key risk factors: {main_reasons}.")
        if positives:
            gt += f" Positive attributes noted include: {', '.join(positives)}."
            
    return gt

def query_llama3_2(readable_profile_text, decision, prob):
    prompt = f"""
You are an expert Credit Risk Analyst and Model Auditor evaluating a Hierarchical Bayesian Network decision.

STRICT OPERATIONAL LAWS:
1. NO FLUFF: Start directly with the 3 required bullets. No introductory or concluding remarks.
2. STRICT NUMERICAL ALIGNMENT:
   - IF Probability > 75% AND profile metrics are mostly Good/Excellent: State "Aligned with banking logic (Strong Profile)".
   - IF Probability > 75% AND profile metrics are mostly Bad/Poor: State "CRITICAL ANOMALY: The model over-approved a subprime profile".
   - IF Probability <= 50% AND profile metrics are mostly Bad/Poor: State "Aligned with banking logic (Risk Correctly Captured)".
3. MITIGATING FACTOR RULE: Do not lazily copy 'None'. If the profile has excellent metrics (e.g., Credit > 750, No Debt, Stable Job), list them explicitly as strengths.
4. ZERO HALLUCINATION OF NUMBERS (CRITICAL): DO NOT invent, calculate, guess, or introduce any new percentages, weights, interest rates, loan terms (months), or dollar amounts. The ONLY allowed number/percentage in your entire output is the exact baseline probability ({prob:.2f}%) provided below. Use purely qualitative terms (e.g., high influence, preferential terms, flexible structures) for institutional recommendations.

DATA TO ANALYZE:
- Model Decision: {decision} | Probability: {prob:.2f}%
- Profile Metrics:
{readable_profile_text}

REQUIRED OUTPUT FORMAT (Strictly 3 Bullets):
- **Decision Status**: State decision, exact baseline probability ({prob:.2f}%), and the exact alignment phrase from Law #2. No other numbers allowed.
- **Risk Evaluation**: 
  - [IF PROFILE IS RISK-FREE]: State exactly "No significant financial risks detected; profile indicates maximum repayment reliability."
  - [IF PROFILE HAS RISKS]: Identify the specific negative inputs (e.g., low income, debt, unemployment) that threaten repayment. DO NOT include the phrase 'No significant financial risks detected' if any risk exists. Max 2 sentences. DO NOT invent numerical risk percentages or metrics.
- **Mitigating Strengths**: List the actual high-performing assets from the profile (e.g., High Income, Excellent Credit, Stable Job). If institutional structuring recommendations are made, use purely qualitative language (e.g., preferential interest rates or adjusted lines of credit) without stating specific figures, percentages, or loan terms. Max 2 sentences.
"""

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get('response', '').strip()
        else:
            return f"Error: Status code {response.status_code}"
    except Exception as e:
        return f"Call failed: {str(e)}"

def run_dataset_generation(num_scenarios=200):
    random.seed(42)  # تثبيت البذرة لضمان التكرارية الإحصائية
    scenarios = []
    seen_evidence = set()  # مجموعة لتتبع السيناريوهات السابقة ومنع التكرار

    print(f"بدء توليد {num_scenarios} سيناريو فريد واستخراج التفسيرات...")

    attempts = 0
    max_attempts = 10000  # حماية من الحلقات اللانهائية

    while len(scenarios) < num_scenarios and attempts < max_attempts:
        attempts += 1

        # 1. اختيار قيم عشوائية لمتغيرات العميل الـ 5
        ev = {
            'credit_score': str(random.choice([0, 1, 2, 3])),
            'annual_income': str(random.choice([0, 1, 2, 3])),
            'existing_loans_count': str(random.choice([0, 1, 2, 3])),
            'num_dependents': str(random.choice([0, 1, 2, 3])),
            'employment_status': str(random.choice([0, 1]))
        }

        # 2. التحقق من عدم تكرار السيناريو
        ev_tuple = tuple(sorted(ev.items()))
        if ev_tuple in seen_evidence:
            continue
        seen_evidence.add(ev_tuple)

        i = len(scenarios) + 1

        # 3. الاستدلال الاحتمالي البايزي (BN Inference)
        t0_bn = time.time()
        query_res = infer.query(variables=['Loan'], evidence=ev)
        t_bn = (time.time() - t0_bn) * 1000

        probs = query_res.values
        approved_idx = 1 if len(probs) > 1 else 0
        prob_approved = probs[approved_idx] * 100
        decision = "Approved" if prob_approved >= 50 else "Rejected"

        # 4. وصف السيناريو وتوليد النص المرجعي الديناميكي
        desc = (f"Credit Score level {ev['credit_score']}, "
                f"Income level {ev['annual_income']}, "
                f"Existing Loans {ev['existing_loans_count']}, "
                f"Dependents {ev['num_dependents']}, "
                f"Employment {ev['employment_status']}.")

        gt = generate_dynamic_ground_truth(ev, decision, prob_approved)

        # 5. استدعاء Llama 3.2 لتوليد التفسير الحواري
        t0_llm = time.time()
        explanation = query_llama3_2(desc, decision, prob_approved)
        t_llm = time.time() - t0_llm

        # 6. حفظ البيانات
        scenarios.append({
            'scenario_id': i,
            'evidence': ev,
            'desc': desc,
            'bn_decision': decision,
            'bn_prob_approved': round(prob_approved, 2),
            'bn_latency_ms': round(t_bn, 2),
            'llm_latency_s': round(t_llm, 2),
            'ground_truth': gt,
            'generated_explanation': explanation
        })

        print(f"Scenario {i:03d}/{num_scenarios} Generated | Decision: {decision:8s} | Latency: {t_llm:.2f}s")

    # 7. حفظ الملف النهائي
    with open("experiment_200_dataset.json", "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)

    print(f"\nComplete! Successfully generated {len(scenarios)} unique scenarios.")
    print("Saved dataset to 'experiment_200_dataset.json'.")

if __name__ == "__main__":
    run_dataset_generation()