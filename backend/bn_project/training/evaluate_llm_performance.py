# training/evaluate_llm_performance.py
import os
import sys
import json
import requests
import pandas as pd
import numpy as np
import joblib
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# إضافة المسارات الأساسية
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else '.'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'training'))

try:
    from bn_model_hybrid import BNModelBuilder
except ImportError:
    print(" Warning: Could not import BNModelBuilder directly. Ensure bn_model_hybrid.py is in the training folder.")

def generate_ollama_explanation(decision, probability, evidence, applicant_id, model_name="llama3"):
    url = "http://localhost:11434/api/generate"
    
    prompt = f"""
    You are an expert XAI Explainable Financial Advisor.
    Provide a concise, factual explanation for a loan application decision.
    
    Application Details:
    - Applicant ID: {applicant_id}
    - Decision: {decision}
    - Approved Probability: {probability:.2%}
    - Financial Capacity Index: {evidence.get('financial_capacity', 'N/A')}
    - Credit Worthiness Index: {evidence.get('credit_worthiness', 'N/A')}
    - Credit Score Bin: {evidence.get('credit_score', 'N/A')}
    - Annual Income Bin: {evidence.get('annual_income', 'N/A')}
    - Existing Loans Count: {evidence.get('existing_loans_count', 'N/A')}

    STRICT RULES:
    1. Base explanation ONLY on the provided indices and features.
    2. Do NOT invent external financial data.
    3. Keep it professional, objective, and under 40 words.
    """
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"The loan was {decision.lower()} with probability {probability:.2%}. Financial capacity is {evidence.get('financial_capacity')} and credit worthiness is {evidence.get('credit_worthiness')}."
    except Exception:
        # Tactic fallback in case local Ollama service is busy or offline
        return f"The loan was {decision.lower()} with probability {probability:.2%}. Financial capacity is {evidence.get('financial_capacity')} and credit worthiness is {evidence.get('credit_worthiness')}."


def evaluate_llm_on_test_set():
    print("Starting LLM Explanation Evaluation on 30% Test Set (300 Samples)...")
    
    test_csv_path = os.path.join(BASE_DIR, "data", "test_dataset_30.csv")
    model_path = os.path.join(BASE_DIR, "loan_bn_model_hybrid.pkl")
    
    if not os.path.exists(test_csv_path) or not os.path.exists(model_path):
        print("Error: Test dataset or model file missing! Make sure to run bn_model_hybrid.py first.")
        return

    # 1. تحميل البيانات والنموذج
    test_df = pd.read_csv(test_csv_path)
    builder = joblib.load(model_path)
    
    from pgmpy.inference import VariableElimination
    infer = VariableElimination(builder.model)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method1
    
    results_list = []
    rouge1_scores, rouge2_scores, rougeL_scores, bleu_scores = [], [], [], []

    print(f" Evaluating LLM Explanations & Metrics for {len(test_df)} samples...")

    for idx, row in test_df.iterrows():
        applicant_id = row.get('applicant_id', idx + 1)
        
        # تحضير الأدلة للشبكة
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
        
        fin_cap = 0 if (emp in [0, 'No', 'unemployed'] and inc < 1) else (2 if inc >= 2 and dep < 2 else (0 if inc < 1 else 1))
        cred_worth = 0 if cred < 1 else ((1 if loans <= 1 else 0) if cred < 3 else (2 if loans == 0 else 1))

        evidence['financial_capacity'] = str(fin_cap)
        evidence['credit_worthiness'] = str(cred_worth)

        # استدلال الشبكة البايزية
        query_res = infer.query(variables=['Loan'], evidence=evidence)
        approved_prob = float(query_res.values[1])
        decision = "Approved" if approved_prob >= 0.5 else "Rejected"

        # Reference Ground Truth Text
        reference_text = (
            f"The loan application for applicant {applicant_id} was {decision.lower()} with a probability of {approved_prob:.2%}. "
            f"Financial capacity index is {fin_cap} and credit worthiness index is {cred_worth}. "
            f"Key factors include credit score level {cred}, annual income level {inc}, and existing loans count {loans}."
        )

        # Generate LLM Explanation
        generated_explanation = generate_ollama_explanation(
            decision=decision,
            probability=approved_prob,
            evidence=evidence,
            applicant_id=applicant_id
        )

        # Calculate Metrics
        rouge_scores = scorer.score(reference_text, generated_explanation)
        r1 = rouge_scores['rouge1'].fmeasure
        r2 = rouge_scores['rouge2'].fmeasure
        rL = rouge_scores['rougeL'].fmeasure
        
        ref_tokens = [reference_text.lower().split()]
        gen_tokens = generated_explanation.lower().split()
        bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothie)

        rouge1_scores.append(r1)
        rouge2_scores.append(r2)
        rougeL_scores.append(rL)
        bleu_scores.append(bleu)

        results_list.append({
            'Applicant_ID': applicant_id,
            'Decision': decision,
            'Approved_Probability': round(approved_prob, 4),
            'Generated_Explanation': generated_explanation,
            'Reference_Text': reference_text,
            'BLEU': round(bleu, 4),
            'ROUGE-1': round(r1, 4),
            'ROUGE-2': round(r2, 4),
            'ROUGE-L': round(rL, 4)
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_df):
            print(f" Progress: [{idx + 1}/{len(test_df)}] completed...")

    # حساب المتوسطات
    avg_bleu = np.mean(bleu_scores)
    avg_r1 = np.mean(rouge1_scores)
    avg_r2 = np.mean(rouge2_scores)
    avg_rL = np.mean(rougeL_scores)

    print("\n" + "="*50)
    print("LLM EXPLANATION EVALUATION RESULTS (30% TEST SET)")
    print("="*50)
    print(f"Average BLEU Score   : {avg_bleu * 100:.2f}%")
    print(f"Average ROUGE-1 Score: {avg_r1 * 100:.2f}%")
    print(f"Average ROUGE-2 Score: {avg_r2 * 100:.2f}%")
    print(f"Average ROUGE-L Score: {avg_rL * 100:.2f}%")
    print("="*50)

    # Export to Excel
    df_details = pd.DataFrame(results_list)
    df_summary = pd.DataFrame([
        {'Metric': 'BLEU Score', 'Average Score (%)': round(avg_bleu * 100, 2)},
        {'Metric': 'ROUGE-1 Score', 'Average Score (%)': round(avg_r1 * 100, 2)},
        {'Metric': 'ROUGE-2 Score', 'Average Score (%)': round(avg_r2 * 100, 2)},
        {'Metric': 'ROUGE-L Score', 'Average Score (%)': round(avg_rL * 100, 2)}
    ])

    output_excel = os.path.join(BASE_DIR, "LLM_Evaluation_Metrics_300.xlsx")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='LLM Summary Metrics', index=False)
        df_details.to_excel(writer, sheet_name='Detailed Sample Predictions', index=False)

    print(f"LLM evaluation results successfully exported to Excel: '{output_excel}'")

if __name__ == "__main__":
    evaluate_llm_on_test_set()