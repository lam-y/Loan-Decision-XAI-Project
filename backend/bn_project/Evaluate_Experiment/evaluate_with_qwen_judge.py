import os
import json
import time
import requests
import pandas as pd
import numpy as np

# 1. إعداد المسارات المباشرة داخل نفس المجلد
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_INPUT_PATH = os.path.join(CURRENT_DIR, "LLM_Generated_Explanations_300_Fresh.json")
RESULTS_OUTPUT_PATH = os.path.join(CURRENT_DIR, "Qwen_Judge_Evaluation_300_Fresh.json")
EXCEL_OUTPUT_PATH = os.path.join(CURRENT_DIR, "Qwen_Judge_Evaluation_300_Fresh.xlsx")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
QWEN_MODEL_NAME = "qwen2.5" 

if not os.path.exists(JSON_INPUT_PATH):
    print(f"File not found: {JSON_INPUT_PATH}")
    exit()

with open(JSON_INPUT_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} samples from Fresh Dataset for Local Qwen Evaluation...")

SYSTEM_PROMPT = """
You are an expert evaluator in Explainable Artificial Intelligence (XAI) for credit risk assessment systems.
Your task is to critically evaluate an LLM-generated explanation against the Bayesian Network model outputs and ground truth context.

Score the explanation carefully on a scale of 1 to 5 for each of the following 3 metrics:
1. Faithfulness & Accuracy (1-5): Is the explanation strictly truthful to the BN Decision and Probability without hallucinations or contradictory facts?
2. Clarity & Readability (1-5): Is the explanation well-formatted, professional, transparent, and easy to read for a non-technical end-user?
3. Completeness & Coverage (1-5): Does the explanation sufficiently cover key factors influencing the decision?

Output MUST be strictly valid JSON without any markdown formatting or commentary:
{
  "faithfulness_score": <int 1-5>,
  "clarity_score": <int 1-5>,
  "completeness_score": <int 1-5>,
  "rationale": "<brief 1-sentence reason for the scores>"
}
"""

evaluation_results = []
excel_rows = []

faithfulness_scores = []
clarity_scores = []
completeness_scores = []

print("\nStarting Local Evaluation using Qwen Judge via Ollama...")

for idx, sample in enumerate(dataset):
    scenario_id = sample.get("scenario_id", idx + 1)
    evidence = sample.get("evidence", {})
    bn_decision = sample.get("bn_decision", "N/A")
    bn_prob = sample.get("bn_prob_approved", 0.0)
    ground_truth = sample.get("ground_truth", "")
    generated_exp = sample.get("generated_explanation", "")

    user_prompt = f"""
{SYSTEM_PROMPT}

--- CASE DETAILS ---
Applicant Profile / Evidence: {json.dumps(evidence)}
Bayesian Network Decision: {bn_decision}
Bayesian Network Probability (Approved): {bn_prob}%
Ground Truth Baseline: {ground_truth}

--- GENERATED EXPLANATION TO EVALUATE ---
{generated_exp}
"""

    payload = {
        "model": QWEN_MODEL_NAME,
        "prompt": user_prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        res_data = response.json()
        raw_response = res_data.get("response", "{}")
        
        res_json = json.loads(raw_response)
        
        f_score = float(res_json.get("faithfulness_score", 4))
        c_score = float(res_json.get("clarity_score", 4))
        comp_score = float(res_json.get("completeness_score", 4))
        rationale = res_json.get("rationale", "")

        faithfulness_scores.append(f_score)
        clarity_scores.append(c_score)
        completeness_scores.append(comp_score)

        record = {
            "scenario_id": scenario_id,
            "bn_decision": bn_decision,
            "bn_prob": bn_prob,
            "faithfulness_score": f_score,
            "clarity_score": c_score,
            "completeness_score": comp_score,
            "overall_avg": round((f_score + c_score + comp_score) / 3.0, 2),
            "qwen_rationale": rationale
        }

        evaluation_results.append(record)
        excel_rows.append(record)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(dataset):
            print(f"Qwen Progress: [{idx + 1}/{len(dataset)}] samples completed...")

    except Exception as e:
        print(f"Error at sample {idx+1}: {e}")

#  حساب المتوسطات والملخص النهائي
avg_f = np.mean(faithfulness_scores)
avg_c = np.mean(clarity_scores)
avg_comp = np.mean(completeness_scores)
overall_avg = np.mean([avg_f, avg_c, avg_comp])

summary_stats = {
    "Total_Samples_Evaluated": len(evaluation_results),
    "Qwen_Mean_Faithfulness": round(avg_f, 2),
    "Qwen_Mean_Clarity": round(avg_c, 2),
    "Qwen_Mean_Completeness": round(avg_comp, 2),
    "Qwen_Overall_Score_Mean": round(overall_avg, 2)
}

print("\n" + "="*55)
print("QWEN JUDGE EVALUATION SUMMARY (300 FRESH SAMPLES)")
print("="*55)
print(f"🔹 Total Samples Evaluated    : {len(evaluation_results)}")
print(f"🔹 Qwen Mean Faithfulness      : {avg_f:.2f} / 5.0")
print(f"🔹 Qwen Mean Clarity           : {avg_c:.2f} / 5.0")
print(f"🔹 Qwen Mean Completeness      : {avg_comp:.2f} / 5.0")
print(f"🔹 Qwen Overall Score Mean     : {overall_avg:.2f} / 5.0")
print("="*55)

with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"summary": summary_stats, "details": evaluation_results}, f, indent=4, ensure_ascii=False)

df_res = pd.DataFrame(excel_rows)
df_res.to_excel(EXCEL_OUTPUT_PATH, index=False)

print(f"\n Saved results to '{RESULTS_OUTPUT_PATH}' and '{EXCEL_OUTPUT_PATH}'")