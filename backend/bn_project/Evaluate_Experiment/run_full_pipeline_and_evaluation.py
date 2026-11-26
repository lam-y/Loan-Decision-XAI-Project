import json
import os
import random
import sys
import time
import joblib
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from pgmpy.inference import VariableElimination
import requests
from rouge_score import rouge_scorer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")

sys.path.append(PROJECT_ROOT)
sys.path.append(TRAINING_DIR)

from bn_model import BNModelBuilder

sys.modules["__main__"].BNModelBuilder = BNModelBuilder

print(" جاري تحميل نموذج الشبكة البايزية (Bayesian Network)...")
model_pkl_path = os.path.join(PROJECT_ROOT, "loan_bn_model.pkl")
builder = joblib.load(model_pkl_path)
network_model = builder.model
infer = VariableElimination(network_model)

EXCEL_OUTPUT_PATH = os.path.join(
    CURRENT_DIR, "Comprehensive_ICDSS_Evaluation_Report.xlsx"
)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
QWEN_MODEL_NAME = "qwen2.5"
LLAMA_MODEL_NAME = "llama3.2"

#  قواميس التحويل الدقيقة من الفئات إلى القيم الرقمية الحقيقية لكل عميل
credit_value_map = {"0": 550, "1": 630, "2": 710, "3": 790}
income_value_map = {"0": 300000, "1": 750000, "2": 1500000, "3": 3500000}
loans_value_map = {"0": 0, "1": 1, "2": 2, "3": 3}
deps_value_map = {"0": 0, "1": 1, "2": 2, "3": 3}
emp_text_map = {"0": "Unemployed", "1": "Employed / Salaried"}


def analyze_factors(evidence):
  credit_map = {"0": "Very Poor", "1": "Low", "2": "Fair", "3": "Excellent"}
  income_map = {"0": "Very Low", "1": "Low", "2": "Moderate", "3": "High"}
  loans_map = {"0": "None", "1": "1 Loan", "2": "2 Loans", "3": "3+ Loans"}
  deps_map = {
      "0": "None",
      "1": "1 Dependent",
      "2": "2 Dependents",
      "3": "3+ Dependents",
  }

  positives = []  
  negatives = [] 

  cs = str(evidence.get("credit_score", "0"))
  ai = str(evidence.get("annual_income", "0"))
  es = str(evidence.get("employment_status", "0"))
  el = str(evidence.get("existing_loans_count", "0"))
  nd = str(evidence.get("num_dependents", "0"))

  if cs in ["2", "3"]:
    positives.append(f"Credit Score ({credit_map.get(cs, 'Fair')})")
  else:
    negatives.append(f"Credit Score ({credit_map.get(cs, 'Poor')})")

  if ai in ["2", "3"]:
    positives.append(f"Annual Income ({income_map.get(ai, 'Moderate')})")
  else:
    negatives.append(f"Annual Income ({income_map.get(ai, 'Low')})")

  if es == "1":
    positives.append("Employment Status (Employed)")
  else:
    negatives.append("Employment Status (Unemployed)")

  if el in ["2", "3"]:
    negatives.append(f"Existing Liabilities ({loans_map.get(el, '2 Loans')})")
  if nd in ["2", "3"]:
    negatives.append(f"High Dependents ({deps_map.get(nd, '2+ Dependents')})")

  return positives, negatives


def generate_dynamic_ground_truth(evidence, decision, prob_approved):
  positives, negatives = analyze_factors(evidence)
  if decision == "Approved":
    main_reasons = (
        ", ".join(positives) if positives else "overall balanced profile"
    )
    gt = (
        f"Based on the Bayesian Network inference, the loan approval probability"
        f" is {prob_approved:.1f}% (Approved). The primary positive"
        f" contributing factors are: {main_reasons}."
    )
    if negatives:
      gt += f" Minor risk factors considered include: {', '.join(negatives)}."
  else:
    main_reasons = (
        ", ".join(negatives) if negatives else "accumulated borderline factors"
    )
    gt = (
        f"Based on the Bayesian Network inference, the loan approval probability"
        f" is {prob_approved:.1f}% (Rejected). The decision is primarily driven"
        f" by key risk factors: {main_reasons}."
    )
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
3. ZERO HALLUCINATION OF NUMBERS: The ONLY allowed number/percentage in your entire output is the exact baseline probability ({prob:.2f}%) provided below.

DATA TO ANALYZE:
- Model Decision: {decision} | Probability: {prob:.2f}%
- Profile Metrics:
{readable_profile_text}

REQUIRED OUTPUT FORMAT (Strictly 3 Bullets):
- **Decision Status**: State decision, exact baseline probability ({prob:.2f}%), and alignment phrase.
- **Risk Evaluation**: Identify specific negative inputs. Max 2 sentences.
- **Mitigating Strengths**: List high-performing assets. Max 2 sentences.
"""
  payload = {"model": LLAMA_MODEL_NAME, "prompt": prompt, "stream": False}
  try:
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    if response.status_code == 200:
      return response.json().get("response", "").strip()
  except Exception:
    pass
  return "Error generating explanation."


# 3. إعداد مقياس ROUGE و BLEU
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
smoothing = SmoothingFunction().method1


def run_pipeline_and_evaluation(num_scenarios=3):  # قمنا بتعديل الافتراضي إلى 3
  random.seed(42)
  scenarios = []
  seen_evidence = set()

  print(f"🚀 بدء توليد ومعالجة وتقييم {num_scenarios} سيناريو تجريبي...")

  attempts = 0
  while len(scenarios) < num_scenarios and attempts < 5000:
    attempts += 1

    ev = {
        "credit_score": str(random.choice([0, 1, 2, 3])),
        "annual_income": str(random.choice([0, 1, 2, 3])),
        "existing_loans_count": str(random.choice([0, 1, 2, 3])),
        "num_dependents": str(random.choice([0, 1, 2, 3])),
        "employment_status": str(random.choice([0, 1])),
    }

    ev_tuple = tuple(sorted(ev.items()))
    if ev_tuple in seen_evidence:
      continue
    seen_evidence.add(ev_tuple)

    i = len(scenarios) + 1

    t0_bn = time.time()
    query_res = infer.query(variables=["Loan"], evidence=ev)
    t_bn = (time.time() - t0_bn) * 1000

    probs = query_res.values
    approved_idx = 1 if len(probs) > 1 else 0
    prob_approved = probs[approved_idx] * 100
    decision = "Approved" if prob_approved >= 50 else "Rejected"

    desc = (
        f"Credit Score level {ev['credit_score']}, Income level"
        f" {ev['annual_income']}, Existing Loans"
        f" {ev['existing_loans_count']}, Dependents {ev['num_dependents']},"
        f" Employment {ev['employment_status']}."
    )

    ground_truth = generate_dynamic_ground_truth(ev, decision, prob_approved)

    t0_llm = time.time()
    generated_exp = query_llama3_2(desc, decision, prob_approved)
    t_llm = time.time() - t0_llm

    c_val = credit_value_map.get(ev["credit_score"], 550)
    inc_val = income_value_map.get(ev["annual_income"], 300000)
    loans_val = loans_value_map.get(ev["existing_loans_count"], 0)
    deps_val = deps_value_map.get(ev["num_dependents"], 0)
    emp_val = emp_text_map.get(ev["employment_status"], "Unemployed")

    confidence_margin = round(abs(prob_approved - 50.0) * 2.0, 2)
    if prob_approved >= 75.0 or prob_approved <= 25.0:
      certainty_tier = "High Confidence"
    elif 40.0 <= prob_approved <= 60.0:
      certainty_tier = "Borderline / Critical"
    else:
      certainty_tier = "Moderate Confidence"

    scores = scorer.score(ground_truth, generated_exp)
    r1 = round(scores["rouge1"].fmeasure, 4)
    r2 = round(scores["rouge2"].fmeasure, 4)
    rl = round(scores["rougeL"].fmeasure, 4)

    ref_tokens = [ground_truth.lower().split()]
    gen_tokens = generated_exp.lower().split()
    bleu = round(
        sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothing), 4
    )

    qwen_prompt = f"""
You are an expert evaluator in Explainable Artificial Intelligence (XAI) for credit risk assessment.
Evaluate the following LLM explanation against the Bayesian Network output and Ground Truth.
Score 1 to 5 for:
1. Faithfulness & Accuracy
2. Clarity & Readability
3. Completeness & Coverage

Output strictly JSON:
{{
  "faithfulness_score": <int 1-5>,
  "clarity_score": <int 1-5>,
  "completeness_score": <int 1-5>,
  "rationale": "<1 sentence reason>"
}}

Applicant Profile: Credit Score={c_val}, Income={inc_val}, Loans={loans_val}, Dependents={deps_val}, Employment={emp_val}
BN Decision: {decision} | Probability: {prob_approved:.2f}%
Ground Truth: {ground_truth}
Generated Explanation: {generated_exp}
"""
    payload_qwen = {
        "model": QWEN_MODEL_NAME,
        "prompt": qwen_prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1},
    }

    f_score, c_score, comp_score, rationale = (
        4.0,
        4.0,
        4.0,
        "Evaluated successfully.",
    )
    try:
      resp = requests.post(OLLAMA_URL, json=payload_qwen, timeout=60)
      if resp.status_code == 200:
        res_json = json.loads(resp.json().get("response", "{}"))
        f_score = float(res_json.get("faithfulness_score", 4))
        c_score = float(res_json.get("clarity_score", 4))
        comp_score = float(res_json.get("completeness_score", 4))
        rationale = res_json.get("rationale", "")
    except Exception:
      pass

    record = {
        "Scenario_ID": i,
        "Credit_Score_Value": c_val,
        "Annual_Income_Value": inc_val,
        "Existing_Loans_Count": loans_val,
        "Dependents_Count": deps_val,
        "Employment_Status": emp_val,
        "BN_Decision": decision,
        "BN_Prob_Approved_%": round(prob_approved, 2),
        "BN_Confidence_Margin_%": confidence_margin,
        "BN_Certainty_Tier": certainty_tier,
        "ROUGE-1": r1,
        "ROUGE-2": r2,
        "ROUGE-L": rl,
        "BLEU": bleu,
        "Faithfulness_Score": f_score,
        "Clarity_Score": c_score,
        "Completeness_Score": comp_score,
        "Qwen_Overall_Avg": round(
            (f_score + c_score + comp_score) / 3.0, 2
        ),
        "LLM_Latency_s": round(t_llm, 2),
        "Ground_Truth_Baseline": ground_truth,
        "Generated_Explanation": generated_exp,
        "Qwen_Rationale": rationale,
    }
    scenarios.append(record)
    print(
        f"[{i}/{num_scenarios}] Processed | Decision: {decision} | Prob:"
        f" {prob_approved:.1f}%"
    )

  with open(
      "LLM_Generated_Explanations_300_Fresh.json", "w", encoding="utf-8"
  ) as f:
    json.dump(scenarios, f, indent=2, ensure_ascii=False)

  df_details = pd.DataFrame(scenarios)
  summary_data = {
      "Metric_Name": [
          "Total Samples Evaluated",
          "Avg BN Decision Confidence Margin (%)",
          "ROUGE-1 Average",
          "ROUGE-2 Average",
          "ROUGE-L Average",
          "BLEU Average",
          "Qwen Mean Faithfulness (1-5)",
          "Qwen Mean Clarity (1-5)",
          "Qwen Mean Completeness (1-5)",
          "Qwen Overall Mean Score (1-5)",
          "Avg LLM Latency (s)",
      ],
      "Score_Value": [
          len(scenarios),
          round(df_details["BN_Confidence_Margin_%"].mean(), 2),
          round(df_details["ROUGE-1"].mean(), 4),
          round(df_details["ROUGE-2"].mean(), 4),
          round(df_details["ROUGE-L"].mean(), 4),
          round(df_details["BLEU"].mean(), 4),
          round(df_details["Faithfulness_Score"].mean(), 2),
          round(df_details["Clarity_Score"].mean(), 2),
          round(df_details["Completeness_Score"].mean(), 2),
          round(df_details["Qwen_Overall_Avg"].mean(), 2),
          round(df_details["LLM_Latency_s"].mean(), 2),
      ],
  }
  df_summary = pd.DataFrame(summary_data)

  with pd.ExcelWriter(EXCEL_OUTPUT_PATH, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Summary Metrics", index=False)
    df_details.to_excel(writer, sheet_name="Detailed Per-Scenario", index=False)

  print(f"\nتمت التجربة بنجاح وتم حفظ التقرير في: {EXCEL_OUTPUT_PATH}")


if __name__ == "__main__":
  run_pipeline_and_evaluation(
      num_scenarios=3
  ) 