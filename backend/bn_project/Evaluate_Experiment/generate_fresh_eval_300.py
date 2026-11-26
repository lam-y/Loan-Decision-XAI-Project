import os
import json
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

# 1. تحديد المسارات
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(EXP_DIR)
DATASET_PATH = os.path.join(PARENT_DIR, "data", "loan_data.csv")

# 2. دالة توليد النص المرجعي المعياري (Rule-Based Ground Truth)
def generate_ground_truth_template(decision, probability, cibil, income, deps, emp):
    """
    توليد نص مرجعي معيار صلب ودقيق يعتمد على القواعد بناءً على مخرجات الشبكة والمدخلات
    """
    emp_str = "Self-Employed" if emp == 1 else "Salaried/Other"
    risk_level = "low risk" if probability >= 70 else ("moderate risk" if probability >= 40 else "high risk")
    
    gt_text = (
        f"The loan application decision is {decision} with a calculated probability of {probability:.2f}%, "
        f"representing a {risk_level} evaluation. "
        f"Key applicant attributes considered by the Bayesian Network model include a CIBIL score of {cibil}, "
        f"annual income of {income}, {deps} dependents, and employment status as {emp_str}."
    )
    return gt_text

# 3. قراءة البيانات وتقسيمها
if os.path.exists(DATASET_PATH):
    df_full = pd.read_csv(DATASET_PATH)
    df_full.columns = df_full.columns.str.strip()
    
    # التقسيم النظيف (أخذ عينة الاختبار)
    _, df_test = train_test_split(df_full, test_size=300, random_state=42)
    test_csv_path = os.path.join(EXP_DIR, "Dataset_Test_300_Fresh.csv")
    df_test.to_csv(test_csv_path, index=False)
    print(f"Saved 300 test samples to '{test_csv_path}'.")
else:
    print(f"File not found at: {DATASET_PATH}")
    exit()

# تجربة سريعة على أول 3 حالات فقط
df_test_sample = df_test

FASTAPI_URL = "http://127.0.0.1:8000/evaluate"
generated_results = []
excel_rows = []

print("\nStarting FAST TEST for 3 samples only...")

for idx, (_, row) in enumerate(df_test_sample.iterrows()):
    c_score = int(row.get("cibil_score", 0))
    inc = int(row.get("income_annum", 0))
    deps = int(row.get("no_of_dependents", 0))
    emp_raw = str(row.get("self_employed", "No")).strip()
    emp = 1 if emp_raw.lower() in ["yes", "1", "true"] else 0
    e_loans = int(row.get("existing_loans_count", row.get("existingLoans", 0)))

    payload = {
        "creditScore": c_score,
        "income": inc,
        "existingLoans": e_loans,
        "dependents": deps,
        "employment": emp,
        "cibil_score": c_score,
        "income_annum": inc,
        "no_of_dependents": deps,
        "self_employed": emp
    }

    try:
        response = requests.post(FASTAPI_URL, json=payload, timeout=60)
        res_data = response.json()

        bn_decision = res_data.get("decision", "Rejected")
        bn_prob = res_data.get("probability", 0.0)
        llm_exp = res_data.get("llm_explanation") or res_data.get("explanation") or res_data.get("generated_explanation") or ""

        ground_truth = generate_ground_truth_template(
            decision=bn_decision,
            probability=bn_prob,
            cibil=c_score,
            income=inc,
            deps=deps,
            emp=emp
        )

        sample_json = {
            "scenario_id": idx + 1,
            "evidence": {
                "cibil_score": c_score,
                "income_annum": inc,
                "existing_loans": e_loans,
                "no_of_dependents": deps,
                "self_employed": emp
            },
            "desc": f"CIBIL Score {c_score}, Income {inc}, Dependents {deps}, Employment {emp}.",
            "bn_decision": bn_decision,
            "bn_prob_approved": bn_prob,
            "ground_truth": ground_truth,
            "generated_explanation": llm_exp
        }
        generated_results.append(sample_json)

        excel_rows.append({
            "Scenario_ID": idx + 1,
            "CIBIL_Score": c_score,
            "Income_Annum": inc,
            "Existing_Loans": e_loans,
            "Dependents": deps,
            "Employment": emp,
            "BN_Decision": bn_decision,
            "BN_Prob_Approved": bn_prob,
            "Ground_Truth": ground_truth,
            "Generated_Explanation": llm_exp
        })

        print(f"Processed sample {idx + 1}/3 successfully!")

    except Exception as e:
        print(f"Error at sample {idx + 1}: {e}")

# حفظ النتائج التجريبية
json_output_path = os.path.join(EXP_DIR, "LLM_Generated_Explanations_300_Fresh.json")
excel_output_path = os.path.join(EXP_DIR, "LLM_Generated_Explanations_300_Fresh.xlsx")

with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(generated_results, f, indent=4, ensure_ascii=False)

pd.DataFrame(excel_rows).to_excel(excel_output_path, index=False)

print("\nFAST 3-SAMPLE TEST COMPLETED!")
print("Check 'LLM_Generated_Explanations_300_Fresh.json' to confirm 'generated_explanation' is non-empty!")