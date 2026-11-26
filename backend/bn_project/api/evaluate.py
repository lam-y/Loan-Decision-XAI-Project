from fastapi import APIRouter
from pydantic import BaseModel, Field
import pandas as pd
import json
import requests
from pgmpy.inference import VariableElimination

router = APIRouter()

class LoanInput(BaseModel):
    creditScore: int = Field(..., ge=300, le=850)
    income: int = Field(..., ge=0)
    existingLoans: int = Field(..., ge=0) 
    dependents: int = Field(..., ge=0)
    employment: int = Field(..., ge=0, le=1)
def get_approval_probability(network_model, evidence_dict):
    string_evidence = {k: str(v) for k, v in evidence_dict.items()}
    infer = VariableElimination(network_model)
    query_result = infer.query(variables=['Loan'], evidence=string_evidence)
    return float(query_result.values[1]) * 100

def create_evaluate_router(builder, ask_llm, set_last_evaluation):
    network_model = builder.model
    bins = builder.bins

    @router.post("/evaluate")
    def evaluate_loan(data: LoanInput):
        print("[FastAPI] Received an evaluation request.")

        if data.existingLoans > 10 or data.dependents > 10:
            unreal_value = max(data.existingLoans, data.dependents)
            error_msg = f"The inputted value ({unreal_value}) is unrealistically high. Max allowed limit is 10."
            return {
                "decision": "Input Validation Error",
                "probability": 0.0,
                "llm_explanation": error_msg
            }
        
        # 1. التقطيع الرقمي (Binning)
        try:
            cs_series = pd.cut([data.creditScore], bins=bins['credit_score'], labels=False, include_lowest=True)
            cs_bin = int(cs_series[0])
        except:
            cs_bin = 0 

        try:
            inc_series = pd.cut([data.income], bins=bins['annual_income'], labels=False, include_lowest=True)
            inc_bin = int(inc_series[0])
        except:
            inc_bin = 0 

        el_bin = min(int(data.existingLoans), 3)   
        dep_bin = min(int(data.dependents), 3)    
        emp_bin = 1 if data.employment >= 1 else 0

        # 2. حساب الركائز الوسيطة
        fin_score = inc_bin + emp_bin - int(dep_bin >= 2)
        financial_capacity = 0 if fin_score <= 0 else (1 if fin_score <= 2 else 2)

        cred_score = cs_bin - el_bin
        credit_worthiness = 0 if cred_score <= 0 else (1 if cred_score <= 1 else 2)

        inputs_dict = {
            'credit_score': cs_bin,
            'annual_income': inc_bin,
            'existing_loans_count': el_bin,
            'num_dependents': dep_bin,
            'employment_status': emp_bin,
            'financial_capacity': financial_capacity,   
            'credit_worthiness': credit_worthiness     
        }

        # 3. حساب الاحتمالية وصياغة القرار
        prob = get_approval_probability(network_model, inputs_dict)

        if prob >= 60: decision = "Likely Approved"
        elif prob <= 40: decision = "Likely Rejected"
        else: decision = "Borderline - Requires Manual Human Review"

        # 4. تحضير المسميات البشرية المصرفية النظيفة (بدون كلمة Pillar)
        human_labels = {
            "credit_score": {0: "Very Poor (Subprime)", 1: "Poor", 2: "Good", 3: "Excellent"},
            "annual_income": {0: "Very Low", 1: "Low", 2: "Medium", 3: "High"},
            "existing_loans_count": {0: "No Debt Burden", 1: "1 Active Loan", 2: "2 Active Loans", 3: "3+ Loans"},
            "num_dependents": {0: "No Dependents", 1: "1 Dependent", 2: "2 Dependents", 3: "3+ Dependents"},
            "employment_status": {0: "Unemployed", 1: "Employed (Stable)"},
            "financial_capacity": {0: "Low Financial Capacity", 1: "Medium Financial Capacity", 2: "High Financial Capacity"},
            "credit_worthiness": {0: "Poor Credit Worthiness (High Risk)", 1: "Fair Credit Worthiness", 2: "Excellent Credit Worthiness"}
        }

        readable_profile = {
            "Credit Score Level": human_labels["credit_score"].get(cs_bin, "Unknown"),
            "Income Level": human_labels["annual_income"].get(inc_bin, "Unknown"),
            "Current Debt Burden": human_labels["existing_loans_count"].get(el_bin, "Unknown"),
            "Family Responsibilities": human_labels["num_dependents"].get(dep_bin, "Unknown"),
            "Job Stability": human_labels["employment_status"].get(emp_bin, "Unknown"),
            "Financial Capacity Assessment": human_labels["financial_capacity"].get(financial_capacity, "Unknown"),
            "Credit Worthiness Evaluation": human_labels["credit_worthiness"].get(credit_worthiness, "Unknown")
        }
        
        readable_profile_text = json.dumps(readable_profile, indent=2)

        set_last_evaluation({"prob": prob, "inputs": inputs_dict})

        # 5. البرومبت 
        llm_prompt = f"""
You are an expert Credit Risk Analyst evaluating a Hierarchical Bayesian Network output for a bank loan officer.
Analyze how the underlying raw features shape the intermediate assessments (Financial Capacity & Credit Worthiness) to drive the decision.

STRICT OPERATIONAL LAWS:
1. NO FLUFF: Output EXACTLY 3 bullet points starting immediately with no greetings, titles, or concluding remarks.
2. PROFESSIONAL ALIGNMENT & TRADE-OFF LOGIC:
   - IF Probability >= 60% AND metrics are mostly Good/Excellent: State "Aligned with banking logic (Strong Profile)".
   - IF Probability >= 60% AND metrics contain severe risks (e.g. Unemployed, Poor Credit): State "High Approval Boundary Driven by Historical Feature Interaction / Compensating Factors". Explain how the model's Bayesian weight assigned higher priority to capacity over credit risk based on historical data.
   - IF Probability < 60%: State "Aligned with banking logic (Risk Factors Correctly Reflected)".
3. ZERO NUMERICAL FABRICATION (CRITICAL):
   - The ONLY allowed percentage figure in your entire response is the exact baseline probability ({prob:.2f}%).
   - NEVER invent, guess, calculate, or introduce external interest rates, score points, loan tenures, or monetary amounts.
4. TERMINOLOGY: Never use technical terms like "Pillar 1", "Pillar 2", or "Bin". Use "Financial Capacity Assessment" and "Credit Worthiness Evaluation".

DATA TO ANALYZE:
- Model Decision: {decision} | Probability: {prob:.2f}%
- Profile Metrics:
{readable_profile_text}

REQUIRED OUTPUT FORMAT (Strictly 3 Bullets):
- **Decision Status**: State decision, exact baseline probability ({prob:.2f}%), and the exact alignment phrase from Law #2.
- **Risk Evaluation**: Identify specific risk factors present (e.g., poor credit, unemployment, active debt). If no risks exist, state exactly: "No significant financial risks detected; profile indicates maximum repayment reliability."
- **Mitigating Strengths**: Highlight high-performing factors (e.g., High Income, Job Stability). Explain any compensating trade-offs where strong factors counterbalance weak ones in the Bayesian logic.
"""
        
        try:
            print("[Ollama] Processing prompt under optimized constraints...")
            ollama_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": llm_prompt,
                    "stream": False,
                    "options": {
                        "num_thread": 4,      
                        "num_predict": 140,  
                        "top_p": 0.9
                    }
                },
                timeout=60 
            )
            data_json = ollama_response.json()
            explanation = data_json.get("response", "Error reading analytical engine output.")
        except requests.exceptions.Timeout:
            explanation = "Analysis Timeout: The local risk interpretation engine exceeded safe processing limits. Please try running the evaluation again."
        except Exception as e:
            explanation = f"Analytical Engine Connection Error: {str(e)}"

        return {
            "decision": decision,
            "probability": round(prob, 2),
            "llm_explanation": explanation
        }

    return router