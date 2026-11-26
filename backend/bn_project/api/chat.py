# chat.py
from fastapi import APIRouter
from pydantic import BaseModel
import json
import requests

router = APIRouter()

# ذاكرة الجلسة لمشاركة البيانات بين الموديلات
last_evaluation = {}

class ChatRequest(BaseModel):
    message: str

def set_last_evaluation(data):
    global last_evaluation
    last_evaluation = data

@router.post("/chat")
def chat_with_system(req: ChatRequest):
    user_raw_message = req.message
    message_lower = user_raw_message.lower()

    # 1. منع الاستعلامات العشوائية قبل التقييم
    if not last_evaluation:
        return {"response": "Welcome! Please navigate to the evaluation form, submit the applicant's financial data first, and I will be ready to discuss the hierarchical risk analysis with you here."}

    prob = last_evaluation.get("prob", 0.0)
    inputs = last_evaluation.get("inputs", {})

    # 2. قاموس المسميات التفسيرية النظيفة (بدون كلمة Pillar)
    human_labels = {
        "credit_score": {"0": "Very Poor (Subprime)", "1": "Poor", "2": "Good", "3": "Excellent", 0: "Very Poor (Subprime)", 1: "Poor", 2: "Good", 3: "Excellent"},
        "annual_income": {"0": "Very Low", "1": "Low", "2": "Medium", "3": "High", 0: "Very Low", 1: "Low", 2: "Medium", 3: "High"},
        "existing_loans_count": {"0": "No Debt Burden", "1": "1 Active Loan", "2": "2 Active Loans", "3": "3+ Active Loans", 0: "No Debt Burden", 1: "1 Active Loan", 2: "2 Active Loans", 3: "3+ Active Loans"},
        "num_dependents": {"0": "No Dependents", "1": "1 Dependent", "2": "2 Dependents", "3": "3+ Dependents", 0: "No Dependents", 1: "1 Dependent", 2: "2 Dependents", 3: "3+ Dependents"},
        "employment_status": {"0": "Unemployed", "1": "Employed / Stable Income", 0: "Unemployed", 1: "Employed / Stable Income"},
        "financial_capacity": {"0": "Low Financial Capacity", "1": "Medium Financial Capacity", "2": "High Financial Capacity", 0: "Low Financial Capacity", 1: "Medium Financial Capacity", 2: "High Financial Capacity"},
        "credit_worthiness": {"0": "Poor Credit Worthiness (High Risk)", "1": "Fair Credit Worthiness", "2": "Excellent Credit Worthiness", 0: "Poor Credit Worthiness (High Risk)", 1: "Fair Credit Worthiness", 2: "Excellent Credit Worthiness"}
    }

    readable_profile = {
        "Raw Credit Score Category": human_labels["credit_score"].get(inputs.get("credit_score"), "Unknown"),
        "Raw Income Category": human_labels["annual_income"].get(inputs.get("annual_income"), "Unknown"),
        "Existing Debt Burden": human_labels["existing_loans_count"].get(inputs.get("existing_loans_count"), "Unknown"),
        "Family Dependents": human_labels["num_dependents"].get(inputs.get("num_dependents"), "Unknown"),
        "Employment Status": human_labels["employment_status"].get(inputs.get("employment_status"), "Unknown"),
        "Financial Capacity Assessment": human_labels["financial_capacity"].get(inputs.get("financial_capacity"), "Unknown"),
        "Credit Worthiness Evaluation": human_labels["credit_worthiness"].get(inputs.get("credit_worthiness"), "Unknown")
    }
    
    profile_json = json.dumps(readable_profile, indent=2)

    # 3. البرومبت الأساسي الموحد مع قانون منع الهلوسة والالتزام بالمسميات المصرفية
    system_base_instruction = f"""
You are an expert Credit Risk AI Advisor evaluating a Hierarchical Bayesian Network output for a bank loan officer.
Analyze how the underlying raw features drive the intermediate assessments (Financial Capacity Assessment & Credit Worthiness Evaluation) to logically trigger the final decision boundary.

STRICT OPERATIONAL LAWS:
1. PROFESSIONAL B2B TONE: Address the bank officer objectively. Never say "Your profile"; always refer to "The applicant's profile" or "The risk profile".
2. NO GREETINGS: Start answering directly. Do not say "Sure!", "Based on...", or "Hello".
3. CONCISE ANALYSIS: Limit your response to a maximum of 3-4 dense, high-impact sentences.
4. ZERO NUMERICAL FABRICATION: The ONLY allowed percentage figure in your output is the exact Final Model Approval Probability provided below ({prob:.2f}%). Never invent external numbers, weights, or metrics.
5. TERMINOLOGY: Never use technical labels like "Pillar 1" or "Pillar 2". Use strictly "Financial Capacity Assessment" and "Credit Worthiness Evaluation".
6. BN STRUCTURAL FIDELITY:
- Financial Capacity depends on annual_income, employment_status, and num_dependents.
- Credit Worthiness depends on credit_score and existing_loans_count.
- Both assessments feed the Loan Approval node.
Never mix the direct inputs of these assessments or invent unsupported relationships.
Approval probability is not repayment probability.
7. HUMAN REVIEW ADVISORY: Always preserve the exact BN decision (Approved/Rejected). If the approval probability is between 40% and 60% (inclusive), add a brief "Human Review Advisory" recommending further assessment by a credit officer. Never change the BN decision or replace it with "Borderline" or "Manual Review".

APPLICANT HIERARCHICAL RISK CONTEXT:
- Final Model Approval Probability: {prob:.2f}%
- Discovered Structural Network Profile:
{profile_json}
"""

    # 4. نوايا الاستعلام المطورة لدعم تحليل الاختناقات وعوامل التعويض
    if any(keyword in message_lower for keyword in ["why", "reason", "reject", "cause", "bottleneck"]):
        specific_prompt = (
            f"{system_base_instruction}\n"
            f"OFFICER QUESTION: {user_raw_message}\n"
            f"TASK: Provide a sharp causal justification for the approval probability ({prob:.2f}%). "
            f"Explain how the bottleneck or compensating trade-off between the intermediate assessments (Financial Capacity Assessment vs Credit Worthiness Evaluation) mathematically drives this output boundary based on historical network patterns."
        )
    elif any(keyword in message_lower for keyword in ["improve", "advice", "mitigat", "chance", "fix", "recourse"]):
        specific_prompt = (
            f"{system_base_instruction}\n"
            f"OFFICER QUESTION: {user_raw_message}\n"
            f"TASK: Provide qualitative risk-mitigation strategies under two distinct sections:\n"
            f"1. INTERNAL VARIABLE SHIFTING: Controllable client behaviors to improve the specific weak intermediate assessment.\n"
            f"2. LOAN STRUCTURING: Bank-side policy adjustments (e.g., collateral, adjusted credit lines) to absorb the risk.\n"
            f"CRITICAL: Do NOT fabricate or guess future percentage metrics or probability figures."
        )
    else:
        specific_prompt = (
            f"{system_base_instruction}\n"
            f"OFFICER QUESTION: {user_raw_message}\n"
            f"TASK: Answer the officer's technical question directly, maintaining absolute mathematical alignment with the Bayesian probabilities provided."
        )

    # 5. استدعاء المحرك المحلي
    try:
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": specific_prompt,
                "stream": False,
                "options": {
                    "num_thread": 4,
                    "num_predict": 250,
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "repeat_penalty": 1.2
                }
            },
            timeout=60
        )
        
        data = ollama_response.json()
        response = data.get("response", "Error processing request from local engine.").strip()

    except requests.exceptions.Timeout:
        response = "Analysis Timeout: The hierarchical risk inquiry exceeded acceptable execution limits under the current compute profile."
    except Exception as e:
        response = f"LLM Connection Error: {str(e)}"

    return {"response": response}