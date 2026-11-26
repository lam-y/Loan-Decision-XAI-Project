# main.py
from fastapi import FastAPI
import joblib
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import requests


sys.path.append(os.path.join(os.path.dirname(__file__), "training"))
try:
    from bn_model_hybrid import BNModelBuilder
    sys.modules['__main__'].BNModelBuilder = BNModelBuilder
    print("Successfully mapped BNModelBuilder to __main__ context.")
except ImportError:
    print(" Warning: Could not find bn_model_hybrid in training folder, relying on fallback configurations.")
    pass


from api.chat import router as chat_router, set_last_evaluation
from api.evaluate import create_evaluate_router 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    model = joblib.load("loan_bn_model_hybrid.pkl")
    print("Hybrid Bayesian Network Model (.pkl) loaded successfully into memory!")
except Exception as e:
    print(f"Critical Error loading pkl file: {e}")
    model = None

def ask_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_thread": 4,      
                    "num_predict": 150,   
                    "temperature": 0.15,  
                    "top_p": 0.9
                }
            },
            timeout=60 
        )

        data = response.json()

        if "response" in data:
            return data["response"]
        else:
            return f"LLM Error: {data}"

    except requests.exceptions.Timeout:
        return "Analysis Timeout: The risk analysis exceeded acceptable execution time limits."
    except Exception as e:
        return f"LLM connection error: {str(e)}"


if model is not None:
    evaluate_router = create_evaluate_router(model, ask_llm, set_last_evaluation)
    app.include_router(evaluate_router)

app.include_router(chat_router)