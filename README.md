# Loan Decision XAI Project

**Loan Decision XAI Project** is a conversational, explainable AI decision-support web application for credit risk and loan decision analysis. It combines a **Hierarchical Bayesian Network** (`pgmpy`) in the Python/FastAPI backend for probabilistic reasoning, sensitivity analysis, and counterfactual "what-if" exploration with a locally hosted **Llama 3.2** model (via **Ollama**) to generate clear, causally grounded explanations, paired with an interactive **React.js** chat interface.

> **Note on Local LLM & Dependencies:** To run the application properly, you need to install **Ollama** and pull the **Llama 3.2** model locally (e.g., `ollama pull llama3.2`), and install the required Python dependencies from `requirements.txt` (`pip install -r requirements.txt`).


## 🎥 Demo
[Watch the Loan Decision app Demo](https://drive.google.com/drive/folders/1-eLXemoOhaFB58pyMEDcRDFj358nq4VU?usp=drive_link)

## 📌 Setup & Running the Application

### 1. Local LLM Setup (Ollama & Llama 3.2)
Install **Ollama** and pull the **Llama 3.2** model so local natural language explanations can be generated. Make sure the Ollama service is running before sending explanation requests.
```bash
ollama pull llama3.2
```

### 2. Environment & Dependencies Setup
Navigate to your project or backend directory.

```bash 
cd path/to/your/project
```

Create and activate a virtual environment named `XAI_BayesianNetwork` (or activate an existing Conda environment).

* Using Conda (if available): `conda activate XAI_BayesianNetwork`
* Or using Python venv (Windows): `python -m venv XAI_BayesianNetwork` then `XAI_BayesianNetwork\Scripts\activate`
* Or using Python venv (Linux/macOS): `python -m venv XAI_BayesianNetwork` then `source XAI_BayesianNetwork/bin/activate`

Upgrade `pip` and install the required Python packages from `requirements.txt`. Note that some packages such as `torch` and `transformers` are large and may take some time to download.

```bash 
python -m pip install --upgrade pip` then `pip install -r requirements.txt
```

### 3. Run the Backend (FastAPI)
Start the FastAPI server from the directory containing `main.py`.

```bash 
uvicorn main:app --reload
```

### 4. Run the Frontend (React.js)
In a separate terminal, navigate to the frontend directory, install dependencies, and start the React application (use `npm run dev` instead if your setup uses Vite).

```bash 
cd path/to/frontend` then `npm install` then `npm start
```

