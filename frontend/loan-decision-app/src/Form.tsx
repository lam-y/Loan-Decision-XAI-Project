import { useState } from "react";
import "./css/form.css"; 

type LoanInput = {
  creditScore: number | "";
  income: number | "";
  existingLoans: number | "";
  dependents: number | "";
  employment: "" | "yes" | "no";
};

type Props = {
  setMessages: any;
};

export default function Form({ setMessages }: Props) {
  const [form, setForm] = useState<LoanInput>({
    creditScore: "",
    income: "",
    existingLoans: "",
    dependents: "",
    employment: "",
  });

  const [errors, setErrors] = useState<any>({});
  const [loading, setLoading] = useState(false);

  const handleChange = (field: keyof LoanInput, value: any) => {
    setForm({ ...form, [field]: value });
    setErrors((prev: any) => ({ ...prev, [field]: "" }));
  };

  const evaluateLoan = async () => {
    if (loading) return; 

    if (!validate()) return;

    setLoading(true);

    setMessages((prev: any) => [
      ...prev,
      {
        sender: "user",
        text: `Loan evaluation request: Credit Score = ${form.creditScore}, Income = ${form.income}, Existing Loans = ${form.existingLoans}, Dependents = ${form.dependents}, Employment = ${form.employment}`,
      },
    ]);

    try {
      console.log("[Frontend] Submitting exclusive evaluation request..."); 

      const res = await fetch("http://127.0.0.1:8000/evaluate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ...form,
          employment: form.employment === "yes" ? 1 : 0,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        let backendErrorMessage = "Invalid input values detected.";
        
        if (res.status === 422 && data.detail) {
          backendErrorMessage = Array.isArray(data.detail) 
            ? data.detail[0].msg 
            : data.detail;
        } else if (data.llm_explanation) {
          backendErrorMessage = data.llm_explanation;
        }

        setMessages((prev: any) => [
          ...prev,
          {
            sender: "system",
            text: `Input Validation Error:\n${backendErrorMessage}`,
          },
        ]);
        return; 
      }

      // في حال النجاح المستقر
      setMessages((prev: any) => [
        ...prev,
        {
          sender: "system",
          text: `${data.decision} - ${data.probability}%\n\n📊 Explanation:\n${data.llm_explanation}`,
        },
      ]);

      // تفريغ الحقول عند النجاح التام
      setForm({
        creditScore: "",
        income: "",
        existingLoans: "",
        dependents: "",
        employment: "",
      });
      setErrors({});

    } catch (error) {
      setMessages((prev: any) => [
        ...prev,
        {
          sender: "system",
          text: "Server connection error. Please check if backend is running.",
        },
      ]);
    } finally {
      setLoading(false); 
    }
  };

  const validate = () => {
    let newErrors: any = {};

    if (form.creditScore === "") newErrors.creditScore = "Required";
    if (form.income === "") newErrors.income = "Required";
    if (form.existingLoans === "") newErrors.existingLoans = "Required";
    if (form.dependents === "") newErrors.dependents = "Required";
    if (form.employment === "") newErrors.employment = "Required";

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  return (
    <div className={`form-wrapper ${loading ? "disabled" : ""}`}>
      <h2>Loan Decision Support System</h2>
      <p style={{ textAlign: "center" }}>
        Enter your data to evaluate loan approval
      </p>

      <div className="card">
        <input
          type="number"
          placeholder="Credit Score (300–850)"
          value={form.creditScore}
          disabled={loading}
          onChange={(e) =>
            handleChange(
              "creditScore",
              e.target.value === "" ? "" : Number(e.target.value)
            )
          }
        />
        {errors.creditScore && <span className="error">{errors.creditScore}</span>}

        <input
          type="number"
          placeholder="Annual Income"
          value={form.income}
          disabled={loading}
          onChange={(e) =>
            handleChange(
              "income",
              e.target.value === "" ? "" : Number(e.target.value)
            )
          }
        />
        {errors.income && <span className="error">{errors.income}</span>}

        <input
          type="number"
          placeholder="Existing Loans"
          value={form.existingLoans}
          disabled={loading}
          onChange={(e) =>
            handleChange(
              "existingLoans",
              e.target.value === "" ? "" : Number(e.target.value)
            )
          }
        />
        {errors.existingLoans && <span className="error">{errors.existingLoans}</span>}

        <input
          type="number"
          placeholder="Number of Dependents"
          value={form.dependents}
          disabled={loading}
          onChange={(e) =>
            handleChange(
              "dependents",
              e.target.value === "" ? "" : Number(e.target.value)
            )
          }
        />
        {errors.dependents && <span className="error">{errors.dependents}</span>}

        <select
          onChange={(e) => handleChange("employment", e.target.value)}
          value={form.employment}
          disabled={loading}
          className={form.employment === "" ? "placeholder" : ""}
        >
          <option value="" disabled hidden>Employment Status</option>
          <option value="no">No</option>
          <option value="yes">Yes</option>
        </select>
        {errors.employment && <span className="error">{errors.employment}</span>}

        <button onClick={evaluateLoan} disabled={loading}>
          {loading ? "Processing..." : "Evaluate"}
        </button>

        {loading && (
          <div className="form-loading-status" style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginTop: "15px",
            padding: "10px",
            backgroundColor: "#f0f4f8",
            borderRadius: "4px",
            borderLeft: "3px solid #0066cc"
          }}>
            <div className="banking-spinner"></div>
            <span style={{ fontSize: "12px", color: "#888", marginLeft: "8px" }}>
              Running Bayesian Network & AI Interpretation...
            </span>
          </div>
        )}
      </div>
    </div>
  );
}