import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
import os


class RealLoanModel:
    def __init__(self):

         # ---------------------------
        # 1. مسار ديناميكي
        # ---------------------------
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(BASE_DIR, "data", "loan_data.csv")

        data = pd.read_csv(data_path)

        # ---------------------------
        # 2. تنظيف أسماء الأعمدة
        # ---------------------------
        data.columns = data.columns.str.strip()

        # ---------------------------
        # 3. حذف الأعمدة غير المهمة
        # ---------------------------
        data = data.drop(columns=["loan_id"])

        # ---------------------------
        # 2. تحويل القيم من مستمرة الى ثنائية
        # ---------------------------
        data['Loan'] = data['loan_status'].map({
            ' Approved': 1,
            ' Rejected': 0
        })

        # qcut يقسم البيانات إلى quantiles متساوية → لن ينتج NaN إلا إذا القيم متكررة جدًا
        data['Income'] = pd.qcut(data['income_annum'], q=2, labels=[0,1])

        data['CreditScore'] = pd.qcut(data['cibil_score'], q=2, labels=[0,1])

        data['Debt'] = pd.qcut(data['loan_amount'], q=2, labels=[1,0])  # low debt=1, high debt=0

        # ---------------------------
        # 6. إزالة أي صفوف بها NaN
        # ---------------------------
        data = data[['Income','CreditScore','Debt','Loan']]
        data = data.dropna()

        # ---------------------------
        # 7. تحويل جميع الأعمدة إلى int
        # ---------------------------
        data = data.astype(int)

        # ---------------------------
        # 3. بناء النموذج
        # ---------------------------
        self.model = DiscreteBayesianNetwork([
            ('Income', 'Loan'),
            ('CreditScore', 'Loan'),
            ('Debt', 'Loan')
        ])

        # تدريب النموذج
        self.model.fit(data, estimator=MaximumLikelihoodEstimator)

        # inference
        self.infer = VariableElimination(self.model)

    # ---------------------------
    # baseline
    # ---------------------------
    def get_baseline(self):
        result = self.infer.query(variables=['Loan'])
        return float(result.values[1])

    # ---------------------------
    # تحليل التأثير
    # ---------------------------
    def analyze_impact(self, income, credit_score, debt):
        baseline = self.get_baseline()

        impacts = {}

        res_income = self.infer.query(
            variables=['Loan'],
            evidence={'Income': income}
        )
        impacts['Income'] = float(res_income.values[1]) - baseline

        res_credit = self.infer.query(
            variables=['Loan'],
            evidence={'CreditScore': credit_score}
        )
        impacts['CreditScore'] = float(res_credit.values[1]) - baseline

        res_debt = self.infer.query(
            variables=['Loan'],
            evidence={'Debt': debt}
        )
        impacts['Debt'] = float(res_debt.values[1]) - baseline

        return impacts

    # ---------------------------
    # التنبؤ
    # ---------------------------
    def predict(self, income, credit_score, debt):

        result = self.infer.query(
            variables=['Loan'],
            evidence={
                'Income': income,
                'CreditScore': credit_score,
                'Debt': debt
            }
        )

        prob = float(result.values[1])
        impacts = self.analyze_impact(income, credit_score, debt)

        return {
            "approval_probability": prob,
            "impacts": impacts
        }