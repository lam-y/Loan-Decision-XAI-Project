from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class LoanModel:
    def __init__(self):
        # تعريف العلاقات
        self.model = DiscreteBayesianNetwork([
            ('Income', 'Loan'),
            ('CreditScore', 'Loan'),
            ('Debt', 'Loan')
        ])

        # تعريف الاحتمالات

        cpd_income = TabularCPD(
            variable='Income',
            variable_card=2,
            values=[[0.6], [0.4]]  # low, high
        )

        cpd_credit = TabularCPD(
            variable='CreditScore',
            variable_card=2,
            values=[[0.7], [0.3]]  # bad, good
        )

        cpd_debt = TabularCPD(
            variable='Debt',
            variable_card=2,
            values=[[0.5], [0.5]]  # high, low
        )

        # Loan يعتمد على الثلاثة
        cpd_loan = TabularCPD(
            variable='Loan',
            variable_card=2,
            values=[
                # رفض
                [0.9, 0.7, 0.8, 0.5, 0.6, 0.4, 0.3, 0.1],
                # قبول
                [0.1, 0.3, 0.2, 0.5, 0.4, 0.6, 0.7, 0.9],
            ],
            evidence=['Income', 'CreditScore', 'Debt'],
            evidence_card=[2, 2, 2]
        )

        self.model.add_cpds(cpd_income, cpd_credit, cpd_debt, cpd_loan)
        self.model.check_model()

        self.infer = VariableElimination(self.model)

    def get_baseline(self):
        result = self.infer.query(variables=['Loan'])
        return float(result.values[1])

    def predict(self, income, credit_score, debt):
        result = self.infer.query(
            variables=['Loan'],
            evidence={
                'Income': income,
                'CreditScore': credit_score,
                'Debt': debt
            }
        )

        prob_approve = float(result.values[1])

        impacts = self.analyze_impact(income, credit_score, debt)

        return {
            "approval_probability": prob_approve,
            "impacts": impacts
        }


    def analyze_impact(self, income, credit_score, debt):

        baseline = self.get_baseline()

        impacts = {}

        # تأثير الدخل
        res_income = self.infer.query(
            variables=['Loan'],
            evidence={'Income': income}
        )
        impacts['Income'] = float(res_income.values[1]) - baseline

        # تأثير السجل الائتماني
        res_credit = self.infer.query(
            variables=['Loan'],
            evidence={'CreditScore': credit_score}
        )
        impacts['CreditScore'] = float(res_credit.values[1]) - baseline

        # تأثير الدين
        res_debt = self.infer.query(
            variables=['Loan'],
            evidence={'Debt': debt}
        )
        impacts['Debt'] = float(res_debt.values[1]) - baseline

        return impacts