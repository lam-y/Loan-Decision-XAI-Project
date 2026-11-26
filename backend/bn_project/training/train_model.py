# from bn_model import BNRealLoanModel
from .bn_model_old import BNRealLoanModel

model = BNRealLoanModel("data/loanapproval.csv")
model.save_model("training/loan_bn_model.pkl")