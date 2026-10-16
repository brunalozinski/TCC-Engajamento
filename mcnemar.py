"""
Teste de McNemar par a par entre os três modelos, usando os dados de teste
salvos pelo treinar.py.
"""

import numpy as np
import joblib
from mlxtend.evaluate import mcnemar_table, mcnemar

from features import RAIZ

X_test = np.load(RAIZ / "X_test.npy")
y_test = np.load(RAIZ / "y_test.npy")

svm = joblib.load(RAIZ / "model_SVM.pkl")
rf = joblib.load(RAIZ / "model_Random_Forest.pkl")
mlp = joblib.load(RAIZ / "model_MLP.pkl")

y_svm = svm.predict(X_test)
y_rf = rf.predict(X_test)
y_mlp = mlp.predict(X_test)

pares = [("SVM", y_svm, "Random Forest", y_rf),
         ("SVM", y_svm, "MLP", y_mlp),
         ("Random Forest", y_rf, "MLP", y_mlp)]

for nome_a, pred_a, nome_b, pred_b in pares:
    tabela = mcnemar_table(y_test, pred_a, pred_b)
    chi2, p = mcnemar(tabela, corrected=True)
    print(f"\n{nome_a} vs {nome_b}")
    print(f"  chi2 = {chi2:.4f}")
    print(f"  p    = {p:.4f}")
    if p < 0.05:
        print("  Resultado: diferença SIGNIFICATIVA (p < 0.05)")
    else:
        print("  Resultado: diferença NÃO significativa (p >= 0.05)")
