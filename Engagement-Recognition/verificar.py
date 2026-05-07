import numpy as np

X = np.load("C:/TCC/Engagement-Recognition/preprocess_data/X_train.npy")
y = np.load("C:/TCC/Engagement-Recognition/preprocess_data/Y_train.npy")

print("Shape X:", X.shape)
print("Shape y:", y.shape)
print("Classes:", np.unique(y))
print("Exemplos por classe:")
for c in np.unique(y):
    print(f"  Classe {c}: {np.sum(y == c)} amostras")