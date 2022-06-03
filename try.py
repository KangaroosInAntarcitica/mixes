from sklearn.decomposition import FactorAnalysis
import numpy as np

fa = FactorAnalysis(n_components=3, rotation='varimax')

data = np.random.random([1000, 8])
d1 = fa.fit_transform(data)
fa.ana
print("res = ", np.mean(np.abs(data - fa.mean_ - d1 @ fa.components_)))
