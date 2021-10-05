
from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7] #gercek değerler(maaslar)
y_pred = [2.5, 0.0, 2, 8] #tahmin degeler(Tahmini maaslar)
print(r2_score(y_true, y_pred))