import time
from pytabkit import MLP_SKL_D_Classifier
import numpy as np
print("pytabkit import OK, time:", time.time())

X = np.random.randn(360, 10).astype(np.float32)
y = np.random.randint(0, 2, 360)
print("Data created")

m = MLP_SKL_D_Classifier()
print("Model created")
m.fit(X, y)
print("Fitted")

oof = m.predict_proba(X[:50])[:, 1]
print("Predicted:", oof[:5])
print("Done!")
