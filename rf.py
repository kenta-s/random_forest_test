from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sklearn.model_selection as sk
from IPython import embed

iris = load_iris()
x_train, x_test, y_train, y_test = sk.train_test_split(iris.data, iris.target, test_size=0.2)

clf = RandomForestClassifier(n_estimators=30, n_jobs=3)
clf.fit(x_train, y_train)

# embed()
# clf.predict(x_test)