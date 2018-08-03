import numpy as np
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegression:

    def __init__(self):

        self.logistic_regression = LogisticRegression()
        self.single_class = True

    def predict(self, state):

        if self.single_class:
            return 0
        else:
            return self.logistic_regression.predict(state)[0]

    def fit(self, x, y):

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)

        if np.all(y == 0):
            self.single_class = True
        else:
            self.single_class = False
            self.logistic_regression.fit(x, y)