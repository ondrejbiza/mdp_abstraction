import numpy as np


class FModel:

    def __init__(self, model):

        self.logistic_regression = model()
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


class GModel:

    def __init__(self, model):

        self.logistic_regression = model()
        self.single_class = True

    def predict(self, state, action):

        if self.single_class:
            return 0
        else:
            x = np.array([[state, action]], dtype=np.float32)
            return self.logistic_regression.predict(x)[0]

    def fit(self, x, y):

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)

        if np.all(y == 0):
            self.single_class = True
        else:
            self.single_class = False
            self.logistic_regression.fit(x, y)
