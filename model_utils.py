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

        self.model = model()
        self.single_class = True

    def predict(self, state, action):

        if self.single_class:
            return 0
        else:
            x = np.array([[state, action]], dtype=np.float32)
            return self.model.predict(x)[0]

    def fit(self, state_action_partition):

        x = []
        y = []

        for idx, block in enumerate(state_action_partition):

            for state, action, _, _, _ in block:
                x.append([state, action])
                y.append(idx)

        x = np.array(x, dtype=np.float32)

        y = np.array(y, dtype=np.int32)

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)

        if np.all(y == 0):
            self.single_class = True
        else:
            self.single_class = False
            self.model.fit(x, y)
