import numpy as np


class GModel:

    def __init__(self, model):

        self.model = model
        self.single_class = True

    def predict(self, state, action):

        if self.single_class:
            return 0
        else:
            x = np.array([[state, action]], dtype=np.float32)
            return self.model.predict(x)[0]

    def batch_predict(self, states, actions):

        if self.single_class:
            return [0] * len(states)
        else:
            x = np.stack([states, actions], axis=-1)
            return self.model.predict(x)

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
