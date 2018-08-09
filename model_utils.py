import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


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


class BalancedMLP:

    def __init__(self, state_shape, hiddens, learning_rate, batch_size, weight_decay, validation_fraction=0.1,
                 momentum=0.9, balanced_sampling=True, verbose=False):

        self.state_shape = state_shape
        self.hiddens = hiddens
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.momentum = momentum
        self.balanced_sampling = balanced_sampling
        self.verbose = verbose

        self.single_class = True
        self.session = None

    def predict(self, state, action):

        if self.single_class:
            return 0
        else:
            state = np.expand_dims(state[0], axis=0)
            action = np.expand_dims([action], axis=0)

            logits = self.session.run(self.logits, feed_dict={
                self.states_pl: state,
                self.actions_pl: action
            })
            logits = logits[0]
            return np.argmax(logits)

    def fit(self, state_action_partition):

        states = []
        actions = []
        blocks = []

        for idx, block in enumerate(state_action_partition):

            for transition in block:
                states.append(transition[0][0])
                actions.append(transition[1])
                blocks.append(idx)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        blocks = np.array(blocks, dtype=np.int32)

        if np.all(blocks == 0):
            self.single_class = True
        else:
            self.single_class = False
            self.__retrain(states, actions, blocks)

    def __retrain(self, states, actions, blocks):

        num_classes = np.max(blocks) + 1

        # free memory
        if self.session is not None:
            self.__stop_session()

        # reset tf graph
        tf.reset_default_graph()

        # build placeholders
        self.__build_placeholders()

        # build network
        self.__build_network(num_classes)
        self.__build_training()

        # start session
        self.__start_session()

        # train network
        self.__train_network(states, actions, blocks)

    def __build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, np.product(self.state_shape)), name="states_pl")
        self.actions_pl = tf.placeholder(tf.float32, shape=(None, 1), name="actions_pl")
        self.labels_pl = tf.placeholder(tf.int32, shape=(None,), name="labels_pl")

    def __build_network(self, num_classes):

        self.num_classes = num_classes

        x = tf.concat([self.states_pl, self.actions_pl], axis=1, name="concat_input")

        for idx in range(len(self.hiddens)):
            with tf.variable_scope("hidden{:d}".format(idx)):
                x = tf.layers.dense(x, self.hiddens[idx], activation=tf.nn.relu,
                                    kernel_initializer=get_mrsa_initializer(),
                                    kernel_regularizer=get_weight_regularizer(self.weight_decay))

        with tf.variable_scope("logits"):
            self.logits = tf.layers.dense(x, num_classes, kernel_initializer=get_mrsa_initializer(),
                                          kernel_regularizer=get_weight_regularizer(self.weight_decay))

    def __build_training(self):

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_pl, logits=self.logits)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels_pl, tf.cast(tf.argmax(self.logits, axis=1),
                                                                                tf.int32)), tf.float32))
        self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss)

    def __train_network(self, states, actions, blocks, max_training_steps=2000, validation_frequency=50):

        # shuffle and preprocess dataset
        states, actions, blocks = shuffle(states, actions, blocks)
        actions = np.expand_dims(actions, -1)

        # split dataset
        num_samples = len(states)
        num_valid_samples = int(num_samples * self.validation_fraction)
        valid_states = states[:num_valid_samples]
        valid_actions = actions[:num_valid_samples]
        valid_blocks = blocks[:num_valid_samples]
        train_states = states[num_valid_samples:]
        train_actions = actions[num_valid_samples:]
        train_blocks = blocks[num_valid_samples:]

        # parameter search run
        best_step = None
        best_accuracy = None

        if self.balanced_sampling:
            epoch_states, epoch_actions, epoch_blocks = self.__balanced_sampling(train_states, train_actions,
                                                                                 train_blocks)
        else:
            epoch_states, epoch_actions, epoch_blocks = train_states, train_actions, train_blocks

        num_steps_per_epoch = len(epoch_states) // self.batch_size

        for step_idx in range(max_training_steps):

            epoch_step_idx = step_idx % num_steps_per_epoch

            if step_idx > 0 and epoch_step_idx == 0:
                # new epoch, reshuffle dataset
                if self.balanced_sampling:
                    epoch_states, epoch_actions, epoch_blocks = self.__balanced_sampling(train_states, train_actions,
                                                                                         train_blocks)
                else:
                    epoch_states, epoch_actions, epoch_blocks = shuffle(train_states, train_actions, train_blocks)

            feed_dict = {
                self.states_pl: epoch_states[epoch_step_idx * self.batch_size:(epoch_step_idx + 1) * self.batch_size],
                self.actions_pl: epoch_actions[epoch_step_idx * self.batch_size:
                                               (epoch_step_idx + 1) * self.batch_size],
                self.labels_pl: epoch_blocks[epoch_step_idx * self.batch_size:(epoch_step_idx + 1) * self.batch_size]
            }

            self.session.run(self.train_step, feed_dict=feed_dict)

            if step_idx > 0 and step_idx % validation_frequency:

                accuracy = self.session.run(self.accuracy, feed_dict={
                    self.states_pl: valid_states,
                    self.actions_pl: valid_actions,
                    self.labels_pl: valid_blocks
                })

                if best_accuracy is None or accuracy > best_accuracy:

                    best_step = step_idx
                    best_accuracy = accuracy

                    if self.verbose:

                        final_accuracy, logits = self.session.run([self.accuracy, self.logits], feed_dict={
                            self.states_pl: valid_states,
                            self.actions_pl: valid_actions,
                            self.labels_pl: valid_blocks
                        })

                        print("step {:d}".format(best_step))

                        for class_idx in range(self.num_classes):
                            class_accuracy = np.mean(
                                (np.argmax(logits[valid_blocks == class_idx], axis=1) == class_idx).astype(np.float32))
                            print("class {:d} accuracy: {:.2f}%, num samples: {:d}".format(class_idx + 1,
                                                                                           class_accuracy * 100,
                                                                                           np.sum(blocks == class_idx)))

                        print("best accuracy: {:.2f}%".format(final_accuracy * 100))

        # final run
        self.session.run(tf.global_variables_initializer())

        if self.balanced_sampling:
            epoch_states, epoch_actions, epoch_blocks = self.__balanced_sampling(states, actions, blocks)
        else:
            epoch_states, epoch_actions, epoch_blocks = states, actions, blocks

        num_steps_per_epoch = len(epoch_states) // self.batch_size

        for step_idx in range(best_step + 1):

            epoch_step_idx = step_idx % num_steps_per_epoch

            if step_idx > 0 and epoch_step_idx == 0:
                # new epoch, reshuffle dataset
                if self.balanced_sampling:
                    epoch_states, epoch_actions, epoch_blocks = self.__balanced_sampling(states, actions, blocks)
                else:
                    epoch_states, epoch_actions, epoch_blocks = shuffle(states, actions, blocks)

            feed_dict = {
                self.states_pl: epoch_states[epoch_step_idx * self.batch_size: (epoch_step_idx + 1) * self.batch_size],
                self.actions_pl: epoch_actions[epoch_step_idx * self.batch_size:
                                               (epoch_step_idx + 1) * self.batch_size],
                self.labels_pl: epoch_blocks[epoch_step_idx * self.batch_size: (epoch_step_idx + 1) * self.batch_size]
            }

            self.session.run(self.train_step, feed_dict=feed_dict)

    def __balanced_sampling(self, states, actions, blocks):

        min_samples = None

        for class_idx in range(self.num_classes):

            num_samples = np.sum(blocks == class_idx)
            if min_samples is None or num_samples < min_samples:
                min_samples = num_samples

        if min_samples < self.batch_size:
            raise ValueError("Minimum number of samples is smaller than batch size.")

        states, actions, blocks = shuffle(states, actions, blocks)

        sampled_states = []
        sampled_actions = []
        sampled_blocks = []

        for class_idx in range(self.num_classes):

            mask = blocks == class_idx

            sampled_states.append(states[mask][:min_samples])
            sampled_actions.append(actions[mask][:min_samples])
            sampled_blocks.append(blocks[mask][:min_samples])

        sampled_states = np.concatenate(sampled_states, axis=0)
        sampled_actions = np.concatenate(sampled_actions, axis=0)
        sampled_blocks = np.concatenate(sampled_blocks, axis=0)

        return shuffle(sampled_states, sampled_actions, sampled_blocks)

    def __start_session(self):

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def __stop_session(self):

        self.session.close()


def get_mrsa_initializer():
    """
    Get MRSA (Microsoft Research Asia) ConvNet weights initializer.
    :return:      The initializer object.
    """

    return tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN", uniform=False)


def get_weight_regularizer(weight_decay):
    """
    Get L2 weight regularizer.
    :param weight_decay:    Weight decay.
    :return:                Regularizer object.
    """
    return tf.contrib.layers.l2_regularizer(weight_decay)
