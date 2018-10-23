import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


class GModel:

    def __init__(self, model):
        """
        Wrapper around a sklearn model for estimating g.
        :param model:       Sklearn model.
        """

        self.model = model
        self.single_class = True

    def predict(self, state, action):
        """
        Predict into which block the given state-action pair belongs.
        :param state:       State.
        :param action:      Action.
        :return:            Index of a state-action block.
        """

        if self.single_class:
            return 0
        else:
            x = np.array([[state, action]], dtype=np.float32)
            return self.model.predict(x)[0]

    def batch_predict(self, states, actions):
        """
        Predict into which block a batch of state-action pairs belong.
        :param states:      Batch of states.
        :param actions:     Batch of actions.
        :return:            List of indices of state-action blocks.
        """

        if self.single_class:
            return [0] * len(states)
        else:
            x = np.stack([states, actions], axis=-1)
            return self.model.predict(x)

    def fit(self, state_action_partition):
        """
        Train the model to recognize given state-action blocks.
        :param state_action_partition:          State-action partition.
        :return:                                None.
        """

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

    def __init__(self, state_shape, hiddens, learning_rate, batch_size, weight_decay, validation_fraction=0.2,
                 momentum=0.9, oversample=True, verbose=False):
        """
        A Tensorflow neural network with dataset balancing.
        :param state_shape:                 Shape of the state.
        :param hiddens:                     List of neuron counts for hidden layers.
        :param learning_rate:               Learning rate.
        :param batch_size:                  Batch size.
        :param weight_decay:                Weight decay, applied to all parameterized layers.
        :param validation_fraction:         Fraction of data to use for validation.
        :param momentum:                    Momentum for SGD optimizer.
        :param oversample:                  Oversample the minority classes.
        :param verbose:                     Print additional information.
        """

        self.state_shape = state_shape
        self.hiddens = hiddens
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.momentum = momentum
        self.oversample = oversample
        self.verbose = verbose

        self.single_class = True
        self.session = None

    def predict(self, state, action):
        """
        Predict into which block the given state-action pair belongs.
        :param state:       State.
        :param action:      Action.
        :return:            Index of a state-action block.
        """

        if self.single_class:
            return 0
        else:
            logits = self.session.run(self.logits, feed_dict={
                self.states_pl: [state],
                self.actions_pl: [action]
            })
            logits = logits[0]
            return np.argmax(logits)

    def predict_prob(self, state, action):
        """
        Predict probabilities of state-action blocks given a state-action pair.
        :param state:       State.
        :param action:      Action.
        :return:            Probability distribution over all state-action blocks.
        """

        if self.single_class:
            return [1]
        else:
            predictions = self.session.run(self.predictions, feed_dict={
                self.states_pl: [state],
                self.actions_pl: [action]
            })
            return predictions

    def batch_predict(self, states, actions):
        """
        Predict into which block a batch of state-action pairs belong.
        :param states:      Batch of states.
        :param actions:     Batch of actions.
        :return:            List of indices of state-action blocks.
        """

        if self.single_class:
            return [0] * len(states)
        else:
            logits = self.session.run(self.logits, feed_dict={
                self.states_pl: states,
                self.actions_pl: actions
            })
            return np.argmax(logits, axis=1)

    def batch_predict_prob(self, states, actions):
        """
        Predict probabilities of state-action blocks for a batch of state-action pairs.
        :param states:      States.
        :param actions:     Actions.
        :return:            Batch of state-action block distributions.
        """

        if self.single_class:
            return [1] * len(states)
        else:
            predictions = self.session.run(self.predictions, feed_dict={
                self.states_pl: states,
                self.actions_pl: actions
            })
            return predictions

    def fit(self, state_action_partition):
        """
        Train the model to recognize given state-action blocks.
        :param state_action_partition:          State-action partition.
        :return:                                None.
        """

        states = []
        actions = []
        blocks = []

        for idx, block in enumerate(state_action_partition):

            for transition in block:
                states.append(transition[0])
                actions.append(transition[1])
                blocks.append(idx)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        blocks = np.array(blocks, dtype=np.int32)

        if np.all(blocks == 0):
            self.single_class = True
        else:
            self.single_class = False
            return self.__retrain(states, actions, blocks)

    def __retrain(self, states, actions, blocks):
        """
        Build a new neural network and train it.
        :param states:      List of states.
        :param actions:     List of actions.
        :param blocks:      Indexes of state-action blocks the state-action pairs transition to.
        :return:            None.
        """

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
        return self.__train_network(states, actions, blocks)

    def __build_placeholders(self):
        """
        Build Tensorflow placeholders.
        :return:        None.
        """

        self.states_pl = tf.placeholder(tf.float32, shape=(None,), name="states_pl")
        self.actions_pl = tf.placeholder(tf.float32, shape=(None,), name="actions_pl")
        self.labels_pl = tf.placeholder(tf.int32, shape=(None,), name="labels_pl")

    def __build_network(self, num_classes):
        """
        Build the neural network.
        :param num_classes:     Number of classes to predict.
        :return:                None.
        """

        self.num_classes = num_classes

        x = tf.stack([self.states_pl, self.actions_pl], axis=1, name="stack_input")

        for idx in range(len(self.hiddens)):
            with tf.variable_scope("hidden{:d}".format(idx + 1)):
                x = tf.layers.dense(x, self.hiddens[idx], activation=tf.nn.relu,
                                    kernel_initializer=get_mrsa_initializer(),
                                    kernel_regularizer=get_weight_regularizer(self.weight_decay))

        with tf.variable_scope("logits"):
            self.logits = tf.layers.dense(x, num_classes, kernel_initializer=get_mrsa_initializer(),
                                          kernel_regularizer=get_weight_regularizer(self.weight_decay))
            self.predictions = tf.nn.softmax(self.logits, axis=-1)

    def __build_training(self):
        """
        Build Tensorflow training operation.
        :return:        None.
        """

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_pl, logits=self.logits)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels_pl, tf.cast(tf.argmax(self.logits, axis=1),
                                                                                tf.int32)), tf.float32))
        self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss)

    def __train_network(self, states, actions, blocks, max_training_steps=2000, validation_frequency=50):
        """
        Train the neural network.
        :param states:                  List of states.
        :param actions:                 List of actions.
        :param blocks:                  Indexes of state-action blocks the state-action pairs transition to.
        :param max_training_steps:      Maximum number of training steps.
        :param validation_frequency:    How often to run validation.
        :return:                        None.
        """

        # create training and validation splits
        indices = np.array(list(range(len(states))), dtype=np.int32)

        train_x, train_labels, valid_x, valid_labels = stratified_split(
            indices, blocks, self.validation_fraction
        )

        train_states = states[train_x]
        train_actions = actions[train_x]
        valid_states = states[valid_x]
        valid_actions = actions[valid_x]

        assert train_states.shape[0] == train_actions.shape[0] == train_labels.shape[0]
        assert valid_states.shape[0] == valid_actions.shape[0] == valid_labels.shape[0]

        # maybe balance the dataset
        if self.oversample:
            indices = np.array(list(range(len(train_states))), dtype=np.int32)

            train_x, train_labels = oversample(indices, train_labels, min_samples=self.batch_size)
            train_states = train_states[train_x]
            train_actions = train_actions[train_x]

            assert train_states.shape[0] == train_actions.shape[0] == train_labels.shape[0]

        # parameter search run
        best_accuracy = None
        best_per_class_accuracy = None
        best_parameters = None

        num_steps_per_epoch = train_states.shape[0] // self.batch_size

        for step_idx in range(max_training_steps):

            epoch_step_idx = step_idx % num_steps_per_epoch

            if step_idx > 0 and epoch_step_idx == 0:
                # new epoch, reshuffle dataset
                train_states, train_actions, train_labels = shuffle(
                    train_states, train_actions, train_labels
                )

            feed_dict = {
                self.states_pl: train_states[epoch_step_idx * self.batch_size:(epoch_step_idx + 1) * self.batch_size],
                self.actions_pl: train_actions[epoch_step_idx * self.batch_size:
                                               (epoch_step_idx + 1) * self.batch_size],
                self.labels_pl: train_labels[epoch_step_idx * self.batch_size:(epoch_step_idx + 1) * self.batch_size]
            }

            self.session.run(self.train_step, feed_dict=feed_dict)

            if step_idx > 0 and step_idx % validation_frequency:

                unbalanced_accuracy, logits = self.session.run([self.accuracy, self.logits], feed_dict={
                    self.states_pl: valid_states,
                    self.actions_pl: valid_actions,
                    self.labels_pl: valid_labels
                })

                class_accuracies = []
                for class_idx in range(self.num_classes):
                    class_accuracy = np.mean(
                        (np.argmax(logits[valid_labels == class_idx], axis=1) == class_idx).astype(np.float32))
                    class_accuracies.append(class_accuracy)
                balanced_accuracy = np.mean(class_accuracies)

                if best_accuracy is None or balanced_accuracy > best_accuracy:

                    best_accuracy = balanced_accuracy
                    best_per_class_accuracy = class_accuracies
                    best_parameters = self.session.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

                    if self.verbose:

                        print("step {:d}".format(step_idx))
                        print("unbalanced accuracy: {:.2f}%".format(unbalanced_accuracy * 100))
                        print("balanced accuracy: {:.2f}%".format(balanced_accuracy * 100))

                        for class_idx in range(self.num_classes):
                            print("class {:d} accuracy: {:.2f}%, num train samples: {:d}, num valid samples: {:d}"
                                  .format(class_idx + 1, class_accuracies[class_idx] * 100,
                                          np.sum(train_labels == class_idx), np.sum(valid_labels == class_idx))
                                  )

        # set the best parameters
        for idx, variable in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
            self.session.run(variable.assign(best_parameters[idx]))

        return best_per_class_accuracy

    def __start_session(self):
        """
        Start Tensorflow session and initialize all variables.
        :return:
        """

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def __stop_session(self):
        """
        Stop Tensorflow session.
        :return:
        """

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


def get_min_and_max_samples_per_class(y):
    """
    Get number of samples for the class with the least and the most samples.
    :param y:       Class indexes for each training sample.
    :return:        Minimum and maximum number of samples.
    """

    min_samples = None
    max_samples = None
    num_classes = np.max(y) + 1

    for class_idx in range(num_classes):

        num_samples = np.sum(y == class_idx)

        if min_samples is None or num_samples < min_samples:
            min_samples = num_samples

        if max_samples is None or num_samples > max_samples:
            max_samples = num_samples

    return min_samples, max_samples


def oversample(x, y, min_samples=None):
    """
    Oversample the data.
    :param x:                   Data.
    :param y:                   Labels.
    :param min_samples:         Minimum number of samples per class.
    :return:                    Oversampled data.
    """

    _, max_samples = get_min_and_max_samples_per_class(y)

    if min_samples is not None:
        max_samples = max(max_samples, min_samples)

    num_classes = np.max(y) + 1

    x, y = shuffle(x, y)

    sampled_x = []
    sampled_y = []

    for class_idx in range(num_classes):

        mask = y == class_idx

        if np.sum(mask) == max_samples:

            sampled_x.append(x[mask])
            sampled_y.append(y[mask])

        else:

            sampled_x.append(np.random.choice(x[mask], size=max_samples, replace=True))
            sampled_y.append(np.random.choice(y[mask], size=max_samples, replace=True))

    sampled_x = np.concatenate(sampled_x, axis=0)
    sampled_y = np.concatenate(sampled_y, axis=0)

    return shuffle(sampled_x, sampled_y)


def stratified_split(x, y, validation_fraction):
    """
    Create a stratified training and validation split.
    :param x:       Training data.
    :param y:       Training labels.
    :param validation_fraction:         Validation fraction.
    :return:                            Training and validation splits.
    """

    num_classes = np.max(y) + 1
    x, y = shuffle(x, y)

    valid_x = []
    valid_y = []
    train_x = []
    train_y = []

    for class_idx in range(num_classes):
        mask = y == class_idx

        validation_size = int(np.sum(mask) * validation_fraction)
        assert validation_size > 0

        valid_x.append(x[mask][:validation_size])
        valid_y.append(y[mask][:validation_size])
        train_x.append(x[mask][validation_size:])
        train_y.append(y[mask][validation_size:])

    valid_x = np.concatenate(valid_x, axis=0)
    valid_y = np.concatenate(valid_y, axis=0)
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    return train_x, train_y, valid_x, valid_y
