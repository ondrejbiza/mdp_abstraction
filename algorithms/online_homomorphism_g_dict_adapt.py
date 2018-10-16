import collections
import random
import numpy as np
import algorithms.utils as utils


class OnlineHomomorphismGDict:

    RESOLVE_IGNORE = 0
    RESOLVE_ADD_CLOSEST = 1
    RESOLVE_ADD_TO_BIGGEST = 2
    RESOLVE_ADD_TO_RANDOM = 3

    def __init__(self, experience, classifier, sample_actions, threshold_multiplier, threshold_minimum,
                 outlier_resolution, max_partition_iteration_steps, visualize_b=None):

        self.partition = {frozenset(experience)}
        self.classifier = classifier
        self.sample_actions = sample_actions

        self.threshold_multiplier = threshold_multiplier
        self.threshold_minimum = threshold_minimum
        self.outlier_resolution = outlier_resolution
        self.max_partition_iteration_steps = max_partition_iteration_steps
        self.visualize_b = visualize_b

        self.ignored = set()
        self.per_class_accuracy = None

    def partition_iteration(self):
        """
        Run partition iteration.
        :return:    Number of partition iteration steps until the algorithm converged or a maximum steps threshold
                    was reached.
        """

        step = 0
        change = True

        while change and step < self.max_partition_iteration_steps:

            if step == 0:
                change = self.__split_rewards()
                self.__train_classifier()
            else:
                change = self.__partition_improvement()

            step += 1

            if self.visualize_b is not None:
                print("step {:d} state-action partition:".format(step))
                self.visualize_b(self.partition)

        return step

    def __partition_improvement(self):
        """
        Run a single step of partition improvement.
        :return:                True if the partition changed, False otherwise.
        """

        change = self.__split()

        if change:
            self.__train_classifier()

        return change

    def __split(self):
        """
        Split state-action blocks with respect to state blocks.
        :return:                        True if the state-action partition changed, otherwise False.
        """

        new_blocks = collections.defaultdict(list)
        confidences = collections.defaultdict(list)
        reward_blocks = collections.defaultdict(list)

        # get all experience
        experience = []
        for block in self.partition:
            experience += list(block)
        if len(self.ignored) > 0:
            experience += list(self.ignored)
            self.ignored = set()

        # split experience based on next state blocks
        for state, action, reward, next_state, done in experience:

            if reward > 0:
                reward_blocks[reward].append((state, action, reward, next_state, done))
            else:
                sampled_actions = self.sample_actions(state)
                blocks = set()

                confidence = 1

                for sampled_action in sampled_actions:

                    prediction = self.classifier.predict(next_state, sampled_action)
                    confidence *= self.per_class_accuracy[prediction]

                    blocks.add(prediction)

                key = frozenset(blocks)

                new_blocks[key].append((state, action, reward, next_state, done))
                confidences[key].append(confidence)

        # resolve outlier state blocks
        flag = True
        while len(new_blocks) > 1 and flag:

            # compute adaptive threshold
            num_samples = len(experience)
            thresholds = {}

            for key in confidences.keys():
                thresholds[key] = self.threshold_multiplier * (1 - np.mean(confidences[key])) * num_samples
                thresholds[key] = max(thresholds[key], self.threshold_minimum)

            # iterate over blocks
            for current_key in new_blocks.keys():

                block = new_blocks[current_key]
                size = len(block)

                if size < thresholds[current_key]:

                    if self.outlier_resolution == self.RESOLVE_ADD_CLOSEST:
                        # add to closest block above threshold
                        closest_key = None
                        min_distance = None

                        for key in new_blocks.keys():

                            # ignore self
                            if key == current_key:
                                continue

                            distance = utils.edit_distance(list(sorted(list(current_key))), list(sorted(list(key))))
                            if min_distance is None or min_distance > distance:
                                min_distance = distance
                                closest_key = key

                        new_blocks[closest_key] += new_blocks[current_key]
                        confidences[closest_key] += confidences[current_key]
                        del new_blocks[current_key]
                        del confidences[current_key]

                    elif self.outlier_resolution == self.RESOLVE_ADD_TO_BIGGEST:
                        # add to the biggest block
                        max_key, _ = max(new_blocks.items(), key=lambda x: len(x[1]))
                        assert max_key != current_key   # this can actually happen

                        new_blocks[max_key] += new_blocks[current_key]
                        confidences[max_key] += confidences[current_key]
                        del new_blocks[current_key]
                        del confidences[current_key]

                    elif self.outlier_resolution == self.RESOLVE_ADD_TO_RANDOM:
                        # add to a randomly selected block
                        keys = list(new_blocks.keys())
                        keys.remove(current_key)
                        rand_key = random.choice(list(new_blocks.keys()))

                        new_blocks[rand_key] += new_blocks[current_key]
                        confidences[rand_key] += confidences[current_key]
                        del new_blocks[current_key]
                        del confidences[current_key]

                    else:
                        # ignore the misclassified experience
                        self.ignored.union(new_blocks[current_key])
                        del new_blocks[current_key]
                        del confidences[current_key]

                    break

                # all blocks are larger than the thresholds, stop
                flag = False

        # create a new partition
        new_partition = set([frozenset(value) for value in new_blocks.values()])
        for block in reward_blocks.values():
            new_partition.add(frozenset(block))

        change = len(new_partition) > len(self.partition)

        if change:
            self.partition = new_partition

        return change

    def __split_rewards(self):
        """
        Split experience based on the observed rewards. Assumes there is only one state-action block.
        :return:            True if the state-action partition changed, otherwise False.
        """

        assert len(self.partition) == 1

        new_blocks = {}

        for state, action, reward, next_state, done in list(self.partition)[0]:

            if reward not in new_blocks:
                new_blocks[reward] = []

            new_blocks[reward].append((state, action, reward, next_state, done))

        self.partition = set()
        self.__add_blocks(new_blocks.values())

        return len(self.partition) > 1

    def __add_blocks(self, new_blocks):
        """
        Add state-action blocks to the state-action partition.
        :param new_blocks:      List of state-action blocks.
        :return:
        """

        for new_block in new_blocks:

            if len(new_block) > 0:
                self.partition.add(frozenset(new_block))

    def __train_classifier(self):
        """
        Train a model to classify state-action pairs into blocks in state-action partition.
        :return:                                None.
        """

        self.per_class_accuracy = self.classifier.fit(self.partition)
