import collections
import random
import algorithms.utils as utils


class OnlineHomomorphismGDict:

    RESOLVE_IGNORE = 0
    RESOLVE_ADD_CLOSEST = 1
    RESOLVE_ADD_TO_BIGGEST = 2
    RESOLVE_ADD_TO_RANDOM = 3

    def __init__(self, experience, classifier, sample_actions, b_size_threshold,
                 outlier_resolution, max_partition_iteration_steps, visualize_b=None):

        self.partition = {frozenset(experience)}
        self.classifier = classifier
        self.sample_actions = sample_actions

        self.b_size_threshold = b_size_threshold
        self.outlier_resolution = outlier_resolution
        self.max_partition_iteration_steps = max_partition_iteration_steps
        self.visualize_b = visualize_b

        self.ignored = set()

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

                for sampled_action in sampled_actions:
                    blocks.add(self.classifier.predict(next_state, sampled_action))

                key = frozenset(blocks)

                new_blocks[key].append((state, action, reward, next_state, done))

        # resolve outlier state blocks
        if self.b_size_threshold > 1:
            while len(new_blocks) > 1:

                min_key, min_block = min(new_blocks.items(), key=lambda x: len(x[1]))
                min_size = len(min_block)

                if min_size >= self.b_size_threshold:
                    break
                else:

                    if self.outlier_resolution == self.RESOLVE_ADD_CLOSEST:
                        # add to closest block above threshold
                        closest_key = None
                        min_distance = None

                        for key in new_blocks.keys():

                            # ignore self
                            if key == min_key:
                                continue

                            distance = utils.edit_distance(list(sorted(list(min_key))), list(sorted(list(key))))
                            if min_distance is None or min_distance > distance:
                                min_distance = distance
                                closest_key = key

                        new_blocks[closest_key] += new_blocks[min_key]
                        del new_blocks[min_key]

                    elif self.outlier_resolution == self.RESOLVE_ADD_TO_BIGGEST:
                        # add to the biggest block
                        max_key, _ = max(new_blocks.items(), key=lambda x: len(x[1]))
                        assert max_key != min_key   # this can actually happen

                        new_blocks[max_key] += new_blocks[min_key]
                        del new_blocks[min_key]

                    elif self.outlier_resolution == self.RESOLVE_ADD_TO_RANDOM:
                        # add to a randomly selected block
                        keys = list(new_blocks.keys())
                        keys.remove(min_key)
                        rand_key = random.choice(list(new_blocks.keys()))

                        new_blocks[rand_key] += new_blocks[min_key]
                        del new_blocks[min_key]

                    else:
                        # ignore the misclassified experience
                        self.ignored.union(new_blocks[min_key])
                        del new_blocks[min_key]

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

        self.classifier.fit(self.partition)
