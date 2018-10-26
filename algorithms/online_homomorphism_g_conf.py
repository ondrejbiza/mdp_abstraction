import numpy as np
import algorithms.utils as utils


class OnlineHomomorphismG:

    def __init__(self, experience, classifier, sample_actions, b_size_threshold, sb_size_threshold,
                 max_partition_iteration_steps, conf_threshold, percentile=None, exclude_blocks=False,
                 order_state_blocks=False, visualize_b=None, visualize_sb=None, visualize_conf=None):

        self.partition = {frozenset(experience)}
        self.classifier = classifier
        self.sample_actions = sample_actions

        self.b_size_threshold = b_size_threshold
        self.sb_size_threshold = sb_size_threshold
        self.max_partition_iteration_steps = max_partition_iteration_steps
        self.conf_threshold = conf_threshold
        self.percentile = percentile
        self.exclude_blocks = exclude_blocks
        self.order_state_blocks = order_state_blocks

        self.visualize_b = visualize_b
        self.visualize_sb = visualize_sb
        self.visualize_conf = visualize_conf

        self.reward_block = None

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

            if self.visualize_sb is not None:
                print("step {:d} state partition:".format(step))
                state_partition, _ = self.__state_projection()
                self.visualize_sb(state_partition)

            if self.visualize_b is not None:
                print("step {:d} state-action partition:".format(step))
                self.visualize_b(self.partition)

            if self.visualize_conf is not None:
                self.visualize_conf(self.partition, self.classifier, self.sample_actions)

        return step

    def __partition_improvement(self):
        """
        Run a single step of partition improvement.
        :return:                True if the partition changed, False otherwise.
        """

        state_partition, confidences = self.__state_projection()
        change = False

        if self.order_state_blocks:
            state_partition = sorted(state_partition, key=lambda item: len(item), reverse=True)

        for state_block in state_partition:

            flag = True

            while flag:

                for new_block in self.partition:

                    if self.__split(new_block, state_block, confidences):
                        change = True
                        break

                    flag = False

        if change:
            self.__train_classifier()

        return change

    def __split(self, state_action_block, state_block, confidences):
        """
        Split a state-action block with respect to a state block.
        :param state_action_block:      State-action block.
        :param state_block:             State block.
        :param confidences:             Confidences for all states.
        :return:                        True if the state-action partition changed, otherwise False.
        """
        new_blocks = {
            True: [],
            False: []
        }
        new_blocks_counts = {
            True: 0,
            False: 0
        }

        for state, action, reward, next_state, done in state_action_block:

            # don't split reward blocks
            if reward > 0:
                return False

            key = next_state in state_block
            new_blocks[key].append((state, action, reward, next_state, done))

            confidence = confidences[next_state]

            if confidence >= self.conf_threshold:
                new_blocks_counts[key] += 1

        if new_blocks_counts[True] >= self.b_size_threshold and new_blocks_counts[False] >= self.b_size_threshold:

            self.partition.remove(state_action_block)
            self.__add_blocks(new_blocks.values())

            return True

        return False

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

    def __state_projection(self):
        """
        Get a state partition from a state-action partition using a state-action classifier.
        :return:                            State partition and confidences for all states.
        """

        states = set()
        for block in self.partition:
            for state, action, reward, next_state, done in block:
                states.add(state)
                states.add(next_state)

        state_partition = {}
        confidences = {}

        for state in states:

            actions = self.sample_actions(state)
            blocks = set()
            probs = []

            predictions = self.classifier.batch_predict_prob([state for _ in range(len(actions))], actions)

            for prediction in predictions:

                # predict a block for the state-action pair
                block = np.argmax(prediction)
                prob = prediction[block]
                probs.append(prob)

                # add the block, but only under some conditions
                if not self.exclude_blocks or prob >= self.conf_threshold:
                    blocks.add(block)

                blocks.add(block)

            if self.percentile is not None:
                confidence = np.percentile(probs, self.percentile)
            else:
                confidence = np.min(probs)

            key = frozenset(blocks)

            if key not in state_partition:
                state_partition[key] = []

            confidences[state] = confidence
            state_partition[key].append(state)

        if self.sb_size_threshold > 1:

            while True:

                min_key, min_block = min(state_partition.items(), key=lambda x: len(x[1]))
                min_size = len(min_block)

                if min_size >= self.sb_size_threshold:
                    break
                else:

                    # block is below threshold, add to closest block above threshold
                    closest_key = None
                    min_distance = None

                    for key in state_partition.keys():

                        # ignore self
                        if key == min_key:
                            continue

                        distance = utils.edit_distance(list(sorted(list(min_key))), list(sorted(list(key))))
                        if min_distance is None or min_distance > distance:
                            min_distance = distance
                            closest_key = key

                    state_partition[closest_key] += state_partition[min_key]
                    del state_partition[min_key]

        state_partition = set([frozenset(value) for value in state_partition.values()])
        return state_partition, confidences
