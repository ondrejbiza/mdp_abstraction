class OnlineHomomorphismG:

    def __init__(self, experience, classifier, sample_actions, b_size_threshold, sb_size_threshold,
                 max_partition_iteration_steps, visualize_b=None, visualize_sb=None):

        self.partition = {frozenset(experience)}
        self.classifier = classifier
        self.sample_actions = sample_actions

        self.b_size_threshold = b_size_threshold
        self.sb_size_threshold = sb_size_threshold
        self.max_partition_iteration_steps = max_partition_iteration_steps
        self.visualize_b = visualize_b
        self.visualize_sb = visualize_sb

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
                state_partition = self.__state_projection()
                self.visualize_sb(state_partition)

            if self.visualize_b is not None:
                print("step {:d} state-action partition:".format(step))
                self.visualize_b(self.partition)

        return step

    def __partition_improvement(self):
        """
        Run a single step of partition improvement.
        :return:                True if the partition changed, False otherwise.
        """

        state_partition = self.__state_projection()
        change = False

        for state_block in state_partition:

            flag = True

            while flag:

                for new_block in self.partition:

                    if self.__split(new_block, state_block):
                        change = True
                        break

                    flag = False

        self.__train_classifier()
        return change

    def __split(self, state_action_block, state_block):
        """
        Split a state-action block with respect to a state block.
        :param state_action_block:      State-action block.
        :param state_block:             State block.
        :return:                        True if the state-action partition changed, otherwise False.
        """

        new_blocks = {
            True: [],
            False: []
        }

        for state, action, reward, next_state, done in state_action_block:

            # don't split reward blocks
            if reward > 0:
                return False

            key = next_state in state_block
            new_blocks[key].append((state, action, reward, next_state, done))

        if len(new_blocks[True]) >= self.b_size_threshold and len(new_blocks[False]) >= self.b_size_threshold:

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
        :return:                            State partition.
        """

        states = set()
        for block in self.partition:
            for state, action, reward, next_state, done in block:
                states.add(state)
                states.add(next_state)

        state_partition = {}

        for state in states:
            actions = self.sample_actions(state)
            blocks = set()
            for action in actions:
                blocks.add(self.classifier.predict(state, action))
            key = frozenset(blocks)
            if key not in state_partition:
                state_partition[key] = []
            state_partition[key].append(state)

        state_partition = set([frozenset(value) for value in state_partition.values()])
        return state_partition
