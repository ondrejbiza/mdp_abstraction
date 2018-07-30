from envs.pick import PickEnv


env = PickEnv()

experience = set()

for initial_state in env.INITIAL_STATES:

    for action_1 in env.ACTIONS:

        for action_2 in env.ACTIONS:

            env.state = initial_state

            state = env.state
            action = action_1
            reward, next_state = env.step(action)
            transition = (state, action, reward, next_state)
            experience.add(transition)

            state = env.state
            action = action_2
            reward, next_state = env.step(action)
            transition = (state, action, reward, next_state)
            experience.add(transition)

            env.state = initial_state

assert len(experience) == (4 * len(env.STATES))
