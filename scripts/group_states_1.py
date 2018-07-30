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

d1 = {}

for transition in experience:

    key = (transition[2], transition[3])
    value = (transition[0], transition[1])

    if key not in d1.keys():
        d1[key] = []

    d1[key].append(value)

d2 = {key: idx for idx, key in enumerate(d1.keys())}
d3 = {}

for key, value in d1.items():

    id = d2[key]

    for t in value:

        state = t[0]

        if state not in d3:
            d3[state] = []

        d3[state].append(id)

print(d3)
