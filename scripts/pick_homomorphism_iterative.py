import envs.pick as pick
import homomorphism


mdp = type('', (), {})()
mdp.STATES = []
mdp.ACTIONS = []
mdp.P = {}
mdp.R = {}

idx = 0

for state, action in pick.P.keys():

    mdp.STATES.append(state)
    mdp.ACTIONS.append(action)
    mdp.P[(state, action)] = pick.P[(state, action)]
    mdp.R[state] = pick.R[state]

    state_action_partition = homomorphism.partition_iteration(mdp)
    state_partition = homomorphism.get_state_partition(state_action_partition)

    print("step {:d}".format(idx + 1))
    print("state partition:")
    for block in state_partition:
        print("block:", list(block))
    print()

    idx += 1
