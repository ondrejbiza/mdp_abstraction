import envs.redundant_pick as redundant_pick
import homomorphism


state_action_partition = homomorphism.partition_iteration(redundant_pick)
state_partition = homomorphism.get_state_partition(state_action_partition)

print("state partition:")
for block in state_partition:
    print("block:", list(block))

print()
print("state-action partition:")
for block in state_action_partition:
    print("block:", list(block))