import envs.redundant_pick as redundant_pick
import homomorphism


partition = homomorphism.partition_iteration(redundant_pick)

print("partition:")
for block in partition:
    print("block:", list(block))