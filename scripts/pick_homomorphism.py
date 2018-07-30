import envs.pick as pick
import homomorphism


partition = homomorphism.partition_iteration(pick)

print("partition:")
for block in partition:
    print("block:", list(block))