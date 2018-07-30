import envs.pick as pick
import bisimulation


partition = bisimulation.partition_iteration(pick)

print("partition:")
for block in partition:
    print("block:", list(block))