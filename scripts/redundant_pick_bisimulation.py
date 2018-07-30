import envs.redundant_pick as redundant_pick
import bisimulation


partition = bisimulation.partition_iteration(redundant_pick)

print("partition:")
for block in partition:
    print("block:", list(block))