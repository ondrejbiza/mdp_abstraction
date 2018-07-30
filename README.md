# MDP Abstraction #

## Environments ##

### Pick ###

![ab](images/pick_env.png)

A simple environment with 5 states and 4 actions. The goal is to pick the puck in the
2x2 grid. The states are illustrated above and the actions are simply PICK for each
of the 4 possible positions in the grid. If the PICK action is executed in the coordinate
where the puck is the environment transitions into state 5 and reward 1 is award. Otherwise,
the environment stays in the same state and the reward is 0.

### Redundant Pick ###

This environment is the same as the Pick environment except there are 4 redundant states 11, 12, 13, and 14 
that behave in the same way as states 1, 2, 3 and 4.

![ab](images/redundant_pick_env.png)

## Abstraction ##

### Bisimulation ###

**Pick:**

state partition = {{1}, {2}, {3}, {4}, {5}}

**Redundant Pick:**

state partition = {{1, 11}, {2, 12}, {3, 13}, {4, 14}, {5}}


### Homomorphism

**Pick:**

state partition = {{1, 2, 3, 4}, {5}}

## Setup ##
 
Install Python 3.

## Usage ##

```
python -m scripts.pick_bisimulation
...
```