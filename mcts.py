

import pandas as pd
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit('Please declare the environment variable SUMO_HOME')
    # Can declare using export SUMO_HOME="/usr/local/Cellar/sumo" (default location)
import traci

import subprocess
import xml.etree.ElementTree as ET

from mdp_sumo import *

'''Traffic simulation doesn't have a terminal condition so we need a fixed depth for the simulation, 
we simplify the simulate function further by relying solely on the fixed depth for the simulation.'''

import random
import math

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

def select(node):
    # Select a child node using UCT (Upper Confidence Bound for Trees) algorithm
    C = 1.0  # Exploration parameter
    
    if not node.children:
        return None  # No children to select, return None
        
    return max(node.children, key=lambda child: child.value / child.visits + C * math.sqrt(2 * math.log(node.visits) / child.visits))

def expand(node):
    # Expand the children of the selected node
    untried_actions = [action for action in node.state.get_possible_actions() if action not in [child.state.action for child in node.children]]
    if untried_actions:
        action = random.choice(untried_actions)
        new_state = node.state.perform_action(action)
        child_node = MCTSNode(new_state, parent=node)
        node.children.append(child_node)
        return child_node
    else:
        return None

def simulate(node, max_depth):
    # Simulate a random rollout from the current state up to max_depth
    state = node.state
    for _ in range(max_depth):
        action = random.choice(state.get_possible_actions())
        state = state.perform_action(action)
    return state.get_reward()

def backpropagate(node, result):
    # Backpropagate the result of a simulation
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent

def mcts(root_state, num_simulations, max_depth):
    root_node = MCTSNode(root_state)

    for _ in range(num_simulations):
        node = root_node

        # Selection
        # while not node.children:
        #     node = select(node)

        # Expansion
        node = expand(node)
        
        # Simulation
        result = simulate(node, max_depth) if node != None else -1

        # Backpropagation
        backpropagate(node, result)

    # Return the best action based on visit counts
    return max(root_node.children, key=lambda child: child.visits).state.action


class State:
    def __init__(self, id=None, action=None, param=None ,wait_time=None):
        self.action = action # a set of how to modify param
        
        self.space_id = id
        self.param = param #[maxDur, minDur, max-gap, next]
        self.wait_time = wait_time if wait_time != None else df_waiting_time[df_waiting_time.idx==id]['waiting_time'].values[0]

    def get_possible_actions(self):
        actions = []

        dfs = df_state_space
        dfs = dfs[dfs.idx == self.space_id]
        if dfs['max_gap'].values[0]!=max_gap_max : actions.append('inMaxGap' )
        if dfs['max_gap'].values[0]!=max_gap_min :actions.append('deMaxGap')
        if dfs['min_dur'].values[0]!=minDur_max : actions.append('inMinDur')
        if dfs['min_dur'].values[0]!=minDur_min : actions.append('deMinDur' )
        if dfs['max_dur'].values[0]!=maxDur_max : actions.append('inMaxDur')
        if dfs['max_dur'].values[0]!=maxDur_min : actions.append('deMaxDur')
        return actions #df_action_space[df_action_space.idx == self.space_id]

    def perform_action(self, action):
        dfs = df_state_space
        dfs = dfs[dfs.idx == self.space_id]
        
        max_gap = dfs['max_gap'].values[0] + 1 if action == 'inMaxGap' else dfs['max_gap'].values[0]
        max_gap = dfs['max_gap'].values[0] - 1 if action == 'deMaxGap' else dfs['max_gap'].values[0]
        min_dur = dfs['min_dur'].values[0] + 1 if action == 'inMinDur' else dfs['min_dur'].values[0]
        min_dur = dfs['min_dur'].values[0] - 1 if action == 'deMinDur' else dfs['min_dur'].values[0]
        max_dur = dfs['max_dur'].values[0] + 1 if action == 'inMaxDur' else dfs['max_dur'].values[0]
        max_dur = dfs['max_dur'].values[0] - 1 if action == 'deMaxDur' else dfs['max_dur'].values[0]
        # print(max_gap,min_dur,max_dur)
        
        id = df_state_space[ (df_state_space['max_gap'] == max_gap) \
                            & (df_state_space['min_dur'] == min_dur) \
                            & (df_state_space['max_dur'] == max_dur) \
                            ]["idx"].values[0]
        return State(id = id, action=action, wait_time=df_waiting_time[df_waiting_time.idx==id]['waiting_time'].values[0])

    def get_reward(self): 
        # curr_waiting_time = state["waitingTime"]
        # next_waiting_time = next_state["waitingTime"]
        # return -1*(float(next_waiting_time) - float(curr_waiting_time))
        
        # Struggling with getting back to parent node to get diff(waiting_time) between states
        return 1./self.wait_time #Use default rewards not the proposed in slides

######################################
# TCL set up
maxDur_max = 53
maxDur_min = 51
minDur_max = 13
minDur_min = 11
max_gap_max = 3
max_gap_min = 1
# next = None
######################################
# Read precomputed wait_time for each state
#     from csv file
import pandas as pd

df = pd.read_csv("./result_16000.csv")
df = df[['Unnamed: 0', 'max_gap', 'min_dur', 'max_dur', 'next_state','waiting_time']]
df = df.rename(columns={'Unnamed: 0': "idx"})


# state_space = []
df_state_space = df[['idx', 'max_gap', 'min_dur', 'max_dur', 'next_state']]
df_waiting_time = (df[['idx','waiting_time']])


# action_space = []
# [inMaxGap deMaxGap inMinDur deMinDur inMaxDur deMaxDur]
df['inMaxGap'] = df['max_gap']!=max_gap_max
df['deMaxGap'] = df['max_gap']!=max_gap_min
df['inMinDur'] = df['min_dur']!=minDur_max
df['deMinDur'] = df['min_dur']!=minDur_min
df['inMaxDur'] = df['max_dur']!=maxDur_max
df['deMaxDur'] = df['max_dur']!=maxDur_min
df_action_space = df[['idx','inMaxGap','deMaxGap','inMinDur','deMinDur','inMaxDur','deMaxDur']]

# print(df_state_space)



######################################
# Example usage:
# >>> python mcts.py mindur maxdur maxgap
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 3: 
        print("\nUsage: python mcts.py mindur maxdur maxgap\n")
    else:
        min_dur, max_dur, max_gap = args[0], args[1], args[2]
        id = df_state_space[ (df_state_space['max_gap'] == int(max_gap)) \
                            & (df_state_space['min_dur'] == int(min_dur)) \
                            & (df_state_space['max_dur'] == int(max_dur)) \
                            ]["idx"]
        if len(id)>0: id = id.values[0]
        root_state = State( id=id) if id!= None else State( id=5555)
        best_action = mcts(root_state, num_simulations=1000, max_depth=150)
        print("Best action for {} {} {}: {}".format(min_dur,max_dur,max_gap,best_action) )
