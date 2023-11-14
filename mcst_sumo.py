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
    # return state.get_reward()
    return node.state.get_reward(state)

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
        while not node.children:
            node = select(node)

        # Expansion
        node = expand(node)

        # Simulation
        result = simulate(node, max_depth)

        # Backpropagation
        backpropagate(node, result)

    # Return the best action based on visit counts
    return max(root_node.children, key=lambda child: child.visits).state.action

# Example usage:
class State:
    def __init__(self, mdp, mdp_state, action=None ):
        self.action = action
        self.status = mdp_state
        self.mdp = mdp
    
    def get_possible_actions(self, mdp):
        return self.mdp.get_actions(self.status) #from mdp_sumo

    def perform_action(self, mdp, action):
        mdp_next_state = self.mdp.transition_function(self.status, action)
        return State(mdp, mdp_next_state, action=action)

    def get_reward(self, next_state):
        reward = self.mdp.reward_function(self.status, self.action, next_state)                
        return reward


# MDP_sumo example usage:
csv_data = "../result_16000.csv"
mdp = MarkovDecisionProcess(csv_data, terminal_state=150)

root_state = State(mdp, mdp.state)
best_action = mcts(root_state, num_simulations=10000, max_depth=5)
print("Best action:", best_action)



