import numpy as np
import random
import csv
import sys
import argparse
import copy 

class MarkovDecisionProcess:
    def __init__(self, csv_data):
        self.csv_data = csv_data
        # initial state
        self.state = {
            "maxDur": 59,
            "minDur": 15,
            "max_gap": 1,
            "next": [3, 5],
            "waitingTime": self.extract_waiting_time(csv_data, maxDur = "59", minDur = "15", max_gap = 1, next = "[3, 5]")
        }
        
        
        self.bounds = {
            "maximum_for_maxDur": 61,
            "minimum_for_maxDur": 59,
            "maximum_for_minDur": 17,
            "minimum_for_minDur": 15, 
            "maximum_for_max_gap": 3,
            "minimum_for_max_gap": 1,
            "maximum_for_next": [3, 5],
            "minimum_for_next": [3, 5]                    
                    
        }

        self.discount_factor = 0.9  # Discount factor for future rewards
    


    def extract_waiting_time(self,csv_data,maxDur,minDur,max_gap,next):
        for entry in csv_data:
            if entry.get('min_dur') == str(minDur) and entry.get('max_dur') == str(maxDur) and entry.get("max_gap") == str(max_gap) and entry.get("next_state") == next:
                waiting_time = entry.get('waiting_time')
        return waiting_time
        

    def get_actions(self, state):
        actions = [
            "increase_maxDur",
            "decrease_maxDur",
            "increase_minDur",
            "decrease_minDur",            
            "increase_max_gap",
            "decrease_max_gap",  
        ]

        # Remove actions that go beyond the upper and lower bounds
        if state["maxDur"] == self.bounds["maximum_for_maxDur"]:
            actions.remove('increase_maxDur')
        if state["maxDur"] == self.bounds["minimum_for_maxDur"]:
            actions.remove('decrease_maxDur')         
        if state["minDur"] == self.bounds["maximum_for_minDur"]:
            actions.remove('increase_minDur')
        if state["minDur"] == self.bounds["minimum_for_minDur"]:
            actions.remove('decrease_minDur')  
        if state["max_gap"] == self.bounds["maximum_for_max_gap"]:
            actions.remove('increase_max_gap')
        if state["max_gap"] == self.bounds["minimum_for_max_gap"]:
            actions.remove('decrease_max_gap')              

        return actions


    # Deterministic, so we can return with Probability 1 for each action.
    def transition_function(self, state, action):
        currWaitingTime = state["waitingTime"]
        nextstate = copy.deepcopy(state)

        if action == 'increase_minDur':
            nextstate["minDur"] += 1
            # Update waiting time state

        elif action == 'decrease_minDur':
            nextstate["minDur"] -= 1            
            # Update waiting time state
        
        elif action == 'increase_maxDur': 
            nextstate["maxDur"] += 1                      

        elif action == 'decrease_maxDur':
            nextstate["maxDur"] -= 1              

        elif action == 'increase_max_gap':
            nextstate["max_gap"] += 1                      

        # elif action == 'decrease_max_gap':
        else:
            nextstate["max_gap"] -= 1                      

        nextWaitingTime = self.extract_waiting_time(self.csv_data, maxDur = str(nextstate["maxDur"]), minDur = str(nextstate["minDur"]), max_gap = str(nextstate["max_gap"]), next = str(nextstate["next"]))            
        nextstate["waitingTime"] = nextWaitingTime
        # reward = -1 * [WaitingTime(s') - WaitingTime(s)]
        # If waiting time decreased (improved): Positive reward
        # If waiting time increased (deproved): Negative reward

        reward = -1*(float(nextWaitingTime) - float(currWaitingTime))
        return state,nextstate,reward



    def value_iteration(self, epsilon=0.01):
            # V array is a value function table that stores the estimated utility for each possible combination of state variables in MDP.
            V = np.zeros((self.bounds["maximum_for_maxDur"] - self.bounds["minimum_for_maxDur"] + 1,
                          self.bounds["maximum_for_minDur"] - self.bounds["minimum_for_minDur"] + 1,
                          self.bounds["maximum_for_max_gap"] - self.bounds["minimum_for_max_gap"] + 1))        
            
            policy = np.empty(V.shape, dtype="<U40")
            a = True
            while a:
                delta = 0
                # 59 to 61
                for maxDur in range(self.bounds["minimum_for_maxDur"], self.bounds["maximum_for_maxDur"] + 1):
                    # 15 to 17
                    for minDur in range(self.bounds["minimum_for_minDur"], self.bounds["maximum_for_minDur"] + 1):
                        # 1 to 3
                        for max_gap in range(self.bounds["minimum_for_max_gap"], self.bounds["maximum_for_max_gap"] + 1):

                            state = {
                                'maxDur': maxDur,
                                'minDur': minDur,
                                'max_gap': max_gap,
                                'next': [3,5],
                                'waitingTime': self.extract_waiting_time(self.csv_data, maxDur = maxDur, minDur = minDur, max_gap = max_gap, next = "[3, 5]")

                            }

                            # print(state)

                            v = V[maxDur - self.bounds["minimum_for_maxDur"],
                                  minDur - self.bounds["minimum_for_minDur"],
                                  max_gap - self.bounds["minimum_for_max_gap"]]       
                    
                    
                            action_values = []
                            actions = self.get_actions(state)
                            print(actions)
                            for action in actions:  # available_actions
                                # next_state, reward = self.perform_action(state, action)
                                state,next_state,reward = self.transition_function(state,action)
                                next_value = V[next_state['maxDur'] - self.bounds["minimum_for_maxDur"],
                                               next_state['minDur'] - self.bounds["minimum_for_minDur"],
                                            next_state['max_gap'] - self.bounds["minimum_for_max_gap"]]  
                                action_values.append(reward + self.discount_factor * next_value)
                    
                            # print(action_values)
    

                            # Update the value function and policy
                            best_action_index = np.argmax(action_values)  # Minimize reward
                            # print(best_action_index)
                    

                            print(f"Action Values: {action_values}, Best Action: {self.get_actions(state)[best_action_index]}")

                            V[maxDur - self.bounds["minimum_for_maxDur"],
                              minDur - self.bounds["minimum_for_minDur"],
                            max_gap - self.bounds["minimum_for_max_gap"]]= max(action_values)

                            policy[maxDur - self.bounds["minimum_for_maxDur"],
                                   minDur - self.bounds["minimum_for_minDur"],
                                   max_gap - self.bounds["minimum_for_max_gap"]] = self.get_actions(state)[best_action_index]

                            delta = max(delta, abs(v - V[maxDur - self.bounds["minimum_for_maxDur"],
                                                         minDur - self.bounds["minimum_for_minDur"],
                                                        max_gap - self.bounds["minimum_for_max_gap"]]))
                    
                print(f"Delta: {delta}")
                if delta < epsilon:
                    break

# # # The optimal value function in the context of Markov Decision Processes (MDPs) represents the expected cumulative reward that an agent can achieve starting from a particular state and following the optimal policy thereafter. 
            return V, policy

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-p",
        dest="path_csv",
        type=str,
        help="path of csv file with waiting time.\n",
    )

    argss = args.parse_args()
    csv_file_path = argss.path_csv

    try:
        csv_data = read_csv_file(csv_file_path)

    except FileNotFoundError:
        print(f"Error: File not found at path {csv_file_path}")
        sys.exit(1)

    mdp = MarkovDecisionProcess(csv_data)
    optimal_value_function,policy = mdp.value_iteration()
    # Print the optimal value function
    print("Optimal Value Function:")
    print(optimal_value_function)
    print()
    print("Policy:")
    print(policy)



