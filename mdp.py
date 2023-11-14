import numpy as np
import random
import csv
import sys
import argparse
import copy 

class MarkovDecisionProcess:
    def __init__(self, csv_data, terminal_state = 300):
        self.terminal_state = terminal_state
        self.csv_data = csv_data
        # initial state
        self.state = {
            "maxDur": 51,
            "minDur": 11,
            "max_gap": 1,
            "next": [1, 3, 5, 7],
            "waitingTime": self.extract_waiting_time(csv_data, maxDur = "51", minDur = "11", max_gap = "1", next = "[1, 3, 5, 7]")
        }

        
        # TODO: expand the range (this increases state space, might cause value iteration to become intractable)
        # Currently: 18 states
        self.bounds = {
            "maximum_for_maxDur": 51,
            "minimum_for_maxDur": 51,
            "maximum_for_minDur": 11,
            "minimum_for_minDur": 11, 
            "maximum_for_max_gap": 3,
            "minimum_for_max_gap": 1,
            "maximum_for_next": [1, 3, 5, 7],
            "minimum_for_next": [1, 3, 5, 7]                      
                    
        }

        self.discount_factor = 0.9  # Discount factor for future rewards

    def extract_waiting_time(self,csv_data,maxDur,minDur,max_gap,next):
        for entry in csv_data:
            if entry.get('min_dur') == str(minDur) and entry.get('max_dur') == str(maxDur) and entry.get("max_gap") == str(max_gap) and entry.get("next_state") == next:
                waiting_time = entry.get('waiting_time')
        return waiting_time
        

    def get_actions(self, state):
        actions = [
            "increase_minDur",
            "decrease_minDur",
            "increase_maxDur",
            "decrease_maxDur",
            "increase_max_gap",
            "decrease_max_gap",  
            "add_next",
            "remove_next"
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
        if state["next"] == self.bounds["maximum_for_next"]:
            actions.remove('add_next')
        if state["next"] == self.bounds["minimum_for_next"]:
            actions.remove('remove_next')    

        return actions


    # Deterministic, so we can return with Probability 1 for each action.
    def transition_function(self, state, action):
        if action == 'increase_minDur':
            state["minDur"] += 1
            # Update waiting time state
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        elif action == 'decrease_minDur':
            state["minDur"] -= 1            
            # Update waiting time state
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        elif action == 'increase_maxDur':
            state["maxDur"] += 1                
            # Update waiting time state
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        elif action == 'decrease_maxDur':
            state["maxDur"] -= 1                          
            # Update waiting time state
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        elif action == 'increase_max_gap':
            state["max_gap"] += 1                      
            # Update waiting time state
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        elif action == 'decrease_max_gap':
            state["max_gap"] -= 1                       
            # Update waiting time state
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        # [1, 3, 5, 7]
        # Ensure sorted list
        elif action == 'add_next':
            max_next = [1, 3, 5, 7]
            curr_next = state["next"]
            options_to_add = [item for item in max_next if item not in curr_next]
            # Randomly choose one item with equal probability
            selected_option = random.choice(options_to_add)
            # Append the selected option to curr_next
            curr_next.append(selected_option)
            state["next"] = sorted(curr_next)            
            
            # Update waiting time state
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state  
               
        # elif action == 'remove_next':
        # [1,5,7] equal chance of removing any.
        else:
            curr_next = state["next"]
            # Randomly choose one item with equal probability
            selected_option = random.choice(curr_next)
            # Append the selected option to curr_next
            curr_next.remove(selected_option)
            state["next"] = curr_next            

            # Update waiting time state
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
    def perform_action(self, state, action):
        # Keep track of the current state
        curr_state = copy.deepcopy(state)

        # Update the state based on the action
        self.state = self.transition_function(state, action)
        # state = self.transition_function(state, action)
        
        # Calculate the reward using reward_function
        reward = round(self.reward_function(curr_state, self.state),2)

        return self.state,reward        


    '''
    -1 * [WaitingTime(s') - WaitingTime(s)]
    If waiting time decreased (improved): Positive reward
    If waiting time increased (deproved): Negative reward
    '''

    def reward_function(self, state, next_state):
        curr_waiting_time = state["waitingTime"]
        next_waiting_time = next_state["waitingTime"]
        return -1*(float(next_waiting_time) - float(curr_waiting_time))
    
    # # random walk
    # def generate_episode(self):
    #     episode = []
    #     while not self.is_terminal(self.state):
    #         available_actions = self.get_actions(self.state)
    #         action = np.random.choice(available_actions)
    #         next_state,reward = self.perform_action(self.state,action)
    #         episode.append((next_state, action, reward))

    #     return episode

    # # calculate utility
    # def calculate_discounted_return(self, episode):
    #     # each episode: (next_state, action, reward)
    #     G = 0
    #     for idx, (_, _, reward) in enumerate(reversed(episode)):
    #         # Reversed so that the final action gets multiplied with discount_factor^n times if n steps
    #         # The first action is added as "+ reward"
    #         # Future actions hold less weight in deciding utility now
    #         G = G * self.discount_factor + reward
    #     return G

    def value_iteration(self, epsilon=0.01):
            # V array is a value function table that stores the estimated utility for each possible combination of state variables in MDP.
            V = np.zeros((self.bounds["maximum_for_maxDur"] - self.bounds["minimum_for_maxDur"] + 1, #3
                        self.bounds["maximum_for_minDur"] - self.bounds["minimum_for_minDur"] + 1, #3
                        self.bounds["maximum_for_max_gap"] - self.bounds["minimum_for_max_gap"] + 1,  #2
                        2**(len(self.bounds["maximum_for_next"]) - len(self.bounds["minimum_for_next"])) # 1 
                        ))        
            
            policy = np.empty(V.shape, dtype="<U40")

            while True:
                delta = 0
                for maxDur in range(self.bounds["minimum_for_maxDur"], self.bounds["maximum_for_maxDur"] + 1):
                    for minDur in range(self.bounds["minimum_for_minDur"], self.bounds["maximum_for_minDur"] + 1):
                        for max_gap in range(self.bounds["minimum_for_max_gap"], self.bounds["maximum_for_max_gap"] + 1):
                            next = [1, 3, 5, 7]
                            state = {
                                'maxDur': maxDur,
                                'minDur': minDur,
                                'max_gap': max_gap,
                                'next': next,
                                'waitingTime': self.extract_waiting_time(csv_data, maxDur, minDur, max_gap, next="[1, 3, 5, 7]")
                            }
                            print(f"State: {state}")
                            v = V[maxDur - self.bounds["minimum_for_maxDur"],
                                minDur - self.bounds["minimum_for_minDur"],
                                max_gap - self.bounds["minimum_for_max_gap"],
                                0]  # because only consider 1 option for next currently

                            action_values = []
                            for action in self.get_actions(state):  # available_actions
                                next_state, reward = self.perform_action(state, action)
                                next_value = V[next_state['maxDur'] - self.bounds["minimum_for_maxDur"],
                                            next_state['minDur'] - self.bounds["minimum_for_minDur"],
                                            next_state['max_gap'] - self.bounds["minimum_for_max_gap"],
                                            0]  # because only consider 1 option for next currently
                                action_values.append(reward + self.discount_factor * next_value)

                            # Update the value function and policy
                            best_action_index = np.argmax(action_values)  # Minimize reward

                            print(f"Action Values: {action_values}, Best Action: {self.get_actions(state)[best_action_index]}")

                            V[maxDur - self.bounds["minimum_for_maxDur"],
                            minDur - self.bounds["minimum_for_minDur"],
                            max_gap - self.bounds["minimum_for_max_gap"],
                            0] = max(action_values)

                            policy[maxDur - self.bounds["minimum_for_maxDur"],
                                minDur - self.bounds["minimum_for_minDur"],
                                max_gap - self.bounds["minimum_for_max_gap"],
                                0] = self.get_actions(state)[best_action_index]

                            delta = max(delta, abs(v - V[maxDur - self.bounds["minimum_for_maxDur"],
                                                        minDur - self.bounds["minimum_for_minDur"],
                                                        max_gap - self.bounds["minimum_for_max_gap"],
                                                        0]))

                print(f"Delta: {delta}")
                if delta < epsilon:
                    break

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

    # Example usage:
    mdp = MarkovDecisionProcess(csv_data, terminal_state = 100)
    # initial_state = copy.deepcopy(mdp.state)
    # # Generate an episode
    # episode = mdp.generate_episode()

    # print("Initial step:")
    # print(initial_state)
    # print()

    # # Print the generated episode
    # print("Generated Episode:")
    # for step in episode:
    #     print(step)
    #     print()

    # # Calculate the discounted return for the episode
    # discounted_return = mdp.calculate_discounted_return(episode)
    # print("\nDiscounted Return:", discounted_return)

    # # Perform value iteration to obtain the optimal value function
    optimal_value_function,policy = mdp.value_iteration()

    # Print the optimal value function
    print("Optimal Value Function:")
    print(optimal_value_function)
    print()
    print("Policy:")
    print(policy)
