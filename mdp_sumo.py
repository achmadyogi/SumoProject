import numpy as np
import random
import csv
import sys
import argparse

class MarkovDecisionProcess:
    '''
    Current terminal_state: Average waiting time <= 150
    '''
    def __init__(self, csv_data, terminal_state = 150):
        # TODO: delta threshold
        self.terminal_state = terminal_state
        self.csv_data = csv_data
        self.state = {
            "maxDur": 51,
            "minDur": 11,
            "max_gap": 1,
            "next": [1, 3, 5, 7],
            "waitingTime": self.extract_waiting_time(csv_data, maxDur = "51", minDur = "11", max_gap = "1", next = "[1, 3, 5, 7]")
        }
        self.discount_factor = 0.9  # Discount factor for future rewards

    def extract_waiting_time(self,csv_data,maxDur,minDur,max_gap,next):
        for entry in csv_data:
            try:
                if entry.get('min_dur') == str(minDur) and entry.get('max_dur') == str(maxDur) and entry.get("max_gap") == str(max_gap) and entry.get("next_state") == next:
                    waiting_time = entry.get('waiting_time')
                return waiting_time

            except:
                print("NOT IN FILE!!!")
                print("parameters")
                # print(maxDur,minDur,max_gap,next)

        
    # TODO: Check for convergence 
    def is_terminal(self, state):
        return state["waitingTime"] == self.terminal_state 

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
        if state["maxDur"] == 70:
            actions.remove('increase_maxDur')
        if state["maxDur"] == 51:
            actions.remove('decrease_maxDur')         
        if state["minDur"] == 20:
            actions.remove('increase_minDur')
        if state["minDur"] == 11:
            actions.remove('decrease_minDur')  
        if state["max_gap"] == 5:
            actions.remove('increase_max_gap')
        if state["max_gap"] == 1:
            actions.remove('decrease_max_gap')              
        if state["next"] == [1, 3, 5, 7]:
            actions.remove('add_next')
        if state["next"] == []:
            actions.remove('remove_next')    

        return actions

    # Deterministic, so we can return with Probability 1 for each action.
    def transition_function(self, state, action):
        if action == 'increase_minDur':
            state["minDur"] += 1
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        elif action == 'decrease_minDur':
            state["minDur"] -= 1
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        elif action == 'increase_maxDur':
            state["maxDur"] += 1
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        elif action == 'decrease_maxDur':
            state["maxDur"] -= 1   
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        elif action == 'increase_max_gap':
            state["max_gap"] += 1
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        elif action == 'decrease_max_gap':
            state["max_gap"] -= 1
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state
        
        # [1, 3, 5, 7]
        elif action == 'add_next':
            max_next = [1, 3, 5, 7]
            curr_next = state["next"]
            options_to_add = [item for item in max_next if item not in curr_next]
            # Randomly choose one item with equal probability
            selected_option = random.choice(options_to_add)
            # Append the selected option to curr_next
            curr_next.append(selected_option)
            state["next"] = curr_next
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
            state["waitingTime"] = self.extract_waiting_time(self.csv_data, maxDur = str(state["maxDur"]), minDur = str(state["minDur"]), max_gap = str(state["max_gap"]), next = str(state["next"]))
            return state

    '''
    -1 * [WaitingTime(s') - WaitingTime(s)]
    If waiting time decreased (improved): Positive reward
    If waiting time increased (deproved): Negative reward
    '''

    def reward_function(self, state, action, next_state):
        return -1*(float(next_state["waitingTime"]) - float(state["waitingTime"]))
    
    # random walk
    def generate_episode(self):
        episode = []
        while not self.is_terminal(self.state):
            available_actions = self.get_actions(self.state)
            action = np.random.choice(available_actions)
            next_state = self.transition_function(self.state, action)
            print(next_state)
            reward = self.reward_function(self.state, action, next_state)
            episode.append((self.state, action, reward))
            self.state = next_state
        return episode

    def calculate_discounted_return(self, episode):
        G = 0
        for t, (_, _, reward) in enumerate(reversed(episode)):
            G = G * self.discount_factor + reward
        return G

    def value_iteration(self, epsilon=0.01):
            V = np.zeros(self.grid_size)
            policy = np.zeros(self.grid_size, dtype='<U5')  # Initialize policy with empty strings

            while True:
                delta = 0
                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        state = (i, j)
                        if not self.is_terminal(state):
                            v = V[i, j] #at initialized state (0,0)
                            action_values = []
                            for action in self.get_actions(state):
                                next_state = self.transition_function(state, action)
                                reward = self.reward_function(state, action, next_state)
                                next_value = V[next_state[0], next_state[1]]
                                action_values.append(reward + self.discount_factor * next_value)

                            # Update the value function and policy
                            best_action_index = np.argmax(action_values)
                            V[i, j] = max(action_values)
                            policy[i, j] = self.get_actions(state)[best_action_index]
                            delta = max(delta, abs(v - V[i, j]))

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
    mdp = MarkovDecisionProcess(csv_data, terminal_state=150)

    # Generate an episode
    episode = mdp.generate_episode()

    # Print the generated episode
    print("Generated Episode:")
    for step in episode:
        print(step)

    # Calculate the discounted return for the episode
    discounted_return = mdp.calculate_discounted_return(episode)
    print("\nDiscounted Return:", discounted_return)

    # Perform value iteration to obtain the optimal value function
    optimal_value_function,policy = mdp.value_iteration()

    # Print the optimal value function
    print("Optimal Value Function:")
    print(optimal_value_function)
    print()
    print("Policy:")
    print(policy)