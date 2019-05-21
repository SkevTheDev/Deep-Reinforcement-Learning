import numpy as np

# This class is a recreation of OpenAI Gym FrozenLake-v0 Environment
# The gym environment does not allow for the flexibility needed to solve using Dynamic Programming
# as you cannot access states individually and perform actions. This environment gives the necessary control

# FrozenLake Configuration
# Environment
# SFFF
# FHFH
# FFFH
# HFFG

# States
# S = Start
# F = Frozen Lake --> (Nominal State)
# H = Hole (End --> Negative)
# G = Goal (End --> Positive)

# Actions
# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3

ACTION_LIST = ("LEFT", "DOWN", "RIGHT", "UP")


class FrozenLake:
    def __init__(self, width, height, start_row, start_col, random_action=0.2):
        self.width = width
        self.height = height
        self.row = start_row
        self.column = start_col
        self.rewards = None
        self.actions = None
        self.random_action = random_action

    def set_rewards_actions(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    def set_state(self, state):
        self.row = state[0]
        self.column = state[1]

    def current_state(self):
        return self.row, self.column

    def get_all_defined_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

    def terminal_state_reached(self, state):
        return state not in self.actions

    def step(self, action):
        # X% of the time, the action we wanted to take will not work and a random action will be executed
        # default is 20%
        if np.random.random() < self.random_action:
            action = np.random.choice(ACTION_LIST)
        # the actions dictionary will define the legal bounds of the environment and what allowable actions can be
        # taken at each state. using these bounds does not help the agent just confines it to the allowable state space
        # if the selected action is not allowed then the agent will not do anything and the current state will persist
        if action in self.actions[(self.row, self.column)]:
            if action == "LEFT":
                self.column -= 1
            elif action == "DOWN":
                self.row += 1
            elif action == "RIGHT":
                self.column += 1
            elif action == "UP":
                self.row -= 1
        # return the reward for the action taken
        return self.rewards.get((self.row, self.column), 0)

    def game_over(self):
        return (self.row, self.column) not in self.actions


def frozen_lake_v0(random_action=0.2, hole_reward=-1):
    print("Initialized with Random Action Probability: " + str(random_action))
    print("Initialized with Hole Reward Value: " + str(hole_reward))
    environment = FrozenLake(4, 4, 0, 0, random_action)
    rewards = {(2, 3): 1, (1, 1): hole_reward, (2, 2): hole_reward, (3, 0): hole_reward}
    actions = {
        (0, 0): ("DOWN", "RIGHT"),
        (0, 1): ("LEFT", "DOWN", "RIGHT"),
        (0, 2): ("LEFT", "DOWN", "RIGHT"),
        (0, 3): ("LEFT", "DOWN"),
        (1, 0): ("DOWN", "RIGHT", "UP"),
        # (1, 1): ("LEFT", "DOWN", "RIGHT", "UP"), --> TERMINAL STATE
        (1, 2): ("LEFT", "DOWN", "RIGHT", "UP"),
        (1, 3): ("LEFT", "DOWN", "UP"),
        (2, 0): ("DOWN", "RIGHT", "UP"),
        (2, 1): ("LEFT", "DOWN", "RIGHT", "UP"),
        # (2, 2): ("LEFT", "DOWN", "RIGHT", "UP"), --> TERMINAL STATE
        # (2, 3): ("LEFT", "DOWN", "UP"), --> TERMINAL STATE (GOAL)
        # (3, 0): ("RIGHT", "UP"), --> TERMINAL STATE
        (3, 1): ("LEFT", "RIGHT", "UP"),
        (3, 2): ("LEFT", "RIGHT", "UP"),
        (3, 3): ("LEFT", "UP")
    }
    environment.set_rewards_actions(rewards, actions)
    return environment


def frozen_lake_v0_negative_reward_steps(random_action=0.2, hole_reward=-1, step_cost=-0.1):
    environment = frozen_lake_v0(random_action=random_action, hole_reward=hole_reward)
    # updates the environment reward system to give small negative reward for taking steps in hopes that this helps
    # the algorithm to find the most efficient paths
    environment.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (0, 3): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (1, 3): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (3, 1): step_cost,
        (3, 2): step_cost,
        (3, 3): step_cost
    })
    return environment




