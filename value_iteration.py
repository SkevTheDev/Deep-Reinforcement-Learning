import numpy as np
import gym
import copy
from tensorboardX import SummaryWriter

CONVERGE_THRESHOLD = 10e-4
GAMMA = 0.9

if __name__ == '__main__':
    # Initialize 4x4 FrozenLake Environment
    environment = gym.make("FrozenLake-v0")

    # Reset the Environment back to its initial state
    environment.reset()

    # Render the Environment in the Console
    # Out put is as follows

    # SFFF
    # FHFH
    # FFFH
    # HFFG

    # S = Start
    # F = Frozen Lake --> (Nominal State)
    # H = Hole (End --> Negative)
    # G = Goal (End --> Positive)

    # LEFT = 0
    # DOWN = 1
    # RIGHT = 2
    # UP = 3

    # Frozen Lake has a reward system of (0, 1)
    # 1 if the Goal State is reached, otherwise 0
    print("\nEnvironment")
    environment.render()

    print("\nNumber of States")
    print(environment.observation_space.n)
    print("\nAvailable Actions")
    print(environment.action_space.n)

    # define terminal states
    terminal_states = [5, 7, 11, 12, 15]

    # for all states in the observation space randomly choose and action
    # implements random policy initialization
    policy = {}
    for state in range(environment.observation_space.n):
        if state not in terminal_states:
            policy[state] = environment.action_space.sample()

    # print initial random policy
    print("\nInitial Random Policy")
    print(policy)

    # Initialize Values for states that aren't terminal states
    V = {}
    for state in range(environment.observation_space.n):
        if state not in terminal_states:
            V[state] = np.random.random()
        else:
            V[state] = 0

    # print randomly selected values
    print("\nInitial Random Values")
    print(V)

    states = {}

    for state in states:
        state.render()

    print(states)

    biggest_change = 0
    for state in range(environment.observation_space.n):
        v_previous = V[state]

        if state in policy:
            v_new = float('-inf')
            environment.render()
            current_state = copy.deepcopy(environment)
            print("debug")
            # for action in range(environment.action_space.n):
            #     print("\nCurrent State")
            #     print(state)
            #     print("\nCurrent Action")
            #     print(action)
            #     current_state.step(action)
            #     current_state.render()
            #     environment.render()
            #     print("debug")









