import numpy as np
import matplotlib.pyplot as plt
from unwrapped_frozen_lake_environment import frozen_lake_v0, frozen_lake_v0_negative_reward_steps

VALUE_CONVERGENCE_THRESHOLD = 10e-4
GAMMA = 0.99
LEARNING_RATE = 0.1
ACTION_LIST = ("LEFT", "DOWN", "RIGHT", "UP")

def print_rewards(Rewards, environment):
    for column in range(environment.width):
        print("------------------------")
        for row in range(environment.height):
            reward = Rewards.get((column, row), 0)
            if reward >= 0:
                print(" %.2f|" % reward, end="")
            else:
                print("%.2f|" % reward, end="")
        print("")


def print_policies(Policies, environment):
    for column in range(environment.width):
        print("-----------------------------------------------------------------")
        for row in range(environment.height):
            policy = Policies.get((column, row), 0)
            if policy == 0:
                policy = "HOLE"
            if policy == "UP":
                policy = "UP  "
            if row == 3 and column == 2:
                policy = "GOAL"
            print("\t%s\t|" % policy, end="")
        print("")

def play_game(environment, policies, random_action_probability=0.1):
    start_state = (0, 0)
    environment.set_state(start_state)

    if np.random.random() < random_action_probability:
        action = np.random.choice(ACTION_LIST)
    else:
        action = policies[start_state]

    states_actions_rewards = [(start_state, action, 0)]

    while True:
        reward = environment.step(action)
        new_state = environment.current_state()

        if environment.game_over():
            states_actions_rewards.append((new_state, None, reward))
            break
        else:
            if np.random.random() < random_action_probability:
                action = np.random.choice(ACTION_LIST)
            else:
                action = policies[new_state]
            states_actions_rewards.append((new_state, action, reward))

    accumulated_return = 0
    states_actions_returns = []
    terminal_state = True
    # compute the return iteratively by starting at the terminal state and moving back to the start_state
    # the return for the terminal state is ignored because the game is over
    for state, action, reward in reversed(states_actions_rewards):
        if terminal_state:
            terminal_state = False
        else:
            states_actions_returns.append((state, action, accumulated_return))
        accumulated_return = reward + (GAMMA * accumulated_return)
    states_actions_returns.reverse()
    return states_actions_returns


if __name__ == "__main__":
    final_policies = []
    for i in range(100):
        environment = frozen_lake_v0_negative_reward_steps()
        print("\nFrozen Lake Reward Structure")
        print_rewards(environment.rewards, environment)

        # create an empty policy dictionary
        Policies = {}
        # initialize a random policy
        for action in environment.actions.keys():
            Policies[action] = np.random.choice(ACTION_LIST)

        # print random policy
        print("\nRandom State Policies")
        print_policies(Policies, environment)

        # initialize Q table and the returns. All Q table values will be set to 0 instead of random values
        Q = {}
        returns = {}
        states = environment.get_all_defined_states()
        for state in states:
            if state in environment.actions:
                Q[state] = {}
                for action in ACTION_LIST:
                    Q[state][action] = 0
                    returns[(state, action)] = []
            else:
                pass

        # create lists to store the difference between the previous and current
        # Q values and the means of these differences
        value_deltas = []
        value_deltas_mean = []

        # set a defined number of episodes rather than just going until converged
        for i in range(5000):
            # define the value delta that will compare previous value with current value
            if i % 500 == 0:
                print("Training Episode: " + str(i))
            value_delta = 0

            # play an episode of the game and return the states, actions, returns tuple for each state visited
            states_actions_returns = play_game(environment, Policies)
            visited_states = set()

            # loop through each visited state and compute the q values for that state.
            # do not compute the value multiple times
            # if the a state has been visited more that once in an episode
            for state, action, accumulated_return in states_actions_returns:
                # define state action pairs
                state_action = (state, action)

                # only if the state has not been visited for this episode will we compute its Q values
                if state_action not in visited_states:
                    previous_q_value = Q[state][action]

                    # Policy Evaluation
                    # method for computing value based on average of returns
                    # returns[state_action].append(accumulated_return)
                    # Q[state][action] = np.mean(returns[state_action])

                    # update Q value using the incremental value function
                    # alternatively could take the mean of the return
                    # but that means returns would need to be stored in a
                    # structure which in some environments might not be practical
                    Q[state][action] = Q[state][action] + (LEARNING_RATE * (accumulated_return - Q[state][action]))

                    value_delta = max(value_delta, np.abs(previous_q_value - Q[state][action]))

                    # Policy Improvement
                    best_action = None
                    max_value = float('-inf')
                    for action, value in Q[state].items():
                        if value > max_value:
                            max_value = value
                            best_action = action
                    Policies[state] = best_action

                    # add state action pair to visited states list
                    visited_states.add(state_action)

            value_deltas.append(value_delta)
            value_deltas_mean.append(np.array(value_deltas).mean())

        final_policies.append(Policies)

        # plt.plot(value_deltas)
        # plt.show()
        #
        # plt.plot(value_deltas_mean)
        # plt.show()

        # Compute final Values of each state
        Values = {}
        for state in Policies.keys():
            max_value = float('-inf')
            for action, value in Q[state].items():
                if value > max_value:
                    max_value = value
            Values[state] = max_value

        print("\n\n\n\n\n")
        print("\nFinal State Values")
        print_rewards(Values, environment)

        print("\nFinal State Policies")
        print_policies(Policies, environment)

    policy00_count = 0
    policy01_count = 0
    policy02_count = 0
    policy12_count = 0
    policy13_count = 0

    for final_policy in final_policies:
        if final_policy.get((0, 0)) == 'RIGHT':
            policy00_count += 1
        if final_policy.get((0, 1)) == 'RIGHT':
            policy01_count += 1
        if final_policy.get((0, 2)) == 'DOWN' or final_policy.get((0, 2)) == 'RIGHT':
            policy02_count += 1
        if final_policy.get((1, 2)) == 'RIGHT':
            policy12_count += 1
        if final_policy.get((1, 3)) == 'DOWN':
            policy13_count += 1

    print("\n\n\n\n\n")
    print("Optimal Policy Average")
    print("Policy 0,0 (RIGHT): " + str(policy00_count/100))
    print("Policy 0,1 (RIGHT): " + str(policy01_count/100))
    print("Policy 0,2 (DOWN): " + str(policy02_count/100))
    print("Policy 1,2 (RIGHT): " + str(policy12_count/100))
    print("Policy 1,3 (DOWN): " + str(policy13_count/100))







