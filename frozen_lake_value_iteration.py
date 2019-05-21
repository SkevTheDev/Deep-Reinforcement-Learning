import numpy as np
from unwrapped_frozen_lake_environment import frozen_lake_v0, frozen_lake_v0_negative_reward_steps

import matplotlib.pyplot as plt

VALUE_CONVERGENCE_THRESHOLD = 10e-4
GAMMA = 0.9
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


if __name__ == '__main__':
    iteration_list = np.zeros(10000)
    final_policies = []
    for i in range(10000):
        # initialize the environment and ensure the reward structure is defined correctly
        # this initialization also sets up the state transition probabilities which is needed for value iteration
        environment = frozen_lake_v0_negative_reward_steps(random_action=0.2, hole_reward=-1)
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

        # initialize value dictionary
        Values = {}
        # get all states that have either a defined action system or reward structure and merge them into the same set
        states = environment.get_all_defined_states()

        # for each state with a defined action system, initialize a random value for that state
        # terminal states get defined with a value of 0
        for state in states:
            if state in environment.actions:
                Values[state] = np.random.random()
            else:
                Values[state] = 0

        # print random values
        print("\nRandom State Values")
        print_rewards(Values, environment)

        print("\n\n\n\n\n")

        print("Policy Evaluation and Improvement Start")

        # iteration counter
        iterations = 0

        value_deltas = []
        value_deltas_mean = []

        # Policy Evaluation Algorithm
        # for each state in the environment execute each available action
        # and compute the value of executing that action in that state
        # set the new value of that state to the largest value computed by a specific action
        while True:
            print(".")
            iterations += 1
            # value_delta keeps track of difference changes in previous values and newly computed values
            # this delta gets carried from state to state because we want to keep track of the max computed delta
            # until that value falls below the convergence threshold
            # computed using iterative value evaluation
            value_delta = 0
            for state in states:
                previous_value = Values[state]

                if state in Policies:
                    # initialize a new value for the state that is very small
                    new_value = float('-inf')
                    best_action = None

                    for action in ACTION_LIST:
                        environment.set_state(state)
                        reward = environment.step(action)
                        next_state = environment.current_state()
                        temp_value = 0.8 * (reward + GAMMA * Values[next_state])
                        if temp_value > new_value:
                            new_value = temp_value
                            # the action that produces the highest value is the best action to take
                            best_action = action
                    Values[state] = new_value
                    # Policy Improvement
                    # the best action should be our policy for that state
                    Policies[state] = best_action
                    # we should use the maximum value difference between the previous delta and newly computed delta
                    # to determine if we have converged
                    value_delta = max(value_delta, np.abs(previous_value - Values[state]))
            # if the difference between the previous value of a state and the new value of a state is less than the
            # threshold the value function has converged
            if value_delta < VALUE_CONVERGENCE_THRESHOLD:
                print("\nConverged! Num Iterations: " + str(iterations))
                break

            value_deltas.append(value_delta)
            value_deltas_mean.append(np.array(value_deltas).mean())

        plt.plot(value_deltas)
        plt.show()

        plt.plot(value_deltas_mean)
        plt.show()

        print("\n\n\n\n\n")
        print("\nFinal State Values")
        print_rewards(Values, environment)

        print("\nFinal State Policies")
        print_policies(Policies, environment)

        iteration_list[i] = iterations
        final_policies.append(Policies)

    iteration_mean = iteration_list.mean()
    print("\n\n\n\n\n")
    print("Average Number of Iterations to Converge")
    print(iteration_mean)
    print("\n\n\n\n\n")

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
        if final_policy.get((0, 2)) == 'DOWN':
            policy02_count += 1
        if final_policy.get((1, 2)) == 'RIGHT':
            policy12_count += 1
        if final_policy.get((1, 3)) == 'DOWN':
            policy13_count += 1

    print("Optimal Policy Average")
    print("Policy 0,0 (RIGHT): " + str(policy00_count/10000))
    print("Policy 0,1 (RIGHT): " + str(policy01_count/10000))
    print("Policy 0,2 (DOWN): " + str(policy02_count/10000))
    print("Policy 1,2 (RIGHT): " + str(policy12_count/10000))
    print("Policy 1,3 (DOWN): " + str(policy13_count/10000))
