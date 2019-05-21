import numpy as np
import matplotlib.pyplot as plt
from unwrapped_frozen_lake_environment import frozen_lake_v0, frozen_lake_v0_negative_reward_steps

GAMMA = 0.9
LEARNING_RATE = 0.1
ACTION_LIST = ("LEFT", "DOWN", "RIGHT", "UP")

def max_dict(d):
  max_key = None
  max_val = float('-inf')
  for k,v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val


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


if __name__ == "__main__":
    final_policies = []
    for i in range(100):
        environment = frozen_lake_v0_negative_reward_steps()
        print("\nFrozen Lake Reward Structure")
        print_rewards(environment.rewards, environment)

        # initialize Q table. All Q table values will be set to 0 instead of random values
        Q = {}
        states = environment.get_all_defined_states()
        for state in states:
            Q[state] = {}
            for action in ACTION_LIST:
                Q[state][action] = 0

        # create dictionaries to keep track of how many times Q gets updated
        q_state_action_update_counts = {}
        q_update_counts = {}
        for state in states:
            q_state_action_update_counts[state] = {}
            for action in ACTION_LIST:
                q_state_action_update_counts[state][action] = 1.0

        # create lists to store the difference between the previous and current
        # Q values and the means of these differences
        value_deltas = []
        value_deltas_mean = []

        # define variable for epsilon decay
        # at first we want all actions to be random, but as we play more episodes the actions should get more greedy based
        # on the experience gained
        random_action_probability = 1.0

        # set a defined number of episodes rather than just going until converged
        for i in range(10000):
            if i % 100 == 0:
                random_action_probability += 10e-3
            if i % 500 == 0:
                print("Training Episode: " + str(i))

            start_state = (0, 0)
            environment.set_state(start_state)

            # get best action
            action = max_dict(Q[start_state])[0]

            value_delta = 0

            while not environment.game_over():
                # for each step in the game select and execute an action and get the action of the new state
                if np.random.random() < 0.5/random_action_probability:
                    action = np.random.choice(ACTION_LIST)

                reward = environment.step(action)
                new_state = environment.current_state()

                new_action = max_dict(Q[new_state])[0]

                if np.random.random() < 0.5/random_action_probability:
                    new_action = np.random.choice(ACTION_LIST)

                # decay the learning rate as a function of the number a state has had its Q value updated
                # this makes the learning rate dynamic for each specific state
                learning_rate = LEARNING_RATE / q_state_action_update_counts[state][action]
                q_state_action_update_counts[state][action] += 0.005

                q_previous = Q[state][action]

                new_action, new_max_value = max_dict(Q[new_state])

                Q[state][action] = Q[state][action] + LEARNING_RATE * (reward + GAMMA*new_max_value - Q[state][action])
                value_delta = max(value_delta, np.abs(q_previous - Q[state][action]))

                q_update_counts[state] = q_update_counts.get(state, 0) + 1

                state = new_state
                action = new_action

            value_deltas.append(value_delta)
            value_deltas_mean.append(np.array(value_deltas).mean())

        # plt.plot(value_deltas)
        # plt.show()
        #
        # plt.plot(value_deltas_mean)
        # plt.show()

        # Compute final Values and Actions of each state
        Policies = {}
        Values = {}
        for s in environment.actions.keys():
            a, max_q = max_dict(Q[s])
            Policies[s] = a
            Values[s] = max_q

        final_policies.append(Policies)

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
    print("Policy 0,0 (RIGHT): " + str(policy00_count / 100))
    print("Policy 0,1 (RIGHT): " + str(policy01_count / 100))
    print("Policy 0,2 (DOWN): " + str(policy02_count / 100))
    print("Policy 1,2 (RIGHT): " + str(policy12_count / 100))
    print("Policy 1,3 (DOWN): " + str(policy13_count / 100))