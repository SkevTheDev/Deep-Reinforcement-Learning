import atari_environment_wrappers
import atari_experience_replay_buffer
import atari_deep_q_neural_network
import atari_agent

import argparse
import numpy as np

import torch
import torch.optim as optim
import time

from tensorboardX import SummaryWriter

# Define Basic Hyperparameters for DQN
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 18.0
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = .0001
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

# Define Random Action Probability and Decay Schedule
EPSILON_DECAY_LAST_FRAME = 100000
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

# Define Hyperparameters for Priority Experience Replay Buffer
PRIORITY_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

if __name__ == "__main__":
    # define arguments that can be passed in through the command line to give operator control to following...
    # 1) GPU utilization enabling
    # 2) Environment Selection
    # 3) Minimum reward for successfully converging to a "win state"
    # 4) Dueling Convolutional Network enabling
    # 5) Priority Experience Replay enabling
    # 6) Double Q Learning enabling
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    parser.add_argument("--dueling", default=False, action="store_true",
                        help="Enable Dueling Deep Q Network")
    parser.add_argument("--priority_replay_buffer", default=False, action="store_true",
                        help="Enable Priority Replay Buffer")
    parser.add_argument("--double", default=False, action="store_true", help="Enable Double DQN")
    args = parser.parse_args()

    # select the processing device
    device = torch.device("cuda" if args.cuda else "cpu")

    # setup game environment with defined wrappers
    env = atari_environment_wrappers.generate_game_environment(args.env)

    # Select Convolutional Network Architecture for learning
    if args.dueling:
        net = atari_deep_q_neural_network.DuelingDeepQNetwork(env.observation_space.shape, env.action_space.n).to(device)
        tgt_net = atari_deep_q_neural_network.DuelingDeepQNetwork(env.observation_space.shape, env.action_space.n).to(device)
    else:
        net = atari_deep_q_neural_network.BasicDeepQNetwork(env.observation_space.shape, env.action_space.n).to(device)
        tgt_net = atari_deep_q_neural_network.BasicDeepQNetwork(env.observation_space.shape, env.action_space.n).to(device)

    # print network architecture
    print(net)

    # define TensorBoard data stream
    writer = SummaryWriter(comment="-" + args.env)

    # define experience replay buffer architecture
    if args.priority_replay_buffer:
        buffer = atari_experience_replay_buffer.PriorityExperienceReplayBuffer(REPLAY_SIZE)
    else:
        buffer = atari_experience_replay_buffer.BasicExperienceReplayBuffer(REPLAY_SIZE)

    # define our agent which takes parameters of the environment and the buffer
    agent = atari_agent.Agent(env, buffer)

    # Define SGD optimizer using ADAM as it is understood to have slightly better performance then RMSProp
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # define array of total rewards observed
    total_rewards = []

    # value to keep track of the number of frames(observations) need to be processed for the model to converge on
    # a solution
    num_frames = 0
    prev_num_frames = 0
    # Define random action probability
    epsilon = EPSILON_START
    # define beta value that will help normalize updating network weights for high priority sample observations
    beta = BETA_START
    eval_states = None

    # timing metric to compute how many frames can be processed a second
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    # timing metric for total training time
    start_time = time.time()

    # training will complete once the average total reward exceeds the defined threshold
    while True:
        # each iteration of the loop processes a single frame/state
        num_frames += 1

        # adjust epsilon and beta in accordance with there update schedules
        epsilon = max(EPSILON_FINAL, EPSILON_START - num_frames / EPSILON_DECAY_LAST_FRAME)
        beta = min(1.0, BETA_START + num_frames * (1.0 - BETA_START) / BETA_FRAMES)

        # have the agent execute an action and receive a reward
        reward = agent.play_step(net, epsilon, device=device)

        # if the agent has reached a terminal state of the game
        if reward is not None:

            # add total reward of completed game to rewards array
            total_rewards.append(reward)

            # compute processing speed
            speed = (num_frames - ts_frame) / (time.time() - ts)
            ts_frame = num_frames
            ts = time.time()

            # compute the average reward for the last 100 observations
            mean_reward = np.mean(total_rewards[-100:])

            # print game metrics to console
            print("\nTotal States/Frames Processed: " + str(num_frames) + "\nGames Completed: " + str(len(total_rewards)) +
                  "\nTotal States/Frames Processed This Game: " + str(num_frames - prev_num_frames) +
                  "\nLast 100 Games Average Total Reward: " + str(round(mean_reward, 2)) + "\nRandom Action Probability: " +
                  str(round(epsilon, 2)) + "\nProcessing Speed: " + str(round(speed, 2)) + " f/s")

            # update prev num frames
            prev_num_frames = num_frames

            # write metrics to TensorBoard
            writer.add_scalar("epsilon", epsilon, num_frames)
            writer.add_scalar("speed", speed, num_frames)
            writer.add_scalar("reward_100", mean_reward, num_frames)
            writer.add_scalar("reward", reward, num_frames)

            # Save model to file if our last 100 game average has improved
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("\n\nBest average reward updated from %.3f to %.3f. Network Model Saved!\n\n" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward

            # training will complete once the average total reward exceeds the defined threshold
            if mean_reward > args.reward:
                end_time = time.time()
                print("Solved in %d frames!" % num_frames)
                print("Training Time: " + str(end_time - start_time))
                break

        # if the experience replay buffer is not sufficiently full enough to draw samples from, skip over computing the
        # loss and back propagation of updating the weights and have the agent take more random actions given a state
        if len(buffer) < REPLAY_START_SIZE:
            continue

        if num_frames % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        # zero out all network gradients prior to computing loss
        optimizer.zero_grad()

        # compute loss of q and q target by sampling
        if args.priority_replay_buffer:
            # sample priority batch
            batch, batch_indices, batch_weights = buffer.sample(BATCH_SIZE, beta)
            values_loss, sample_priority_values = atari_deep_q_neural_network.priority_calc_loss(batch, batch_weights, net, tgt_net,
                                                                                    GAMMA, double=args.double,
                                                                                    device=device)
        else:
            # sample random batch
            batch = buffer.sample(BATCH_SIZE)
            values_loss = atari_deep_q_neural_network.basic_calc_loss(batch, net, tgt_net, GAMMA, double=args.double,
                                                                 device=device)

        # compute gradients for the network based on the computed loss
        values_loss.backward()

        # update weights using ADAM optimizer
        optimizer.step()

        # update sample priorities
        if args.priority_replay_buffer:
            buffer.update_priorities(batch_indices, sample_priority_values.data.cpu().numpy())

    # close TensorBoard writing session
    writer.close()

