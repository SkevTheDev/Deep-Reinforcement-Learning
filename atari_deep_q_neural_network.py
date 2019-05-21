import torch
import torch.nn as nn
import numpy as np


# define convolutional neural network based on the architecture defined in deepminds paper
# "Human-level control through deep reinforcement learning"
# This is a pretty standard architecture minus the fact that there are no max pooling layers, because this could
# downsample important information from the state space and since the state is preprocessed efficiently and
# our network is simple there isnt the need for pooling dimensionality reduction
class BasicDeepQNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(BasicDeepQNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self.process_flattened_convolutional_layer(input_shape)

        self.fully_connected = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def process_flattened_convolutional_layer(self, shape):
        f = self.conv(torch.zeros(1, *shape))
        return int(np.prod(f.size()))

    def forward(self, x):
        flattened_convolutional_layer = self.conv(x).view(x.size()[0], -1)
        return self.fully_connected(flattened_convolutional_layer)


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDeepQNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self.process_flattened_convolutional_layer(input_shape)

        self.fully_connected_advantage_action = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.fully_connected_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def process_flattened_convolutional_layer(self, shape):
        f = self.conv(torch.zeros(1, *shape))
        return int(np.prod(f.size()))

    def forward(self, x):
        flattened_convolutional_layer = self.conv(x).view(x.size()[0], -1)
        value = self.fully_connected_value(flattened_convolutional_layer)
        advantage_action = self.fully_connected_advantage_action(flattened_convolutional_layer)
        return value + advantage_action - advantage_action.mean()


def basic_calc_loss(batch, net, tgt_net, gamma, double=True, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # if using double q learning get values from target network
    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def priority_calc_loss(batch, batch_weights, net, tgt_net, gamma, double=True,device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # if using double q learning get values from target network
    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean(), losses_v + 1e-5


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.new_state is None)
        if exp.new_state is None:
            last_states.append(state)
        else:
            last_states.append(np.array(exp.new_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)
