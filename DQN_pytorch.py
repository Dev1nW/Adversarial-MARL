# I asked Chat GPT for tips and it said to try this 

import numpy as np
from env import AdversarialEnv
import torch
import random

import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


num_W = 0
num_L = 0
num_T = 0

# Define the environment
env = AdversarialEnv()
attempts = 500
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# DQN parameters
gamma = 0.99
alpha = 0.01
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
batch_size = 32
buffer_size = 10000
update_target_frequency = 1000

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# DQN setup
# policy_net = DQN(n_states, n_actions).to(device)
# target_net = DQN(n_states, n_actions).to(device)
hidden_size = 6
policy_net1 = DQN(6, hidden_size, n_actions).to(device)
target_net1 = DQN(6, hidden_size, n_actions).to(device)

optimizer1 = optim.Adam(policy_net1.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

target_net1.load_state_dict(policy_net1.state_dict())
target_net1.eval()

policy_net2 = DQN(6, hidden_size, n_actions).to(device)
target_net2 = DQN(6, hidden_size, n_actions).to(device)

optimizer2 = optim.Adam(policy_net2.parameters(), lr=0.003)

target_net2.load_state_dict(policy_net2.state_dict())
target_net2.eval()

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(buffer_size)

# Update the target network
def update_target(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())

# Training loop
episode_rewards = []

for episode in range(attempts):
    obs = env.reset()
    episode_reward = 0
    done = False
    steps = 0

    while not done:
        #env.render()

        state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action1 = policy_net1(state_tensor).argmax(dim=1).item()
            action2 = policy_net2(state_tensor).argmax(dim=1).item()
        if np.random.random() < epsilon:
            action1 = np.random.randint(0, n_actions)
            action2 = np.random.randint(0, n_actions)

        new_obs, reward, done, _ = env.step(action1, action2)
        reward -= 1
        # Save the transition to the replay buffer
        replay_buffer.push(obs, action1, reward, new_obs, done)

        obs = new_obs
        episode_reward += reward
        steps += 1
        env.plt_counter = steps

        if len(replay_buffer) >= batch_size:
            # Sample a batch of transitions from the replay buffer
            batch = replay_buffer.sample(batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            state_batch = np.array(state_batch)
            action_batch = np.array(action_batch)
            reward_batch = np.array(reward_batch)
            next_state_batch = np.array(next_state_batch)
            done_batch = np.array(done_batch)

            state_batch = torch.tensor(state_batch, dtype=torch.float32, device=device)
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=device).unsqueeze(1)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
            next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=device)
            done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device)

            # Compute the Q-values of the next states
            with torch.no_grad():
                next_state_values1 = target_net1(next_state_batch).max(1)[0]
                next_state_values2 = target_net1(next_state_batch).max(1)[0]

            # Compute the target Q-values
            target_q_values1 = reward_batch + gamma * next_state_values1 * (1 - done_batch)
            target_q_values2 = reward_batch + gamma * next_state_values2 * (1 - done_batch)

            # Compute the predicted Q-values
            predicted_q_values1 = policy_net1(state_batch).gather(1, action_batch).squeeze()
            predicted_q_values2 = policy_net2(state_batch).gather(1, action_batch).squeeze()

            # Compute the loss
            loss1 = loss_fn(predicted_q_values1, target_q_values1)
            loss2 = loss_fn(predicted_q_values2, target_q_values2)

            # Optimize the policy network
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            # Update the target network
            if steps % update_target_frequency == 0:
                update_target(policy_net1, target_net1)
                update_target(policy_net2, target_net2)

        if steps >= 100:
            done = True

        if done:
            if obs[0] == obs[2] and obs[1] == obs[3]:
                num_L += 1
            elif obs[0] == obs[4] and obs[1] == obs[5]:
                num_W += 1
            else:
                num_T += 1

    episode_rewards.append(episode_reward)

    # Decay epsilon for epsilon-greedy exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode}: Reward {episode_reward}")

env.print_episode_rewards(episode_rewards)

