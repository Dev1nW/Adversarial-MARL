import numpy as np
from env import AdversarialEnv
import torch
import random
import matplotlib.pyplot as plt
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
attempts = 1000
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# DQN parameters
gamma = 0.95 # Discount Factor (Larger = care more about future reward)
alpha = 0.018 # Attacker Learning Rate
epsilon = 0.1 # Randomization 
epsilon_min = 0.01
epsilon_decay = 0.2
batch_size = 32
buffer_size = 10000
update_target_frequency = 1000

def_alpha = 0.000001 # Defender Learning Rate

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
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

optimizer2 = optim.Adam(policy_net2.parameters(), lr=def_alpha)

target_net2.load_state_dict(policy_net2.state_dict())
target_net2.eval()

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action1, action2, reward1, reward2, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action1, action2, reward1, reward2, next_state, done)
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
    episode_reward1 = 0
    episode_reward2 = 0
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

        new_obs, reward1, reward2, done, _ = env.step(action1, action2)
        reward1 -= 1
        reward2 -= 1
        # Save the transition to the replay buffer
        replay_buffer.push(obs, action1, action2, reward1, reward2, new_obs, done)

        obs = new_obs
        episode_reward1 += reward1
        episode_reward2 += reward2
        steps += 1
        env.plt_counter = steps

        if len(replay_buffer) >= batch_size:
            # Sample a batch of transitions from the replay buffer
            batch = replay_buffer.sample(batch_size)
            state_batch, action_batch1, action_batch2, reward_batch1, reward_batch2, next_state_batch, done_batch = zip(*batch)
            state_batch = np.array(state_batch)
            action_batch1 = np.array(action_batch1)
            action_batch2 = np.array(action_batch2)
            reward_batch1 = np.array(reward_batch1)
            reward_batch1 = np.array(reward_batch2)
            next_state_batch = np.array(next_state_batch)
            done_batch = np.array(done_batch)

            state_batch = torch.tensor(state_batch, dtype=torch.float32, device=device)
            action_batch1 = torch.tensor(action_batch1, dtype=torch.long, device=device).unsqueeze(1)
            reward_batch1 = torch.tensor(reward_batch1, dtype=torch.float32, device=device)
            action_batch2 = torch.tensor(action_batch2, dtype=torch.long, device=device).unsqueeze(1)
            reward_batch2 = torch.tensor(reward_batch2, dtype=torch.float32, device=device)
            next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=device)
            done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device)

            # Compute the Q-values of the next states
            with torch.no_grad():
                next_state_values1 = target_net1(next_state_batch).max(1)[0]
                next_state_values2 = target_net1(next_state_batch).max(1)[0]

            # Compute the target Q-values
            target_q_values1 = reward_batch1 + gamma * next_state_values1 * (1 - done_batch)
            target_q_values2 = reward_batch2 + gamma * next_state_values2 * (1 - done_batch)

            # Compute the predicted Q-values
            predicted_q_values1 = policy_net1(state_batch).gather(1, action_batch1).squeeze()
            predicted_q_values2 = policy_net2(state_batch).gather(1, action_batch2).squeeze()

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
                print("Loss")
                num_L += 1
            elif obs[0] == obs[4] and obs[1] == obs[5]:
                print("Win")
                num_W += 1
            else:
                print("Tie")
                num_T += 1

    episode_rewards.append(episode_reward1)

    # Decay epsilon for epsilon-greedy exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    #print(f"Episode {episode}: Reward for Attacker: {episode_reward1}")
    #print(f"Episode {episode}: Reward for Defender: {episode_reward2}")

#env.print_episode_rewards(episode_rewards)

# Plot the episode rewards over time
plt.figure(figsize=(6, 6))  
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Attacker Reward per Episode")
plt.show()

# Plot the bar graph for game results
plt.figure(figsize=(6, 6))  
labels = ['Defender wins', 'Attacker wins', 'Ties']
values = [num_L, num_W, num_T]
colors = ['red', 'blue', 'gray']

plt.bar(labels, values, color=colors)
plt.title('Game Results')
plt.xlabel('Result')
plt.ylabel('Number of Games')
plt.show()
plt.pause(100000)