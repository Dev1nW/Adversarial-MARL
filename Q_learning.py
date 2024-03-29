import numpy as np
from env import AdversarialEnv
import matplotlib.pyplot as plt

# Create the Environment
env = AdversarialEnv()

attempts = 1000 # Number of episodes 
max_steps = 100 # Number of steps per episode
n_states = 100  # Number of states
n_actions = env.action_space.n  # Number of actions in the Environment

gamma = 0.99  # Discount factor
lr = 0.01  # Learning rate
epsilon = 0.1  # Epsilon for epsilon-greedy exploration
EPS_DECAY = 0.9998

# From the Attacker 
num_T = 0 # Number of Ties
num_L = 0 # Number of Losses
num_W = 0 # Number of Wins 

# Size of Grid 
SIZE = 11

# Create a Q Table for the Attacker 
q_table_attacker = {}
for i in range(-SIZE+1, SIZE):
    for ii in range(-SIZE+1, SIZE):
        q_table_attacker[(i, ii)] = [0 for i in range(8)]

# Create a Q Table for the Defender
q_table_defender = {}
for i in range(-SIZE+1, SIZE):
    for ii in range(-SIZE+1, SIZE):
        q_table_defender[(i, ii)] = [0 for i in range(8)]

for episode in range(attempts):
    episode_rewards1 = []
    episode_rewards2 = []
    obs = env.reset()
    episode_reward1 = 0
    episode_reward2 = 0
    done = False 
    step = 0 
    while done == False:
    
        attacker = (obs[0], obs[1])
        defender = (obs[2], obs[3])
        
        if np.random.random() > epsilon:
            action1 = np.argmax(q_table_attacker[attacker])
            action2 = np.argmax(q_table_defender[defender])
        else:
            action1 = np.random.randint(0, 8)
            action2 = np.random.randint(0, 8)

        #env.render()
        
        # Take the action!
        new_obs, reward1, reward2, done, _ = env.step(action1, action2)
        
        max_future_q_attacker = np.max(q_table_attacker[(new_obs[0], new_obs[1])])
        current_q_attacker = q_table_attacker[attacker][action1]

        new_q_attacker = (1 - lr) * current_q_attacker + lr * (reward1 + gamma * max_future_q_attacker)
        q_table_attacker[attacker][action1] = new_q_attacker

        max_future_q_defender = np.max(q_table_defender[(new_obs[2], new_obs[3])])
        current_q_defender = q_table_defender[defender][action2]

        new_q_defender = (1 - lr) * current_q_defender + lr * (reward2 + gamma * max_future_q_defender)
        q_table_defender[defender][action2] = new_q_defender

        if step >= max_steps:
            done = True

        episode_reward1 += reward1
        step += 1
        env.plt_counter = step 
        obs = new_obs

        if done:
            if obs[0] == obs[2] and obs[1] == obs[3]:
                num_L += 1
            elif obs[0] == obs[4] and obs[1] == obs[5]:
                num_W += 1
            else:
                num_T += 1

        episode_rewards1.append(episode_reward1)
        episode_rewards2.append(episode_reward2)
    epsilon *= EPS_DECAY

# Plot the episode rewards over time
plt.figure(figsize=(6, 6))  
plt.plot(episode_rewards1)
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

