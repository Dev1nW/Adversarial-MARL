import numpy as np
from env import AdversarialEnv
# Define the environment
# Git test 2

env = AdversarialEnv()

n_states = 400  # Number of states
n_actions = env.action_space.n  # Number of actions (0 for "left", 1 for "right")

print(n_actions, n_states)
gamma = 0.99  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.1  # Epsilon for epsilon-greedy exploration
q_learn = True
EPS_DECAY = 0.9998


SIZE = 20
q_table = {}
for i in range(-SIZE+1, SIZE):
    for ii in range(-SIZE+1, SIZE):
        q_table[(i, ii)] = [1 for i in range(8)]


env = AdversarialEnv()

for episode in range(1000):
    episode_rewards = []
    obs = env.reset()
    episode_reward = 0
    done = False 
    step = 0 
    while not done:
    
        attacker = (obs[0], obs[1])
        
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[attacker])
        else:
            action = np.random.randint(0, 8)

        #print(action)
        #env.render()
        
        # Take the action!
        new_obs, reward, done, _ = env.step(action)
        
        max_future_q = np.max(q_table[(new_obs[0], new_obs[1])])
        current_q = q_table[attacker][action]

        new_q = (1 - 0.001) * current_q + 0.001 * (reward + 0.99 * max_future_q)
        q_table[attacker][action] = new_q

        if step >= 1000:
            break
        episode_reward += reward
        step += 1
        obs = new_obs
        

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
