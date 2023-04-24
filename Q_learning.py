import numpy as np
from env import AdversarialEnv
# Define the environment
# Git test 2

env = AdversarialEnv()
attempts = 10
max_steps = 1000
n_states = 400  # Number of states
n_actions = env.action_space.n  # Number of actions (0 for "left", 1 for "right")

print(n_actions, n_states)
gamma = 0.99  # Discount factor
lr = 0.01  # Learning rate
epsilon = 0.1  # Epsilon for epsilon-greedy exploration
q_learn = True
EPS_DECAY = 0.9998


SIZE = 20
q_table_attacker = {}
for i in range(-SIZE+1, SIZE):
    for ii in range(-SIZE+1, SIZE):
        q_table_attacker[(i, ii)] = [0 for i in range(8)]

q_table_defender = {}
for i in range(-SIZE+1, SIZE):
    for ii in range(-SIZE+1, SIZE):
        q_table_defender[(i, ii)] = [0 for i in range(8)]


env = AdversarialEnv()

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
            # GET THE ACTION
            action1 = np.argmax(q_table_attacker[attacker])
            action2 = np.argmax(q_table_defender[defender])
        else:
            action1 = np.random.randint(0, 8)
            action2 = np.random.randint(0, 8)

        #print(action)
        env.render()
        
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
        env.plt_counter = step #this is for the matplot
        obs = new_obs
        
        #print(episode_reward)
        episode_rewards1.append(episode_reward1)
        episode_rewards2.append(episode_reward2)
    epsilon *= EPS_DECAY


#print(episode_rewards)

env.print_episode_rewards(episode_rewards)

