import numpy as np
from PIL import Image  
import cv2  
import matplotlib.pyplot as plt  
import pickle  
from matplotlib import style  
import time 
import tensorflow as tf
from gym import Env
from gym.spaces import Discrete, Box
from IPython.display import clear_output

style.use("ggplot")

SIZE = 20
HM_EPISODES = 25000
ENEMY_PENALTY = 300  
FOOD_REWARD = 25 
epsilon = 0.25  # randomness
EPS_DECAY = 0.995  # How fast/slow we want to perform random actions
SHOW_EVERY = 1000
MOVE_PENALTY = 1

start_q_table = None 

LEARNING_RATE = 0.1
DISCOUNT = 0.95

Attacker_N = 1  # Attacker key in dict
Target_N = 2  # Target key in dict
Defender_N = 3  # Defender key in dict

d = {1: (0, 0, 255),  # Attacker (red)
     2: (0, 255, 0),  # Target (green)
     3: (255, 175, 0)}  # Defender (blue)

class AdversarialEnv(Env):
    def __init__(self):
      # Create the edge parameters for our Box Environmnet
      self.x_min = 0
      self.x_max = 20
      self.y_min = 0
      self.y_max = 20

      # Create our action space such that there is 8 movements (this would be all surrounding boxes to the current box)
      self.action_space = Discrete(8)

      #Create the Observation Space using our provided edge parameters
      self.observation_space = Box(low=np.array([self.x_min, self.y_min]), high=np.array([self.x_max, self.y_max]))

      self.render_obs = []

    def reset(self):

        self.render_obs = []
        #Define our attacker, defender and target
        self.attacker = Attacker()
        self.defender = Defender()
        self.target = Target()

        self.state = np.array([self.attacker.x, self.attacker.y, self.defender.x, self.defender.y, self.target.x, self.target.y])

        self.render_obs.append([self.attacker.x, self.attacker.y, self.defender.y, self.defender.y, self.target.x, self.target.y])

        return self.state

    def step(self, action):
        # This will be where we will have to implement 2 different step functions (1: Attacker, 2: Defender)
        if action == 0:
            new_obs = self.move(x=0, y=1)
        elif action == 1:
            new_obs = self.move(x=1, y=0)
        elif action == 2:
            new_obs = self.move(x=1, y=1)
        elif action == 3:
            new_obs = self.move(x=-1, y=0)
        elif action == 4:
            new_obs = self.move(x=-1, y=1)
        elif action == 5:
            new_obs = self.move(x=-1, y=-1)
        elif action == 6:
            new_obs = self.move(x=1, y=-1)
        elif action == 7:
            new_obs = self.move(x=0, y=-1)
        
        self.state = np.array([self.attacker.x, self.attacker.y, self.defender.x, self.defender.y, self.target.x, self.target.y])

        self.render_obs.append([self.attacker.x, self.attacker.y, self.defender.y, self.defender.y, self.target.x, self.target.y])

        return self.state, -1, False, None

    
    def move(self, x=False, y=False):
        # If no value for x, move randomly
        if not x:
            self.attacker.x += np.random.randint(-1, 2)
        else:
            self.attacker.x += x
            self.defender.x += np.random.randint(-1, 2)
        # If no value for y, move randomly
        if not y:
            self.attacker.y += np.random.randint(-1, 2)
        else:
            self.attacker.y += y
            self.defender.y += np.random.randint(-1, 2)

        # If we are out of bounds, fix!
        if self.attacker.x < 0:
            self.attacker.x = 0
        elif self.attacker.x > SIZE-1:
            self.attacker.x = SIZE-1
        if self.defender.x < 0:
            self.defender.x = 0
        elif self.defender.x > SIZE-1:
            self.defender.x = SIZE-1
        if self.attacker.y < 0:
            self.attacker.y = 0
        elif self.attacker.y > SIZE-1:
            self.attacker.y = SIZE-1
        if self.defender.y < 0:
            self.defender.y = 0
        elif self.defender.y > SIZE-1:
            self.defender.y = SIZE-1

    def render(self):
        #This creates a single frame
        video = np.zeros((SIZE, SIZE, 3), dtype=np.uint8) 
        video[int(self.target.x)][int(self.target.y)] = d[Target_N]  
        video[int(self.attacker.x)][int(self.attacker.y)] = d[Attacker_N]  
        video[int(self.defender.x)][int(self.defender.y)] = d[Defender_N]  
        img = Image.fromarray(video, 'RGB')  
        clear_output(wait=True)
        plt.imshow(img)  
        plt.pause(0.05)
        plt.show()
        #cv2.imshow("image", np.array(img))  # show it!

class Attacker():
    def __init__(self):
        # Set the initial position to (0, 0)
        self.x = 0
        self.y = 0
    
    def step(self, action):
        pass

class Defender():
    def __init__(self):
        # Set the initial position to (10, 10)
        self.x = 10
        self.y = 10
    
    def step(self, action):
        pass

class Target():
    def __init__(self):
        # Set the initial position to (19, 19)
        self.x = 19
        self.y = 19

if __name__ == '__main__':
    env = AdversarialEnv()

    episode_rewards = []

    for episode in range(HM_EPISODES):
        
        if episode % SHOW_EVERY == 0:
            print(f"on #{episode}, epsilon is {epsilon}")
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
            show = True
        else:
            show = False
        
        done = False
        episode_reward = 0

        obs = env.reset()

        step = 0
        while not done:

            action = np.random.randint(0, 7)
            # Take the action!
    
            new_obs, reward, done, _ = env.step(action)     
            # ^This will have to be something like 
            # attacker_obs, attacker_reward, done, _ = env.attacker.step(action)  
            # defender_obs, defender_reward, done, _ = env.defender.step(action) 

            if env.attacker == env.defender:
                reward = -ENEMY_PENALTY

            elif env.attacker == env.target:
                reward = FOOD_REWARD

            else:
                reward = -MOVE_PENALTY
            episode_reward += reward
            env.render()
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                break

            if step >= 200:
                break
            step += 1
        episode_rewards.append(episode_reward)
        epsilon *= EPS_DECAY

    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"Reward {SHOW_EVERY}ma")
    plt.xlabel("episode #")
    plt.show()

