import numpy as np
from PIL import Image  
import cv2  
import matplotlib.pyplot as plt  
import time 
import tensorflow as tf
from gym import Env
from gym.spaces import Discrete, Box
from IPython.display import clear_output
#pygame


epsilon = 0.25  # randomness
EPS_DECAY = 0.995  # How fast/slow we want to perform random actions

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
        self.attacker.step(action)
        #if action < 5:
        #    self.defender.step(action+4)
        #else:
        #    self.defender.step(action-4)
        
        attacker_reward = self.attacker.reward_function([self.defender.x, self.defender.y], [self.target.x, self.target.y])

        defender_reward = self.defender.reward_function([self.attacker.x, self.attacker.y])

        self.state = np.array([self.attacker.x, self.attacker.y, self.defender.x, self.defender.y, self.target.x, self.target.y])

        self.render_obs.append([self.attacker.x, self.attacker.y, self.defender.y, self.defender.y, self.target.x, self.target.y])

        return self.state, attacker_reward, False, None

    
    def render(self):
        #This creates a single frame
        video = np.zeros((20, 20, 3), dtype=np.uint8) 
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
        if action == 0:
            _ = self.move(x=0, y=1)
        elif action == 1:
            _ = self.move(x=1, y=0)
        elif action == 2:
            _ = self.move(x=1, y=1)
        elif action == 3:
            _ = self.move(x=-1, y=0)
        elif action == 4:
            _ = self.move(x=-1, y=1)
        elif action == 5:
            _ = self.move(x=-1, y=-1)
        elif action == 6:
            _ = self.move(x=1, y=-1)
        elif action == 7:
            _ = self.move(x=0, y=-1)

        return True

    def move(self, x=None,  y=None):
        if x == None:
            self.x = np.random.randint(-1, 1)
        else:
            self.x += x

        if y == None:
            self.y = np.random.randint(-1, 1)
        else:
            self.y += y
        
        if self.x < 0:
            self.x = 0
        elif self.x > 19:
            self.x = 19

        if self.y < 0:
            self.y = 0
        elif self.y > 19:
            self.y = 19

        return True
        
    def reward_function(self, defender, goal):
        reward = 0

        if (defender[0]-self.x + defender[1]-self.y) < 10:
            reward = -1
        elif (defender[0]-self.x + defender[1]-self.y) < 3:
            reward = -3
        elif (defender[0]-self.x + defender[1]-self.y) < 0:
            reward = -300
            print('Reached Terminal State, the Denfeder got the Attacker')
            print(reward)

        if (goal[0]-self.x + goal[1]-self.y) < 10:
            reward += 1
        elif (goal[0]-self.x + goal[1]-self.y) < 3:
            reward += 5
        elif (goal[0]-self.x + goal[1]-self.y) < 0:
            reward += 300
            print('Reached Terminal State, the Attacker got the Goal!!!!!!')
            print(reward)

        return reward

    

class Defender():
    def __init__(self):
        # Set the initial position to (10, 10)
        self.x = 10
        self.y = 10
    
    def step(self, action):
        if action == 0:
            _ = self.move(x=0, y=1)
        elif action == 1:
            _ = self.move(x=1, y=0)
        elif action == 2:
            _ = self.move(x=1, y=1)
        elif action == 3:
            _ = self.move(x=-1, y=0)
        elif action == 4:
            _ = self.move(x=-1, y=1)
        elif action == 5:
            _ = self.move(x=-1, y=-1)
        elif action == 6:
            _ = self.move(x=1, y=-1)
        elif action == 7:
            _ = self.move(x=0, y=-1)

        return True

    def move(self, x=None,  y=None):
        if x == None:
            self.x = np.random.randint(-1, 1)
        else:
            self.x += x

        if y == None:
            self.y = np.random.randint(-1, 1)
        else:
            self.y += y
        
        if self.x < 0:
            self.x = 0
        elif self.x > 19:
            self.x = 19

        if self.y < 0:
            self.y = 0
        elif self.y > 19:
            self.y = 19

        return True

    def reward_function(self, attacker):
        reward = 0
        if (attacker[0]-self.x + attacker[1]-self.y) < 10:
            reward += 1
        elif (attacker[0]-self.x + attacker[1]-self.y) < 3:
            reward += 5
        elif (attacker[0]-self.x + attacker[1]-self.y) < 0:
            reward += 300
            print(reward)
        return reward

class Target():
    def __init__(self):
        # Set the initial position to (19, 19)
        self.x = 19
        self.y = 19

if __name__ == '__main__':
    env = AdversarialEnv()

    episode_rewards = []

    for episode in range(25):
        
        obs = env.reset()
        episode_reward = 0

        step = 0
        while step <= 2000:

            action = np.random.randint(0, 7)
            # Take the action!
    
            new_obs, reward, done, _ = env.step(action)     
            # ^This will have to be something like 
            # attacker_obs, attacker_reward, done, _ = env.attacker.step(action)  
            # defender_obs, defender_reward, done, _ = env.defender.step(action) 
            #env.render()

            episode_reward += reward

            step += 1

        print('Episode Reward: {}'.format(episode_reward))