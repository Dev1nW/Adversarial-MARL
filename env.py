import numpy as np
from PIL import Image  
import cv2  
import matplotlib.pyplot as plt  
import time 
from gym import Env
from gym.spaces import Discrete, Box
from IPython.display import clear_output
import math
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
      self.x_max = 10
      self.y_min = 0
      self.y_max = 10

      # Create our action space such that there is 8 movements (this would be all surrounding boxes to the current box)
      self.action_space = Discrete(8)

      #Create the Observation Space using our provided edge parameters
      self.observation_space = Box(low=np.array([self.x_min, self.y_min]), high=np.array([self.x_max, self.y_max]))

      self.render_obs = []
      
      self.initial_diff_defender = 0
      self.initial_diff_goal = 0

    def reset(self):

        self.render_obs = []
        #Define our attacker, defender and target
        self.attacker = Attacker()
        self.defender = Defender()
        self.target = Target()

        self.state = np.array([self.attacker.x, self.attacker.y, self.defender.x, self.defender.y, self.target.x, self.target.y])

        #self.render_obs.append([self.attacker.x, self.attacker.y, self.defender.y, self.defender.y, self.target.x, self.target.y])
        self.render_obs.append([self.attacker.x, self.attacker.y, self.defender.x, self.defender.y, self.target.x, self.target.y]) #fixed a typo for self.defender.x


        return self.state

    def step(self, action):
        # This will be where we will have to implement 2 different step functions (1: Attacker, 2: Defender)
        out_of_bounds_flag = self.attacker.step(action)
        #if action < 5:
        #    self.defender.step(action+4)
        #else:
        #    self.defender.step(action-4)
        
        attacker_reward = self.attacker.reward_function([self.defender.x, self.defender.y], [self.target.x, self.target.y])

        if out_of_bounds_flag:
            attacker_reward = -10

        defender_reward = self.defender.reward_function([self.attacker.x, self.attacker.y])

        self.state = np.array([self.attacker.x, self.attacker.y, self.defender.x, self.defender.y, self.target.x, self.target.y])

        self.render_obs.append([self.attacker.x, self.attacker.y, self.defender.y, self.defender.y, self.target.x, self.target.y])

        return self.state, attacker_reward, False, None

    
    def render(self):
        #This creates a single frame
        video = np.zeros((11, 11, 3), dtype=np.uint8) 
        video[int(self.target.x)][int(self.target.y)] = d[Target_N]  
        video[int(self.attacker.x)][int(self.attacker.y)] = d[Attacker_N]  
        video[int(self.defender.x)][int(self.defender.y)] = d[Defender_N]  
        img = Image.fromarray(video, 'RGB')  
        clear_output(wait=True)
        plt.imshow(img)  
        plt.pause(0.05)
        plt.show()

class Attacker():
    def __init__(self):
        # Set the initial position to (0, 0)
        self.x = 0
        self.y = 0
    
    def step(self, action):
        if action == 0:
            self.x += 0 
            self.y += 1
        elif action == 1:
            self.x += 1 
            self.y += 0
        elif action == 2:
            self.x += 1
            self.y += 1
        elif action == 3:
            self.x -= 1 
            self.y += 0
        elif action == 4:
            self.x -= 1 
            self.y += 1
        elif action == 5:
            self.x -= 1 
            self.y -= 1
        elif action == 6:
            self.x += 1 
            self.y -= 1
        elif action == 7:
            self.x -= 0 
            self.y -= 1

        if self.x < 0 or self.x > 10 or self.y < 0 or self.y > 10:
            out_of_bounds_flag = True
        else:
            out_of_bounds_flag = False


        if self.x < 0:
            self.x = 0
        elif self.x > 10:
            self.x = 10
        

        if self.y < 0:
            self.y = 0
        elif self.y > 10:
            self.y = 10

        return out_of_bounds_flag
        
    def reward_function(self, defender, goal):
        reward = 0
        defender_diff_x = abs(defender[0]-self.x)
        defender_diff_y = abs(defender[1]-self.y)

        total_defender_diff = math.sqrt(defender_diff_x**2 + defender_diff_y**2)
        
        if total_defender_diff <= 0:
            reward -= 300
            print('Reached Terminal State, the Attacker got the Goal!!!!!!')
            print(reward)
        '''
        elif total_defender_diff <= 3:
            reward -= 5
        elif total_defender_diff <= 10:
            reward -= 1
        elif total_defender_diff <= 20:
            reward -= 0.1
        elif total_defender_diff <= 25:
            reward -= 0.01
        '''
        reward -= total_defender_diff
        
        goal_diff_x = abs(goal[0]-self.x)
        goal_diff_y = abs(goal[1]-self.y)

        total_goal_diff = math.sqrt(goal_diff_x**2 + goal_diff_y**2)
        
        if total_goal_diff <= 0:
            reward += 300
            print('Reached Terminal State, the Attacker got the Goal!!!!!!')
            print(reward)
        '''
        elif total_goal_diff <= 3:
            reward += 5.1
        elif total_goal_diff <= 10:
            reward += 1.1
        elif total_goal_diff <= 20:
            reward += 0.2
        elif total_goal_diff <= 25:
            reward += 0.02
        '''
        reward += total_goal_diff

        
        return reward   

class Defender():
    def __init__(self):
        # Set the initial position to (10, 10)
        self.x = 5
        self.y = 5
    
    def step(self, action):
        
        def step(self, action):
            if action == 0:
                self.x += 0 
                self.y += 1
            elif action == 1:
                self.x += 1 
                self.y += 0
            elif action == 2:
                self.x += 1
                self.y += 1
            elif action == 3:
                self.x -= 1 
                self.y += 0
            elif action == 4:
                self.x -= 1 
                self.y += 1
            elif action == 5:
                self.x -= 1 
                self.y -= 1
            elif action == 6:
                self.x += 1 
                self.y -= 1
            elif action == 7:
                self.x -= 0 
                self.y -= 1

            if self.x < 0:
                self.x = 0
            elif self.x > 10:
                self.x = 10

            if self.y < 0:
                self.y = 0
            elif self.y > 10:
                self.y = 10

            return True

    def reward_function(self, attacker):
        reward = 0
        if (attacker[0]-self.x + attacker[1]-self.y) < 10:
            reward += 1
        elif (attacker[0]-self.x + attacker[1]-self.y) < 3:
            reward += 5
        elif (attacker[0]-self.x + attacker[1]-self.y) < 0:
            reward += 300
        return reward

class Target():
    def __init__(self):
        # Set the initial position to (19, 19)
        self.x = np.random.randint(7, 10)
        self.y = np.random.randint(7, 10)

if __name__ == '__main__':
    env = AdversarialEnv()

    episode_rewards = []

    for episode in range(1):
        
        obs = env.reset()
        episode_reward = 0

        step = 0
        while step <= 5:

            action = np.random.randint(0, 7)
            # Take the action!
            new_obs, reward, done, _ = env.step(action)     
            # ^This will have to be something like 
            # attacker_obs, attacker_reward, done, _ = env.attacker.step(action)  
            # defender_obs, defender_reward, done, _ = env.defender.step(action) 
            env.render()

            episode_reward += reward

            step += 1

        print('Episode Reward: {}'.format(episode_reward))