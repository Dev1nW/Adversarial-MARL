import numpy as np
from PIL import Image  
import cv2  
import matplotlib.pyplot as plt  
import time 
from gym import Env
from gym.spaces import Discrete, Box
from IPython.display import display, clear_output
import math
#pygame


epsilon = 0.3  # randomness
EPS_DECAY = 0.995  # How fast/slow we want to perform random actions

LEARNING_RATE = 0.01
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
        

        plt.ion() #this is for the plotting 
        self.fig = None
        self.ax = None
        self.plt_counter = 0

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

    def step(self, action1, action2):
        # This will be where we will have to implement 2 different step functions (1: Attacker, 2: Defender)
        out_of_bounds_flag = self.attacker.step(action1)
        #if action < 5:
        out_of_bounds_flag_defender = self.defender.step(action2)
        #else:
        #    self.defender.step(action-4)
        
        attacker_reward = self.attacker.reward_function([self.defender.x, self.defender.y], [self.target.x, self.target.y])

        if out_of_bounds_flag:
            attacker_reward = -10

        defender_reward = self.defender.reward_function([self.attacker.x, self.attacker.y], [self.target.x, self.target.y])

        if out_of_bounds_flag_defender:
            defender_reward = -10

        self.state = np.array([self.attacker.x, self.attacker.y, self.defender.x, self.defender.y, self.target.x, self.target.y])

        self.render_obs.append([self.attacker.x, self.attacker.y, self.defender.x, self.defender.y, self.target.x, self.target.y])


        if (self.attacker.x == self.target.x and self.attacker.y == self.target.y) or (self.attacker.x == self.defender.x and self.attacker.y == self.defender.y):
            done = True
        else:
            done = False

        return self.state, attacker_reward, defender_reward, done, None

    
    def render(self):
        # This creates a single frame
        video = np.zeros((11, 11, 3), dtype=np.uint8)
        video[int(self.target.x)][int(self.target.y)] = d[Target_N]
        video[int(self.attacker.x)][int(self.attacker.y)] = d[Attacker_N]
        video[int(self.defender.x)][int(self.defender.y)] = d[Defender_N]
        img = Image.fromarray(video, 'RGB')
        
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()

        self.ax.clear()
        self.ax.imshow(img)
        self.ax.set_title(f"Step: {self.plt_counter}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        #self.plt_counter += 1

    # def print_episode_rewards(self, episode_rewards):
    #     plt.figure(figsize=(6, 6))  
    #     plt.plot(episode_rewards)
    #     plt.xlabel("Episode")
    #     plt.ylabel("Episode Reward")
    #     plt.title("Episode Rewards Over Time")
    #     plt.show()
    #    # plt.waitforbuttonpress()
    #     #plt.pause(inf)
    #     plt.pause(100000)


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
            print('Reached Terminal State, the Defender got the Attacker!!!!!!')
            #print(reward)
        '''
        elif total_defender_diff <= 3:
            reward -= 10
        elif total_defender_diff <= 10:
            reward -= 1
        elif total_defender_diff <= 20:
            reward -= 0.1
        elif total_defender_diff <= 25:
            reward -= 0.01
        '''
        reward -= 1/(total_defender_diff+0.01)
        
        goal_diff_x = abs(goal[0]-self.x)
        goal_diff_y = abs(goal[1]-self.y)

        total_goal_diff = math.sqrt(goal_diff_x**2 + goal_diff_y**2)
        
        if total_goal_diff <= 0:
            reward += 300
            print('Reached Terminal State, the Attacker got the Goal!!!!!!')
            #print(reward)
        '''
        elif total_goal_diff <= 3:
            reward += 11
        elif total_goal_diff <= 10:
            reward += 5
        elif total_goal_diff <= 20:
            reward += 0.1
        elif total_goal_diff <= 25:
            reward += 0.01
        '''
        reward += 1/(total_goal_diff+0.1)

        #print(total_defender_diff, total_goal_diff)
        
        return reward   

class Defender():
    def __init__(self):
        # Set the initial position to (10, 10)
        self.x = 6
        self.y = 6
    
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

    def reward_function(self, attacker, goal):
        reward = 0
        defender_diff_x = abs(attacker[0]-self.x)
        defender_diff_y = abs(attacker[1]-self.y)

        total_defender_diff = math.sqrt(defender_diff_x**2 + defender_diff_y**2)
        
        if total_defender_diff <= 0:
            reward += 300
            #print('Reached Terminal State, the Defender got the Attacker!!!!!!')
            #print(reward)
        '''
        elif total_defender_diff <= 3:
            reward -= 10
        elif total_defender_diff <= 10:
            reward -= 1
        elif total_defender_diff <= 20:
            reward -= 0.1
        elif total_defender_diff <= 25:
            reward -= 0.01
        '''
        reward += 1/(total_defender_diff+0.1)
        
        goal_diff_x = abs(goal[0]-attacker[0])
        goal_diff_y = abs(goal[1]-attacker[1])

        total_goal_diff = math.sqrt(goal_diff_x**2 + goal_diff_y**2)
        
        if total_goal_diff <= 0:
            reward -= 300
            #print('Reached Terminal State, the Attacker got the Goal!!!!!!')
            #print(reward)
        '''
        elif total_goal_diff <= 3:
            reward += 11
        elif total_goal_diff <= 10:
            reward += 5
        elif total_goal_diff <= 20:
            reward += 0.1
        elif total_goal_diff <= 25:
            reward += 0.01
        '''

        reward -= 1/(total_goal_diff+0.1)

        #print(total_defender_diff, total_goal_diff)
        
        return reward

class Target():
    def __init__(self):
        # Set the initial position to (19, 19)
        self.x = 10#np.random.randint(7, 10)
        self.y = 10#np.random.randint(7, 10)

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