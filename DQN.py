import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 


class DQN():
    def __init__(self):
        self.buffer = np.array([None for i in range(100)])
        #print(len(self.buffer))
        self.Q_Values = np.ones(128)
        print(self.Q_Values)


    def BuildModel(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

if __name__ ==  '__main__':
    model = DQN()