import torch
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np

class MultiBandit(gym.Env):
    action_space = Discrete(30)
    observation_space = Discrete(1)
    MaxStep = 1
    gamma = 1
    
    def __init__(self, mean_factor=1.0, scale_factor=1.0) -> None:
        super().__init__()
        self.ready = False
        self.state = torch.zeros(1,1)
        self.name = 'MultiBandit'

        self.means = np.random.rand(30) * mean_factor
        self.scales = np.random.rand(30) * scale_factor
        # print("=====================================")
        # print("Creating MultiBandit Environment")
        # print("means: ", self.means)
        # print("scales: ", self.scales)
        # print(f'maximum expected returns: {self.means.max()}')
        # print("=====================================")
        np.random.seed(None)
        
    def step(self, action: int):
        assert self.ready, "please reset."
        assert action in range(30)

        r = np.random.normal(self.means[action], self.scales[action])
        self.ready = False
        d = True

        return self.state, r, d, d,''

    def reset(self):
        self.ready = True
        return self.state

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

    def policy_value(self, pi):
        action_probabilities = pi.action_prob(self.state)
        return np.sum([
            p * self.means[i] for i, p in enumerate(action_probabilities)
        ])

    def optimal_policy(self, state):
        return np.argmax(self.means)