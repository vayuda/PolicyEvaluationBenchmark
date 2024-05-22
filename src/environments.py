import gym
import numpy as np
import torch
from gym.spaces import Discrete, Box
import copy
from models import Policy, RosReinforceActor, MultiBanditNetwork


def get_exp_config(name):
    cfg = Configs[name]
    
    if name.startswith('MultiBandit'):
        cfg['env'] = MultiBandit()
        randomness = name.split('MultiBandit')[-1]
        name = 'MultiBandit'
        if randomness:
            mean_factor, scale_factor = randomness.split('M')[-1].split('S')
            cfg['env'] = MultiBandit(int(mean_factor), int(scale_factor))
            
    return cfg



class MultiBandit(gym.Env):
    action_space = Discrete(30)
    observation_space = Discrete(1)
    MaxStep = 1
    gamma = 1

    # observation_space = Box(0, 0, (1,))

    def __init__(self, mean_factor=1.0, scale_factor=1.0) -> None:
        super().__init__()
        self.ready = False
        self.state = torch.zeros(1,1)
        self.name = 'MultiBandit'

        self.means = np.random.rand(30) * mean_factor
        self.scales = np.random.rand(30) * scale_factor
        print("=====================================")
        print("Creating MultiBandit Environment")
        print("means: ", self.means)
        print("scales: ", self.scales)
        print(f'maximum expected returns: {self.means.max()}')
        print("=====================================")
        
        # maximum expected returns: 0.978618342232764
        

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

    def policy_value(self, pi: Policy):
        actions, action_probabilities = pi.action_dist(self.state)
        return np.sum([
            p * self.means[i] for i, p in enumerate(action_probabilities)
        ])

    def optimal_policy(self, state):
        return np.argmax(self.means)
    

Configs = {
    'CartPole': {
        'env': gym.make('CartPole-v1'),
        'max_time_step': 500,
        'gamma': 0.99,
        'lr': 1e-3,
        'save_episodes': [5000, 10000, 15000, 20000, 25000, 30000],
        'action_discrete': True,
        'state_discrete': False,
        'model': RosReinforceActor
    },
    'MultiBandit':{
        'env': None,  # env
        'max_time_step': MultiBandit.MaxStep,  # max_time_step
        'gamma': MultiBandit.gamma,  # gamma
        'lr': 1e-2,  # lr
        'save_episodes': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],  # save_episodes
        'action_discrete': True,
        'state_discrete': True,
        'model': MultiBanditNetwork
    }
}