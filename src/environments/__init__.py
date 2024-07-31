import torch
import gymnasium as gym

from environments.MultiBandit import MultiBandit   
from environments.GridWorld import GridWorld
from policies import RosReinforceActor, MultiBanditNetwork, TabularNN



def get_exp_config(name):
    cfg = Configs[name]
    
    if name.startswith('MultiBandit'):
        cfg['env'] = MultiBandit()
        randomness = name.split('MultiBandit')[-1]
        name = 'MultiBandit'
        if randomness:
            mean_factor, scale_factor = randomness.split('M')[-1].split('S')
            cfg['env'] = MultiBandit(int(mean_factor), int(scale_factor))
    
    if name.startswith('GridWorld'):
        scale = int(name.split('-')[-1])
        cfg['env'] = GridWorld(width=scale, 
                               t_max=scale,
                               normalize_reward=cfg['normalize_reward'],
                               offline_data_number=cfg['offline_data_number'])
        name = 'GridWorld'
        
    return cfg

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
        'save_episodes': [0, 10, 25, 50, 100, 200],  # save_episodes
        'action_discrete': True,
        'state_discrete': True,
        'model': TabularNN
    },
    'GridWorld-1': {
        'env': None,  # env
        'max_time_step': 1,  # max_time_step
        'grid_size': 1,
        'normalize_reward': True,
        'offline_data_number': 0, # how much offline data to use
        'gamma': 0.99,  # gamma
        'lr': 1e-2,  # lr,
        'action_discrete': True,
        'state_discrete': True,
        'model': MultiBanditNetwork
    },
    'GridWorld-5': {
        'env': None,  # env
        'max_time_step': 5,  # max_time_step
        'grid_size': 5,
        'normalize_reward': True,
        'offline_data_number': 0, # how much offline data to use
        'gamma': 0.99,  # gamma
        'lr': 1e-2,  # lr
        'save_episodes': [30],  # save_episodes
        'action_discrete': True,
        'state_discrete': True,
        'model': MultiBanditNetwork
    },
    'GridWorld-10': {
        'env': None,  # env
        'max_time_step': 10,  # max_time_step
        'grid_size': 10,
        'normalize_reward': True,
        'offline_data_number': 0, # how much offline data to use
        'gamma': 0.99,  # gamma
        'lr': 1e-2,  # lr
        'save_episodes': [30],  # save_episodes
        'action_discrete': True,
        'state_discrete': True,
        'model': MultiBanditNetwork
    }
}