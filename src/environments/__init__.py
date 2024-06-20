from models import RosReinforceActor, MultiBanditNetwork
import gymnasium as gym
from environments.MultiBandit import MultiBandit   
from environments.GridWorld import GridWorld
import torch


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
        cfg['env'] = GridWorld(width=cfg['grid_size'], 
                               t_max=cfg['max_time_step'],
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
        'save_episodes': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],  # save_episodes
        'action_discrete': True,
        'state_discrete': True,
        'model': MultiBanditNetwork
    },
    'GridWorld': {
        'env': None,  # env
        'max_time_step': 10,  # max_time_step
        'grid_size': 10,
        'normalize_reward': True,
        'offline_data_number': 0, # how much offline data to use
        'gamma': 0.99,  # gamma
        'lr': 1e-2,  # lr
        'save_episodes': [20000],  # save_episodes
        'action_discrete': True,
        'state_discrete': True,
        'model': MultiBanditNetwork
    }
}