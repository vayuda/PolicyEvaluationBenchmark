import gymnasium as gym
from gym.spaces import Discrete, Box
from collections import defaultdict
import numpy as np
import torch

from models import REINFORCE


def play_episode(env: gym.Env, max_time_step: int, actor: REINFORCE):
    t = 0
    d = False
    returns = 0
    s = env.reset(seed=0)
    trajectory = []
    while not d:
        if type(s) == np.ndarray:
            s = torch.tensor(s).unsqueeze(0)
        elif type(s) == tuple:
            s = torch.tensor(s[0]).unsqueeze(0)
        a, _ = actor.sample(s, min_sigma=0.00)
        a = [np.clip(a[0], env.action_space.low[0], env.action_space.high[0])] if isinstance(env.action_space, Box) else a

        s_, r, truncated, terminated , info = env.step(a)
        d = truncated or terminated
        d = d or t == max_time_step - 1
        returns += r

        # learning trick for MountainCar
        if exp_name == 'MountainCar' or exp_name == 'MountainCarContinuous':
            d = d or (t == 200)  # max_time_step
            r = (max([t[1][0] for t in trajectory]) + 0.5) * 100 if d else r  # higher reward if closer to goal

        trajectory.append((t, s, a, r, d, s_))
        s = s_
        t += 1

    return trajectory, returns


def train():
    print(f'training: {exp_name}, config: {Configs[exp_name]}')
    config = Configs[exp_name]
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    env = config['env']
    max_time_step = config['max_time_step']
    gamma = config['gamma']
    save_episodes = config['save_episodes']
    
    training_episodes = max(save_episodes) + 1
    actor = REINFORCE(config, device='cpu')
    batch = []
    batch_size = 16
    for episode in range(training_episodes):
        if episode in save_episodes and SAVE:
            actor.save(f'policies/model_{exp_name}_{episode}.pt')

        tr, returns = play_episode(env, max_time_step, actor)
        batch.append(tr)

        if len(batch) == batch_size:
            actor.update(batch)
            batch.clear()

        if episode % 500 == 0:
            eval_trs, eval_returns = zip(*[play_episode(env, max_time_step, actor) for i in range(100)])
            print(
                f'episode {episode}\t'
                f'training_tot_returns={returns:.2f}, '
                f'training_dis_returns={sum([t[3] * gamma ** i for i, t in enumerate(tr)]):.2f}\t'
                f'eval_tot_returns={np.mean(eval_returns):.2f}({np.min(eval_returns):.2f}, {np.max(eval_returns):.2f})\t'
                f'eval_dis_returns={np.mean([sum([t[3] * gamma ** i for i, t in enumerate(tr)]) for tr in eval_trs]):.2f}\t'
            )
            

def get_exp_config(exp_name):
    return Configs[exp_name]
Configs = {
    'CartPole': {
        'env': gym.make('CartPole-v1'),
        'max_time_step': 500,
        'gamma': 0.99,
        'lr': 1e-3,
        'save_episodes': [5000, 10000, 15000, 20000, 25000, 30000],
        'action_discrete': True,
        'state_discrete': False
    }
}

if __name__ == '__main__':
    SAVE = True

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='CartPole')
    args = parser.parse_args()
    exp_name = args.exp_name

    train()
