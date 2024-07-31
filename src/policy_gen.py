import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import torch
import yaml
import wandb
import os

from policies import REINFORCE
from environments import get_exp_config

MODE = "offline"

def play_episode(env: gym.Env, max_time_step: int, actor: REINFORCE):
    t = 0
    d = False
    returns = 0
    s = env.reset()
    trajectory = []
    while not d:
        if type(s) == np.ndarray:
            s = torch.tensor(s,dtype=torch.float32).unsqueeze(0)
        elif type(s) == tuple:
            s = torch.tensor(s[0], dtype=torch.float32).unsqueeze(0)
        a, _ = actor.sample(s)
        a = [np.clip(a[0], env.action_space.low[0], env.action_space.high[0])] if isinstance(env.action_space, Box) else a

        s_, r, truncated, terminated , info = env.step(a)
        # print(a, r)
        d = truncated or terminated
        d = d or t == max_time_step - 1
        returns += r

        trajectory.append((t, s, a, r, d, s_))
        s = s_
        t += 1

    return trajectory, returns


def train(cfg):
    os.makedirs(f'policies/{cfg.env_id}/{cfg.policy}', exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env_config = get_exp_config(cfg.env_id)
    print(cfg.policy)
    print(f'training: {cfg.env_id}, config: {env_config}')
    
    
    env = env_config['env']
    max_time_step = env_config['max_time_step']
    gamma = env_config['gamma']
    
    save_step = cfg.train_steps // cfg.n_policies
    actor = REINFORCE(env_config, device='cpu')
    batch = []
    batch_size = 10
    for episode in range(cfg.train_steps):
        tr, returns = play_episode(env, max_time_step, actor)
        batch.append(tr)

        if len(batch) == batch_size:
            actor.update(batch)
            batch.clear()

        if episode % save_step == 0:
            eval_trs, eval_returns = zip(*[play_episode(env, max_time_step, actor) for i in range(100)])
            print(
                f'episode {episode}\t'
                f'training_tot_returns={returns:.2f}, '
                f'training_dis_returns={sum([t[3] * gamma ** i for i, t in enumerate(tr)]):.2f}\t'
                f'eval_tot_returns={np.mean(eval_returns):.2f}({np.min(eval_returns):.2f}, {np.max(eval_returns):.2f})\t'
                f'eval_dis_returns={np.mean([sum([t[3] * gamma ** i for i, t in enumerate(tr)]) for tr in eval_trs]):.2f}\t'
            )
            actor.save(f'policies/{cfg.env_id}/{cfg.policy}/model_{episode}_{cfg.seed}.pt')
             

if __name__ == '__main__':
    with open('config/policy_gen.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        wandb.init(config=config, mode=MODE)
    train(wandb.config)
    