import torch
from torch import nn
from torch.distributions.categorical import Categorical
import gymnasium as gym
import os
import sys
import numpy as np
import time
import wandb
import yaml

class Args:
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    seed: int = 1
    "random number gen seed for reproducibility"
    num_iterations: int = 100000
    """the number of runs to do for each policy"""
    model_folder: str = "./policies/CartPole-v1/"
    """the folder where models are stored"""
    save_file: str = "./results/CartPole-v1/ground_truth.txt"
    """the file to store results in"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        print(envs.action_space)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


with open('ground_truth.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    wandb.init(config=config)

args =Args()
env = gym.make(wandb.config.env_id)
model_file = f'agent{wandb.config.policy_id}.pth'
model_path = os.path.join(args.model_folder, model_file)
truth_reward = []
truth_steps = []
variance = 0
# Load the PyTorch model
agent = Agent(env)
print(f"using model {model_path}")
model = torch.load(model_path)
agent.load_state_dict(model)
start = time.time()
for iteration in range(args.num_iterations):
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs)
    next_done = False
    r = 0
    steps = 0
    while True:
        action, logprob, _, value = agent.get_action_and_value(next_obs)
        next_obs, reward, terimated, truncated, infos = env.step(action.cpu().numpy())
        steps +=1
        r+=reward
        next_done = terimated or truncated
        next_obs = torch.Tensor(next_obs)
        if next_done:
            truth_reward.append(r)
            truth_steps.append(steps)
            break
        
    if not iteration % 1000:    
        print(f"iteration {iteration+1} complete in {time.time() - start :.3f} sec")
        start = time.time()
    

# Calculate the average excluding NaN values
variance = np.var(truth_reward)
truth_reward = np.mean(truth_reward)
truth_steps = np.mean(truth_steps)
wandb.log({'truth_reward': truth_reward, 'truth_steps': truth_steps, 'variance': variance})
with open(args.save_file,'a') as f:
    f.write(f'{model_file}, {truth_reward}, {truth_steps}, {variance}\n')