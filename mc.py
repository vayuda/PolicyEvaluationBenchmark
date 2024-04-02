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
import csv

class Args:
    """the id of the environment"""
    env_id: str = "CartPole-v1"

    "random number gen seed for reproducibility"
    seed: int = 1

    """the number of runs to do for each policy"""
    num_iterations: int = 10000

    """the folder where models are stored"""
    model_folder: str = "policies/CartPole-v1/"

    """the file to store results in"""
    save_file: str = "results/CartPole-v1/"

    

# define the agent architecture so that we can load one
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
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


# retrieve config from weights and biases + yaml file
with open('config/mc.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    wandb.init(config=config)


#initialize the environment
args = Args()
env = gym.make(wandb.config.env_id)

# Load the PyTorch model
model_file = f'agent{wandb.config.policy_id}.pth'
model_path = os.path.join(f"./policies/{wandb.config.env_id}/", model_file)
model = torch.load(model_path)

# create agents
agent = Agent(env) # Monte Carlo
agent.load_state_dict(model)

#tracking variables
steps=[]
return_estimate = []
variance = []

#Run the policy evaluation cycle many iterations

# monte carlo
print("monte carlo eval")
start = time.time()
for iteration in range(args.num_iterations):
    #reset the environment and episode variables
    state, _ = env.reset(seed=args.seed)
    state = state_ = torch.tensor(state)
    done = False
    episode_reward = 0
    episode_steps = 0
    episode_trajectory = []
    #run the episode
    while not done:
        # update current state
        state = state_

        #take action
        action, _, _, _ = agent.get_action_and_value(state)

        # step through the environment
        state_, reward, terimated, truncated, infos = env.step(action.cpu().numpy())
        state_ = torch.tensor(state_)
        done = terimated or truncated
        # update episode variables
        episode_steps +=1
        episode_reward+=reward
    
    
    

    if not iteration:
        variance.append(0)
        steps.append(episode_steps)
        return_estimate.append(episode_reward)
    else:
        return_estimate.append(
            (len(return_estimate) * return_estimate[-1] + episode_reward) / (len(return_estimate) + 1)
        )
        # variance.append(
        #     variance[-1] + (episode_reward - return_estimate[-2])*(episode_reward - return_estimate[-1])
        # )
        steps.append(steps[-1]+episode_steps)

    #display progress
    if not iteration % 500:    
        print(f"iteration {iteration+1} complete in {time.time() - start :.3f} sec")
        start = time.time()

# retrieve ground truth data
with open(f'results/{wandb.config.env_id}/ground_truth_results.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    truth_reward = data[wandb.config.policy_id]["truth_reward"]
    truth_steps = data[wandb.config.policy_id]["truth_steps"]
    
#calculate errors
return_estimate = np.array(return_estimate)
mse = (return_estimate - truth_reward)**2
norm_err = np.abs(return_estimate - truth_reward) / np.abs(return_estimate[0] - truth_reward)
# standard_error = np.sqrt(variance) / np.sqrt(np.array(range(len(variance))))

# Append return_estimate values to a csv file
np.append(return_estimate, wandb.config.policy_id)
np.append(steps, wandb.config.policy_id)


with open(f'results/{wandb.config.env_id}/mc_return_estimate.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(return_estimate)

with open(f'results/{wandb.config.env_id}/mc_steps.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(steps)

with open(f'results/{wandb.config.env_id}/mc_mse.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(mse)

with open(f'results/{wandb.config.env_id}/mc_norm_err.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(norm_err)

for i in range(len(return_estimate)):
    wandb.log({
        'cumulative_steps': steps[i],
        'return_estimate': return_estimate[i], 
        'mse': mse[i], 
        'norm_err': norm_err[i],
        # 'standard_error': standard_error[i],
    })


