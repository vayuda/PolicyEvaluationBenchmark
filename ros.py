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
    """the id of the environment"""
    env_id: str = "CartPole-v1"

    "random number gen seed for reproducibility"
    seed: int = 1

    """the number of parallel game environments"""
    num_envs: int = 123

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
with open('ros.yaml') as f:
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
target_policy = Agent(env) # ROS
target_policy.load_state_dict(model)
behavior_policy = Agent(env) # ROS
behavior_policy.load_state_dict(model)

#tracking variables
k=0
steps=[]
episode_reward_estimate = [0]
episode_variance = []
step_reward_estimate = [0]
step_variance_estimate = []

#Configure ROS related variables
optimizer = torch.optim.Adam(behavior_policy.parameters(), lr=wandb.config.alpha)
gradient = torch.tensor(0.0, requires_grad=True)

# retrieve ground truth data
with open(f'results/{wandb.config.env_id}/ground_truth_results.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    truth_reward = data[wandb.config.policy_id]["truth_reward"]
    truth_steps = data[wandb.config.policy_id]["truth_steps"]

#simulate one montecarlo episode to calculate normalized error
state, _ = env.reset(seed=wandb.config.seed)
state = torch.tensor(state)
done = False
mc_estimate = 0
while not done:
    action, _, _, _ = target_policy.get_action_and_value(state)
    state, reward, done, _, _ = env.step(action.cpu().numpy())
    state = torch.tensor(state)
    mc_estimate+=reward


#Run ROS
print("ROS eval")
for iteration in range(wandb.config.num_iterations):
    #reset the environment and episode variables
    state, _ = env.reset(seed=wandb.config.seed)
    state = state_ = torch.tensor(state)
    done = False
    episode_reward = 0
    episode_steps = 0
    episode_trajectory = []
    k = sum(steps)
    #run the episode
    while not done:
        # update current state
        state = state_

        # update ros policy
        behavior_policy.load_state_dict(target_policy.state_dict())
        optimizer.zero_grad()
        gradient.backward(retain_graph=True)
        optimizer.step()

        # ros action / update
        action, _, _, _ = behavior_policy.get_action_and_value(state)
        _, logprob, _, _ = target_policy.get_action_and_value(state, action)
        gradient = torch.tensor(k * gradient.detach() / (k + 1) + logprob/ (k + 1), requires_grad=True) # incremental average
        
        # step through the environment
        state_, reward, terimated, truncated, infos = env.step(action.cpu().numpy())
            
        state_ = torch.tensor(state_)
        done = terimated or truncated

        # update episode variables
        episode_steps +=1
        episode_reward+=reward
        k+=1
    
    episode_reward_estimate.append(
        (len(episode_reward_estimate) * episode_reward_estimate[-1] + episode_reward) / (len(episode_reward_estimate) + 1)
    )
    steps.append(episode_steps)

    if not iteration:
        episode_variance.append(0)
    else:
        episode_variance.append(
            episode_variance[-1] + (episode_reward - episode_reward_estimate[-2])*(episode_reward - episode_reward_estimate[-1])
        )
        
    
    #display progress
    if not iteration % 100:    
        print(f"iteration {iteration+1} complete in {time.time() - start :.3f} sec")
        start = time.time()


#calculate errors

episode_mse = (episode_reward_estimate - truth_reward)**2
episode_norm_err = (episode_reward_estimate - truth_reward) / (mc_estimate - truth_reward)
step_mse = (step_reward_estimate - truth_reward)**2
step_norm_err = (step_reward_estimate - truth_reward) / (step_reward_estimate[0] - truth_reward)

for i in range(len(episode_reward_estimate)):
    wandb.log({
        'episode_reward_estimate': episode_reward_estimate[i], 
        'episode_mse': episode_mse[i], 
        'norm_mse':
        'episode_variance': episode_variance[i],
        'norm_err': norm_err[i]
    })
