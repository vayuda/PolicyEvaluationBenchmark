import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.optim import Optimizer
import gymnasium as gym
import os
import numpy as np
import time
import wandb
import yaml
import csv
import copy

class Args:
    "random number gen seed for reproducibility"
    seed: int = 1

    """the number of runs to do for each policy"""
    num_iterations: int = 10000
    
    """whether to use wandb"""
    track: bool = False
    

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


#ROS optimizer from original paper
class AvgAccumGradOptimizer(Optimizer):

    def __init__(self, params, lr):
        self.lr = lr
        # self.step_i = 0

        super().__init__(params, {'lr': lr, 'avg_grad': 0})

    @torch.no_grad()
    def step(self, old_weight: float, new_weight: float, update_avg_grad=True):
        grad_sum = 0
        grad_num = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                state = self.state[p]
                avg_grad = state.setdefault('avg_grad', 0)
                # avg_grad = self.step_i / (self.step_i + 1) * avg_grad + d_p / (self.step_i + 1)
                avg_grad = (old_weight * avg_grad + new_weight * d_p) / (old_weight + new_weight)
                if update_avg_grad:
                    state['avg_grad'] = avg_grad
                if group['lr'] > 0:
                    p.add_(avg_grad, alpha=-group['lr'])

                grad_sum += avg_grad.abs().sum().item()
                # grad_sum += avg_grad.pow(2).sum().item()
                grad_num += avg_grad.numel()
        # if update:
        #     self.step_i += 1

        return grad_sum / grad_num
        # return (grad_sum / grad_num) ** 0.5
        
# ROS sampler from original paper
class RobustOnPolicySampler():
    def __init__(self, pi_e: Agent, env: gym.Env, max_time_step: int, **kwargs):
        self.pi_e: Agent = pi_e
        self.step = 0
        self.env = env
        self.max_time_step = max_time_step

        self.lr = kwargs['ros_lr']
        self.pi_b = copy.deepcopy(pi_e)

        self.grad0 = 0
        self.avg_grad = 0
        self.smooth_grad = 0
        self.dynamic_lr = kwargs['ros_dynamic_lr']
        self.optimizer = AvgAccumGradOptimizer(self.pi_b.parameters(), self.lr)

    def recover_parameters(self):
        for source_param, dump_param in zip(self.pi_b.parameters(), self.pi_e.parameters()):
            source_param.data.copy_(dump_param.data)

    def act(self, state):
        action, pi_b, _, _ = self.pi_b.get_action_and_value(state)
        _, pi_e, _, _ = self.pi_e.get_action_and_value(state, action)
        return action, pi_e, pi_b

    def update(self, state, action):
        self.step += 1
        self.pi_b.eval()  # fix BN
        self.recover_parameters()
        with torch.enable_grad():
            _, loss, _, _ = self.pi_b.get_action_and_value(state, action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(old_weight=self.step - 1, new_weight=1)


# retrieve config from weights and biases + yaml file
with open('config/ros.yaml') as f:
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
eval_policy = Agent(env)
eval_policy.load_state_dict(model)
ros_agent = RobustOnPolicySampler(eval_policy, env, 500, **{'ros_lr': wandb.config.alpha, 'ros_dynamic_lr': False, 'ros_first_layer': False, 'ros_only_weight': False})

#tracking variables

steps=[]
return_estimate = []


# # Retrieve montecarlo data
# mc_return_estimate = []
# with open(f'results/{wandb.config.env_id}/mc_results.yaml') as f:
#     data = yaml.load(f, Loader=yaml.FullLoader)
#     mc_return_estimate = data["return_estimate"]

#Run the policy evaluation cycle many iterations

# monte carlo
print("ROS eval")
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

        # ros action / update
        action, pi_e, pi_b = ros_agent.act(state)
        ros_agent.update(state, action)
        
        # step through the environment
        state_, reward, terimated, truncated, infos = env.step(action.cpu().numpy())
        state_ = torch.tensor(state_)
        done = terimated or truncated

        # update episode variables
        episode_steps +=1
        episode_reward+=reward
    

    if not iteration:
        steps.append(episode_steps)
        return_estimate.append(episode_reward)
    else:
        return_estimate.append(
            (len(return_estimate) * return_estimate[-1] + episode_reward) / (len(return_estimate) + 1)
        )
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

#retrieve mse values from mc csv
with open(f'results/{wandb.config.env_id}/mc/summary.csv', newline='') as f:
    reader = csv.reader(f)
    mc_1 = next(reader)[0]
    mc_1 = np.abs(float(mc_1) ** 0.5) 

#calculate errors
return_estimate = np.array(return_estimate)
mse = (return_estimate - truth_reward)**2
mse_norm = mse/mc_1
norm_err = np.abs(return_estimate - truth_reward) / mc_1
    
for i in range(len(return_estimate)):
    wandb.log({
        'cumulative_steps': steps[i],
        'return_estimate': return_estimate[i], 
        'mse': mse[i], 
        'mse_norm': mse_norm[i],
        'norm_err': norm_err[i],
    })


