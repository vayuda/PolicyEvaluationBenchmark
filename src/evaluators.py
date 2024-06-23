import torch
from torch.optim import SGD
import gymnasium as gym
import copy

from policies import Policy, REINFORCE
from utils import AvgAccumGradOptimizer


class PolicyEvaluator:
    def __init__(self, pi_e: Policy, **kwargs) -> None:
        self.pi_e = pi_e

    def act(self, state):
        raise NotImplementedError

    def update(self, state, action, ep_info=None):
        raise NotImplementedError
    
class MonteCarlo(PolicyEvaluator):

    def __init__(self, pi_e: Policy, **kwargs) -> None:
        super().__init__(pi_e, **kwargs)
        self.step = 0

    def act(self, state):
        return self.pi_e.sample(state)


    def update(self,state, action, ep_info=None):
        self.step += 1
    

        
class ROS(PolicyEvaluator):
    def __init__(self, pi_e: REINFORCE, env: gym.Env, max_time_step: int, lr=1e-3):
        self.pi_e = pi_e
        self.step = 0
        self.env = env
        self.max_time_step = max_time_step

        self.lr = lr
        self.pi_b = copy.deepcopy(pi_e)
            
        parameters = (parameter for name, parameter in self.pi_b.model.named_parameters() if 'bias' not in name)
        parameter_names = ((name, parameter.shape) for name, parameter in self.pi_b.model.named_parameters() if 'bias' not in name)
        self.optimizer = AvgAccumGradOptimizer(parameters, lr=self.lr)

    def recover_parameters(self):
        for source_param, dump_param in zip(self.pi_b.model.parameters(), self.pi_e.model.parameters()):
            source_param.data.copy_(dump_param.data)

    def act(self, state):
        return self.pi_b.sample(state)
                
    def update(self, state, action, ep_info=None):
        self.step += 1
        self.pi_b.model.eval()  # fix BN
        self.recover_parameters() # reset pi_b to pi_e
        loss = torch.tensor(0.0, requires_grad=True)
        cv = self.pi_b.action_prob(state, action, require_grad=True, lp=True)
        loss = loss + cv
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(old_weight=self.step - 1, new_weight=1)
            
            
class BehaviorPolicySearch(PolicyEvaluator):
    def __init__(self, pi_e, env, max_time_step, lr = 1e-3, k = 100):
        self.pi_e = pi_e
        self.env = env
        self.max_time_step = max_time_step
        self.pi_b = copy.deepcopy(pi_e)
        self.trajectories = []
        self.optimizer = SGD(self.pi_b.model.parameters(), lr=lr)
        self.k = k
        
    def act(self, state):
        return self.pi_b.sample(state)
    
    def update(self, state, action, ep_info=None):
        if ep_info is None:
            return
        
        self.trajectories.append(ep_info)
        if len(self.trajectories) % self.k:
            return
        # update pi_b
        err = torch.tensor(0.0, requires_grad=True)
        for g, trajectory in self.trajectories:
            # calculate Importance Sampling Ratio
            lp = 0
            # Separate the elements
            states, actions, probs = zip(*trajectory)
            # States are already tensors, so we can stack them
            s = torch.stack(states)
            a = torch.stack(actions)
            pi_b = torch.sum(torch.tensor(probs).log())
            pi_e = torch.sum(self.pi_e.action_prob(s, a, lp=True))
            lp += self.pi_b.action_prob(s, a, require_grad=True, lp=True).sum()
            ratio = g * (pi_e - pi_b).exp()
            err = err - ratio**2 * lp # torch optimizers minimize the err so make it negative to maximize the objective

        err /= self.k
        self.optimizer.zero_grad()
        err.backward()
        self.optimizer.step()
        self.trajectories = []