import torch
import gymnasium as gym
import copy

from policies import Policy, REINFORCE
from utils import AvgAccumGradOptimizer


class PolicyEvaluator:
    def __init__(self, pi_e: Policy, **kwargs) -> None:
        self.pi_e = pi_e

    def act(self, state):
        raise NotImplementedError

    def update(self, state, action, trajectories=None):
        raise NotImplementedError
    
class MonteCarlo(PolicyEvaluator):

    def __init__(self, pi_e: Policy, **kwargs) -> None:
        super().__init__(pi_e, **kwargs)
        self.step = 0

    def act(self, state):
        action, pi_e = self.pi_e.sample(state)
        return action

    def update(self,state, action, trajectories=None):
        self.step += 1
    

        
class ROS(PolicyEvaluator):
    def __init__(self, pi_e: REINFORCE, env: gym.Env, max_time_step: int, **kwargs):
        self.pi_e = pi_e
        self.step = 0
        self.env = env
        self.max_time_step = max_time_step

        self.lr = kwargs['ros_lr']
        self.pi_b = copy.deepcopy(pi_e)

        self.dynamic_lr = kwargs['ros_dynamic_lr']
        
        if kwargs['ros_first_layer']:
            update_modules = self.pi_b.model.last_layer()
        else:
            update_modules = self.pi_b.model
            
        parameters = (parameter for name, parameter in update_modules.named_parameters() if not kwargs['ros_only_weight'] or 'bias' not in name)
        parameter_names = ((name, parameter.shape) for name, parameter in update_modules.named_parameters() if not kwargs['ros_only_weight'] or 'bias' not in name)
        self.optimizer = AvgAccumGradOptimizer(parameters, lr=self.lr)

    def recover_parameters(self):
        for source_param, dump_param in zip(self.pi_b.model.parameters(), self.pi_e.model.parameters()):
            source_param.data.copy_(dump_param.data)

    def act(self, state):
        action, _ = self.pi_b.sample(state)
        return action

    def update(self, state, action, trajectories=None):
        self.step += 1
        self.pi_b.model.eval()  # fix BN
        self.recover_parameters() # reset pi_b to pi_e
        loss = self.pi_b.action_prob(state, action, require_grad=True).log()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(old_weight=self.step - 1, new_weight=1)
            
            
class BehaviorPolicySearch(PolicyEvaluator):
    def __init__(self, pi_e, env, max_time_step, **kwargs):
        self.pi_e = pi_e
        self.step = 0
        self.env = env
        self.max_time_step = max_time_step
        self.pi_b = copy.deepcopy(pi_e)
        
    def act(self, state):
        action, _ = self.pi_b.sample(state)
        return action
    
    def update(self, state, action, ep_info):
        g, trajectory = ep_info
        self.step += 1
        self.pi_b.model.eval()
        # calculate Importance Sampling Ratio
        ratio = 1
        with torch.no_grad():
            for s,a,r in trajectory:
                ratio *= self.pi_e.action_prob(s,a) / self.pi_b.action_prob(s,a)