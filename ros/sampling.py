import torch
from torch.optim import Optimizer
import gymnasium as gym
import copy
from models import Policy, REINFORCE

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

class DataCollector:
    def __init__(self, pi_e: Policy, **kwargs) -> None:
        self.pi_e = pi_e

    def act(self, state):
        raise NotImplementedError

    def update(self, state, action, trajectories=None):
        raise NotImplementedError
    
    
class MonteCarlo(DataCollector):

    def __init__(self, pi_e: Policy, **kwargs) -> None:
        super().__init__(pi_e, **kwargs)
        self.step = 0

    def act(self, state):
        action, pi_e = self.pi_e.sample(state)
        return action

    def update(self,state, action, trajectories=None):
        self.step += 1
    

# class RobustOnPolicySampler(DataCollector):
#     def __init__(self, pi_e: REINFORCE, hasher: DefaultHasher, env: gym.Env, max_time_step: int, **kwargs):
#         DataCollector.__init__(self, pi_e)
#         self.pi_e: REINFORCE = pi_e
#         self.step = 0
#         self.env = env
#         self.max_time_step = max_time_step
#         self.count = AdaptiveDefaultDict(int, hasher)

#         self.lr = kwargs['ros_lr']
#         self.pi_b = copy.deepcopy(pi_e)

#         self.grad0 = 0
#         self.avg_grad = 0
#         self.smooth_grad = 0
#         self.dynamic_lr = kwargs['ros_dynamic_lr']

#         if kwargs['ros_first_layer']:
#             # update_modules = self.pi_b.model.first_layer()
#             update_modules = self.pi_b.model.last_layer()
#         else:
#             update_modules = self.pi_b.model

#         parameters = (parameter for name, parameter in update_modules.named_parameters() if not kwargs['ros_only_weight'] or 'bias' not in name)
#         parameter_names = ((name, parameter.shape) for name, parameter in update_modules.named_parameters() if not kwargs['ros_only_weight'] or 'bias' not in name)

#         # parameters = (parameter for name, parameter in update_modules.named_parameters() if name != 'model.1.0.bias')
#         # parameter_names = ((name, parameter.shape) for name, parameter in update_modules.named_parameters() if name != 'model.1.0.bias')

#         self.optimizer = AvgAccumGradOptimizer(parameters, lr=self.lr)
#         print('ROS update:')
#         [print(item) for item in parameter_names]

#     def recover_parameters(self):
#         for source_param, dump_param in zip(self.pi_b.model.parameters(), self.pi_e.model.parameters()):
#             source_param.data.copy_(dump_param.data)

#     def act(self, time_step, state):
#         action, pi_b = self.pi_b.sample(state)
#         pi_e = self.pi_e.action_prob(state, action)
#         # if np.isnan(pi_e):
#         #     pass
#         return action, pi_e, pi_b

#     def update(self, time_step, state, action, reward, next_state, done, pie_a, pib_a, trajectories=None):
#         self.step += 1
#         self.count[(state, 0, action)] += 1

#         self.pi_b.model.eval()  # fix BN
#         self.recover_parameters()
#         loss = self.pi_b.action_prob(state, action, require_grad=True).log()
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step(old_weight=self.step - 1, new_weight=1)
        
class ROS(DataCollector):
    def __init__(self, pi_e: REINFORCE, env: gym.Env, max_time_step: int, **kwargs):
        self.pi_e = pi_e
        self.step = 0
        self.env = env
        self.max_time_step = max_time_step

        self.lr = kwargs['ros_lr']
        self.pi_b = copy.deepcopy(pi_e)

        self.dynamic_lr = kwargs['ros_dynamic_lr']
        
        if kwargs['ros_first_layer']:
            # update_modules = self.pi_b.model.first_layer()
            update_modules = self.pi_b.model.last_layer()
        else:
            update_modules = self.pi_b.model
            
        parameters = (parameter for name, parameter in update_modules.named_parameters() if not kwargs['ros_only_weight'] or 'bias' not in name)
        parameter_names = ((name, parameter.shape) for name, parameter in update_modules.named_parameters() if not kwargs['ros_only_weight'] or 'bias' not in name)

        # parameters = (parameter for name, parameter in update_modules.named_parameters() if name != 'model.1.0.bias')
        # parameter_names = ((name, parameter.shape) for name, parameter in update_modules.named_parameters() if name != 'model.1.0.bias')

        self.optimizer = AvgAccumGradOptimizer(parameters, lr=self.lr)
        print('ROS update:')
        [print(item) for item in parameter_names]

    def recover_parameters(self):
        for source_param, dump_param in zip(self.pi_b.model.parameters(), self.pi_e.model.parameters()):
            source_param.data.copy_(dump_param.data)

    def act(self, state):
        action, _ = self.pi_b.sample(state)
        return action

    def update(self, state, action,debug=False):
        self.step += 1
        self.pi_b.model.eval()  # fix BN
        self.recover_parameters() # reset pi_b to pi_e
        loss = self.pi_b.action_prob(state, action, require_grad=True).log()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(old_weight=self.step - 1, new_weight=1)
        if debug:
            print(loss)
