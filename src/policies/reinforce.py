import torch
import gymnasium as gym
from gymnasium.spaces import Discrete
from torch.nn.functional import softplus
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

from policies.base import Policy

class REINFORCE(Policy):

    def __init__(self, env_config, file: str = None, **kwargs) -> None:
        super().__init__()
        self.env = env_config['env']
        self.state_discrete = env_config['state_discrete']
        self.action_discrete = env_config['action_discrete']
        self.gamma = env_config['gamma']
        self.device = torch.device(kwargs['device'])
        self.model = env_config['model'](self.env, 
                                         self.state_discrete,
                                         self.action_discrete,
                                         self.device, 
                                         env_config.get('hidden_size', 32))
        self.lr = env_config['lr']
        if file:
            self.load(file)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, eps=1e-3, weight_decay=1e-5)

    def save(self, file):
        self.env.close()
        torch.save(self.model, file)

    def load(self, file):
        self.model = torch.load(file).to(self.device)

    def update(self, trajectories):
        self.model.train()

        states, actions, returns = [], [], []
        for trajectory in trajectories:
            g = 0
            for t, s, a, r, d, s_ in reversed(trajectory):
                g = r + self.gamma * g
                # trainings.append(([t], s, a, r, s_, d, g))
                states.append(s)
                actions.append(a)
                returns.append(g)
        # time_steps, states, actions, rewards, next_states, dones, gs = (torch.Tensor(tensors).to(self.device) for tensors in zip(*trainings))
        # advantages = gs
        states = torch.cat(states)
        actions = torch.tensor(actions)
        returns = torch.tensor(returns)
        returns = returns / returns.abs().max()  # reduce variance, but need batch.
        action_dists = self.action_dist(states, require_grad=True)
        log_action_probs = action_dists.log_prob(actions)
        loss = (-returns * log_action_probs.squeeze(-1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def logit_dist(self, state):
        assert isinstance(self.env.action_space, Discrete)
        self.model.eval()
        state = torch.Tensor([state])
        with torch.no_grad():
            return self.model(state, logit=True)[0].numpy()

    def action_dist(self, state, require_grad=False):
        mode = torch.enable_grad if require_grad else torch.no_grad
        with mode():
            outputs = self.model(state)
            if self.action_discrete:    
                dist = Categorical(outputs)
            else:
                action_mean = outputs[..., 0]
                action_scale = outputs[..., 1]
                dist = torch.distributions.Normal(action_mean, action_scale)
            return dist

    
    def action_prob(self, state: torch.Tensor, action: torch.Tensor, lp=False, require_grad=False):
        mode = torch.enable_grad if require_grad else torch.no_grad
        with mode():
            dist = self.action_dist(state, require_grad)
            action_prob = dist.log_prob(action)
            if not lp:
                action_prob = action_prob.exp()
            return action_prob
        
    def sample(self, state, require_grad=False):
        action_dist = self.action_dist(state, require_grad)
        action = action_dist.sample()
        action = action.tolist()[0]
        action_prob = action_dist.log_prob(torch.Tensor([action])[0]).exp().item()
        return action, action_prob

    def discrete_action_dist(self, state, n_action=9):  # no grad.
        action_dist = self.action_dist(state)
        if isinstance(action_dist, torch.distributions.Categorical):
            action_probabilities = action_dist.probs.numpy()
            actions = list(range(len(action_probabilities)))
        else:
            actions = action_dist.sample((n_action,)).tolist()
            action_probabilities = np.ones(n_action) / n_action
        assert np.isclose(action_probabilities.sum(), 1)
        return actions, action_probabilities
    
    