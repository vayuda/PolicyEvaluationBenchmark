import torch
import gymnasium as gym
from gym.spaces import Discrete
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

        states, actions, advantages = [], [], []
        for trajectory in trajectories:
            g = 0
            for t, s, a, r, d, s_ in reversed(trajectory):
                g = r + self.gamma * g
                # trainings.append(([t], s, a, r, s_, d, g))
                states.append(s)
                actions.append(a)
                advantages.append(g)
        # time_steps, states, actions, rewards, next_states, dones, gs = (torch.Tensor(tensors).to(self.device) for tensors in zip(*trainings))
        # advantages = gs
        states = torch.cat(states)
        actions = torch.tensor(actions)
        advantages = torch.tensor(advantages)
        advantages = advantages / advantages.abs().max()  # reduce variance, but need batch.
        action_dists = self.action_dists(states, require_grad=True, min_sigma=0.00, fix_bn=False)
        log_action_probs = action_dists.log_prob(actions)
        pg_loss = (-advantages * log_action_probs.squeeze(-1)).mean()

        self.optimizer.zero_grad()
        pg_loss.backward()
        self.optimizer.step()

        # if np.random.random() < .05:
        #     pie_info = f'max_p: {action_dists.probs.max(-1)[0].mean()}' if isinstance(action_dists, torch.distributions.Categorical) else f'sigma: {action_dists.scale.mean()}'
        #     print(f'\ta_mean: {advantages.mean().item():.2f}, a_abs_mean: {advantages.abs().mean().item():.2f}, pg_loss: {pg_loss.item():.2f}, {pie_info}')

    def logit_dist(self, state):
        assert isinstance(self.env.action_space, Discrete)
        self.model.eval()
        state = torch.Tensor([state])
        with torch.no_grad():
            return self.model(state, logit=True)[0].numpy()

    def action_dists(self, states, require_grad=False, sigma_grad=True, min_sigma=0., fix_bn=True, _single_state=False):
        if fix_bn:
            self.model.eval()
                
        states = states if isinstance(states, torch.Tensor) else torch.FloatTensor(states)
        with torch.enable_grad() if require_grad else torch.no_grad():
            outputs = self.model(states)
        outputs = outputs[0] if _single_state else outputs

        if self.action_discrete:
            action_dists = torch.distributions.Categorical(outputs)
        else:
            action_means = outputs[..., 0]
            action_scales = outputs[..., 1]
            action_dists = torch.distributions.Normal(action_means, (action_scales if sigma_grad else action_scales.detach()) + min_sigma * action_scales.detach().abs())

        return action_dists

    def action_dist(self, state, min_sigma=0., fix_bn=True):
        if fix_bn:
            self.model.eval()
        outputs = self.model(state)
        # print(outputs)
        if self.action_discrete:
            action_dist = Categorical(outputs)
        else:
            action_mean = outputs[..., 0]
            action_scale = outputs[..., 1]
            action_dist = torch.distributions.Normal(action_mean, action_scale)
        return action_dist
    
    def action_probs(self, states, actions, **kwargs):
        action_dists = self.action_dists(states, **kwargs)
        actions = actions if isinstance(actions, torch.Tensor) else torch.Tensor(actions)
        action_probs = action_dists.log_prob(actions).exp()
        return action_probs

    def action_prob(self, state, action, require_grad=False):
        if require_grad:
            with torch.enable_grad():
                action_dist = self.action_dist(state)
                action = torch.tensor(action)
                action_prob = action_dist.log_prob(action).exp()
                return action_prob
        else:
            with torch.no_grad():
                action_dist = self.action_dist(state)
                action_prob = action_dist.log_prob(action).exp()
                return action_prob.item()

    def sample(self, state, min_sigma=0.,fix_bn=True):
        action_dist = self.action_dist(state, min_sigma=min_sigma, fix_bn=fix_bn)
        # print(action_dist)
        action = action_dist.sample()
        # print(action)
        # action = action_dist.mean
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
    
    