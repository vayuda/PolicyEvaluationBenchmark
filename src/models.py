import torch
import numpy as np
from torch import nn
from torch.nn.functional import softplus
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gymnasium as gym
from gym.spaces import Discrete


class PPOAgentCRL(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, envs.action_space.n), std=0.01),
        )
    
    # define the agent architecture so that we can load one
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action)

class MultiBanditNetwork(nn.Module):
    def __init__(self, env: gym.Env, state_discrete: bool, action_discrete: bool, device: torch.device, hidden_size = 32):
        super().__init__()
        self.discrete_state = state_discrete
        self.discrete_action = action_discrete
        self.env = env 
        self.network = nn.Sequential(
            nn.Linear(self.env.observation_space.n, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.env.action_space.n),
            nn.Softmax(-1)
        )
        
    def forward(self, state): 
        return self.network(state)

    def to(self, device):
        self.device = device
        return super().to(device)    
    
class RosVNetwork(nn.Module):
    def __init__(self, env: gym.Env, state_discrete: bool, action_discrete: bool, device: torch.device, max_time_step: int = None, dropout: bool = False):
        super().__init__()
        self.env = env
        self.device = device
        self.max_time_step = max_time_step
        self.discrete_state = state_discrete
        self.discrete_action = action_discrete
        self.hidden_dim = 64
        if self.discrete_state:
            self.state_embedding = torch.nn.Embedding(self.env.observation_space.n, 64).to(self.device)
            self.state_dim = 64
        else:
            self.state_dim = env.observation_space.shape[0]

        if self.discrete_action:
            self.action_dim = self.env.action_space.n
            output_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.action_dim),
                torch.nn.Softmax(-1)
            )
        else:
            assert self.env.action_space.shape[0] == 1, 'Only support action_dim = 1.'
            self.action_dim = self.env.action_space.shape[0]
            self.action_scale = torch.from_numpy(self.env.action_space.high)[0]
            output_layer = torch.nn.Embedding(self.state_dim, self.action_dim * 2) if self.discrete_state else torch.nn.Linear(self.hidden_dim, self.action_dim * 2)
            
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.state_dim),
            torch.nn.Linear(self.state_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5) if dropout else torch.nn.Identity(),
            output_layer
        ).to(self.device)

    def forward(self, states):
        if self.discrete_state:
            states = self.state_embedding(states)
        return self.net(states)

    def to(self, device):
        self.device = device
        return super().to(device)
    
class RosReinforceActor(torch.nn.Module):
    def __init__(self, env: gym.Env, state_discrete: bool, action_discrete: bool, device: torch.device):
        super().__init__()
        self.env = env
        self.device = device

        self.discrete_state = state_discrete
        self.discrete_action = action_discrete
        hidden_dim = 64

        if self.discrete_state:
            self.state_dim = self.env.observation_space.n
            backbone = torch.nn.Identity()
        else:
            self.state_dim = env.observation_space.shape[0]
            backbone = torch.nn.Sequential(
                torch.nn.BatchNorm1d(self.state_dim),
                torch.nn.Linear(self.state_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )

        if self.discrete_action:
            self.action_dim = self.env.action_space.n
            output_layer = torch.nn.Sequential(
                torch.nn.Embedding(self.state_dim, self.action_dim) if self.discrete_state else torch.nn.Linear(hidden_dim, self.action_dim),
                torch.nn.Softmax(-1)
            )
        else:
            assert self.env.action_space.shape[0] == 1, 'Only support action_dim = 1.'
            self.action_dim = self.env.action_space.shape[0]
            self.action_scale = torch.from_numpy(self.env.action_space.high)[0]
            output_layer = torch.nn.Embedding(self.state_dim, self.action_dim * 2) if self.discrete_state else torch.nn.Linear(hidden_dim, self.action_dim * 2)

        # !: if change, remember to check ROS and RIS, both of which involve update parts of parameters
        self.model = torch.nn.Sequential(
            backbone,
            output_layer
        ).to(self.device)

    def forward(self, states, logit=False):
        states = states.to(self.device)
        states = states.long() if self.discrete_state else states

        if logit:
            outputs = self.model[0](states)
            return self.model[1][0](outputs)  # skip softmax

        outputs = self.model(states)
        if not self.discrete_action:
            outputs = outputs.view(outputs.size(0), self.action_dim, 2)
            outputs = torch.stack((
                # outputs[..., 0],
                torch.tanh(outputs[..., 0]) * self.action_scale,

                # outputs[..., 1].abs(),
                # outputs[..., 1].pow(2),
                # outputs[..., 1].exp(),
                softplus(outputs[..., 1]),
            ), -1)

        return outputs

    def to(self, device):
        self.device = device
        return super().to(device)

    def first_layer(self):
        return self.model[0][1]

    def last_layer(self):
        return self.model[1]
    


class Policy:
    def action_dist(self, state):
        raise NotImplementedError

    def discrete_action_dist(self, state, n_action=None):
        raise NotImplementedError

    def action_prob(self, state, action):
        raise NotImplementedError

    def sample(self, state):
        raise NotImplementedError
    

class REINFORCE(Policy):

    def __init__(self, env_config, file: str = None, **kwargs) -> None:
        super().__init__()
        self.env = env_config['env']
        self.state_discrete = env_config['state_discrete']
        self.action_discrete = env_config['action_discrete']
        self.gamma = env_config['gamma']
        self.device = torch.device(kwargs['device'])
        self.model = env_config['model'](self.env, self.state_discrete, self.action_discrete, self.device)
        self.lr = kwargs.get('lr', 1e-2)
        if file:
            self.load(file)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, eps=1e-3, weight_decay=1e-5)

    def save(self, file):
        print('saving', file)
        self.env.close()
        torch.save(self.model, file)

    def load(self, file):
        print('loading', file)
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

        if np.random.random() < .05:
            pie_info = f'max_p: {action_dists.probs.max(-1)[0].mean()}' if isinstance(action_dists, torch.distributions.Categorical) else f'sigma: {action_dists.scale.mean()}'
            print(f'\ta_mean: {advantages.mean().item():.2f}, a_abs_mean: {advantages.abs().mean().item():.2f}, pg_loss: {pg_loss.item():.2f}, {pie_info}')

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

