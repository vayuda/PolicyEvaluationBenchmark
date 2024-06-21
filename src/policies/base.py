class Policy:
    def action_dist(self, state):
        raise NotImplementedError

    def discrete_action_dist(self, state, n_action=None):
        raise NotImplementedError

    def action_prob(self, state, action):
        raise NotImplementedError

    def sample(self, state):
        raise NotImplementedError