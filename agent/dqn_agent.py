import numpy as np
import random
import models
import torch
import torch.optim as optim
from agent import Agent
from configs import get_dqn_cfg_defaults

cfgs = get_dqn_cfg_defaults().HYPER_PARAMETER
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DqnAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        Agent.__init__(self, state_size, action_size, seed)
        # Q-Network
        self.qnetwork_local = getattr(models, cfgs.MODEL_TYPE)(self.state_size, self.action_size, seed).to(device)
        self.qnetwork_target = getattr(models, cfgs.MODEL_TYPE)(self.state_size, self.action_size, seed).to(device)
        if cfgs.OPTIMIZER == 'SGD':
            self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=cfgs.LR, momentum=cfgs.MOMENTUM)
        elif cfgs.OPTIMIZER == 'ADAM':
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=cfgs.LR)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        pred_actions = self.qnetwork_local(states)
        pred_actions = pred_actions[range(actions.shape[0]), actions.squeeze(-1)].unsqueeze(-1)
        max_val, _ = torch.max(self.qnetwork_target(next_states).detach(), dim=-1)
        target_actions = gamma * max_val.unsqueeze(-1) * (1 - dones) + rewards
        # MSE_LOSS
        loss = self.criterion(pred_actions, target_actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, cfgs.TAU)


