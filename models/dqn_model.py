import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.agent_config import get_cfg_defaults

cfgs_model = get_cfg_defaults().MODEL_PARAMETER


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(self.state_size, cfgs_model.H1)
        self.fc2 = nn.Linear(cfgs_model.H1, cfgs_model.H2)
        self.fc3 = nn.Linear(cfgs_model.H2, cfgs_model.H3)
        self.fc4 = nn.Linear(cfgs_model.H3, self.action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
