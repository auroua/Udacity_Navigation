import random
from utils import ReplayBuffer
from configs.agent_config import get_cfg_defaults
import torch.nn.functional as F
import torch

cfgs = get_cfg_defaults().HYPER_PARAMETER
cfgs_env = get_cfg_defaults().ENV_PARAMETER


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = cfgs_env.STATE_SIZE
        self.action_size = cfgs_env.ACTION_SIZE
        self.seed = random.seed(seed)
        # Replay memory
        self.memory = ReplayBuffer(cfgs_env.ACTION_SIZE, cfgs.BUFFER_SIZE, cfgs.BATCH_SIZE, seed)
        assert cfgs.LOSS_TYPE in ('MSE', 'F1'), 'Loss type %s is not support, please choose "MSE" or "F1"'
        if cfgs.LOSS_TYPE == 'MSE':
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = F.smooth_l1_loss
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % cfgs.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > cfgs.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, cfgs.GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        assert False, 'you should implement this method in subclass'

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        assert False, 'you should implement this method im subclass'

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)