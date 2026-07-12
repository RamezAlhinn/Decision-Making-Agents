# =============================================================================
# dqn.py — Deep Q-Network agent for the ContinuousMazeEnv
# =============================================================================
#
# The environment (env.py) has a continuous 2D state (x, y in [0,1]) and a
# discrete action space (4 directions), so a tabular Q-table is not a good
# fit — states never repeat exactly. Instead we approximate Q(s, a) with a
# small feed-forward neural network (the "Deep" in Deep Q-Network).
#
# Core DQN ingredients implemented here:
#   1. Q-network        — MLP mapping state -> Q-values for all 4 actions.
#   2. Target network    — a slowly-updated copy of the Q-network used to
#                          compute stable TD targets (reduces the "moving
#                          target" problem of bootstrapping from itself).
#   3. Replay buffer     — stores past transitions and samples random
#                          mini-batches, breaking the correlation between
#                          consecutive steps in an episode.
#   4. Epsilon-greedy    — same exploration/exploitation idea as Q-learning.
# =============================================================================

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """MLP: state (2,) -> Q-values (n_actions,). Hidden layer sizes are configurable
    so experiments.py can sweep over network architecture."""

    def __init__(self, state_dim, n_actions, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size ring buffer of (s, a, r, s', done) transitions with uniform random sampling."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent with a target network and experience replay.

    Parameters
    ----------
    state_dim, n_actions : int
        Dimensions of the environment's observation and action spaces.
    hidden_sizes : tuple[int, ...]
        Sizes of the Q-network's hidden layers.
    lr : float
        Adam learning rate.
    gamma : float
        Discount factor.
    epsilon, epsilon_min, epsilon_decay : float
        Epsilon-greedy exploration schedule (multiplicative decay per episode).
    buffer_capacity, batch_size : int
        Replay buffer size and mini-batch size for each gradient step.
    target_update_freq : int
        Number of training steps between hard copies of the Q-network's
        weights into the target network.
    device : str
        "cpu" or "cuda"/"mps".
    """

    def __init__(
        self,
        state_dim,
        n_actions,
        hidden_sizes=(64, 64),
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_capacity=50_000,
        batch_size=64,
        target_update_freq=500,
        double_dqn=False,
        tau=None,
        device="cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        # Double DQN (van Hasselt et al., 2016): pick the next action with the
        # online network but score it with the target network, to counter
        # max-over-target overestimation. Off by default: a 3-seed A/B test on
        # this maze showed no benefit (one seed failed outright) — the task's
        # danger cliffs seem to benefit from vanilla DQN's optimism.
        self.double_dqn = double_dqn
        # tau: if set, soft-update the target net every step
        # (target <- tau*online + (1-tau)*target) instead of hard-copying
        # every target_update_freq steps. Also off by default: ~2x slower on
        # CPU with no measured stability gain here.
        self.tau = tau
        self.device = torch.device(device)

        self.q_net = QNetwork(state_dim, n_actions, hidden_sizes).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self._train_steps = 0

    def select_action(self, state, greedy=False):
        """Epsilon-greedy action selection. Pass greedy=True to always exploit (evaluation)."""
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Sample a mini-batch and run one gradient step. No-op until the buffer has enough data."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            td_target = rewards + self.gamma * next_q_values * (1.0 - dones)

        # Huber loss + gradient clipping (as in the original DQN paper):
        # the env's -100 danger reward produces TD errors large enough that
        # a plain MSE gradient step can wipe out a partially-learned policy.
        loss = nn.functional.smooth_l1_loss(q_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self._train_steps += 1
        if self.tau is not None:
            with torch.no_grad():
                for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
                    tp.lerp_(op, self.tau)
        elif self._train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
        print(f"Q-network saved -> {path}")

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        print(f"Q-network loaded <- {path}")
