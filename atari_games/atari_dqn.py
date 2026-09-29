import random
import collections
from typing import Tuple, Deque, List

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T


# ==========================
# Replay Buffer
# ==========================

class ReplayBuffer:
    """
    Simple replay buffer for experience replay.
    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.stack(state),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.stack(next_state),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ==========================
# DQN Networks
# ==========================

class MLPDQN(nn.Module):
    """
    DQN network for low-dimensional state spaces (e.g., CartPole).
    """

    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNDQN(nn.Module):
    """
    DQN network for image-based Atari environments.
    Input: stacked frames (e.g., 4 x 84 x 84).
    """

    def __init__(self, input_channels: int, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute conv output size for fully connected layer
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        conv_out_size = self.conv(dummy_input).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ==========================
# Atari Preprocessing
# ==========================

class AtariPreprocessor:
    """
    Preprocess Atari frames:
    - Convert to grayscale
    - Resize to 84x84
    - Normalize to [0, 1]
    - Stack frames
    """

    def __init__(self, num_frames: int = 4):
        self.num_frames = num_frames
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor(),
        ])
        self.frames: Deque = collections.deque(maxlen=num_frames)

    def reset(self, initial_frame: np.ndarray) -> torch.Tensor:
        self.frames.clear()
        processed = self._process_frame(initial_frame)
        for _ in range(self.num_frames):
            self.frames.append(processed)
        return torch.stack(list(self.frames), dim=0)

    def step(self, frame: np.ndarray) -> torch.Tensor:
        processed = self._process_frame(frame)
        self.frames.append(processed)
        return torch.stack(list(self.frames), dim=0)

    def _process_frame(self, frame: np.ndarray) -> torch.Tensor:
        frame = np.ascontiguousarray(frame)
        tensor = self.transform(frame)
        return tensor


# ==========================
# DQN Agent
# ==========================

class DQNAgent:
    """
    Generic DQN agent for both CartPole and Atari environments.
    """

    def __init__(
        self,
        model: nn.Module,
        num_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 32,
        buffer_capacity: int = 100_000,
        min_buffer_size: int = 1_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 100_000,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.target_model = type(model)(*model.parameters()).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.min_buffer_size = min_buffer_size

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0

    def select_action(self, state: torch.Tensor) -> int:
        """
        Epsilon-greedy action selection.
        """
        self.total_steps += 1
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.total_steps / self.epsilon_decay)
        )

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self) -> None:
        """
        Sample from replay buffer and perform a gradient update.
        """
        if len(self.replay_buffer) < self.min_buffer_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Q(s, a)
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())


# ==========================
# Training Loops
# ==========================

def train_cartpole(
    num_episodes: int = 500,
    target_update_freq: int = 10,
    device: str = "cpu",
) -> None:
    """
    Train a DQN agent on CartPole-v1.
    """
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = MLPDQN(input_dim=obs_dim, num_actions=num_actions)
    agent = DQNAgent(
        model=model,
        num_actions=num_actions,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_capacity=50_000,
        min_buffer_size=1_000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=20_000,
        device=device,
    )

    for episode in range(num_episodes):
        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0.0

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            agent.replay_buffer.push(state, action, reward, next_state_tensor, float(done))

            state = next_state_tensor
            total_reward += reward

            agent.update()

        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"[CartPole] Episode {episode + 1}/{num_episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    env.close()


def train_atari(
    env_name: str,
    num_episodes: int = 500,
    target_update_freq: int = 1000,
    device: str = "cpu",
) -> None:
    """
    Train a DQN agent on an Atari environment (SpaceInvaders, MsPacman).
    """
    env = gym.make(env_name, render_mode=None)
    num_actions = env.action_space.n

    preprocessor = AtariPreprocessor(num_frames=4)
    model = CNNDQN(input_channels=4, num_actions=num_actions)
    agent = DQNAgent(
        model=model,
        num_actions=num_actions,
        gamma=0.99,
        lr=1e-4,
        batch_size=32,
        buffer_capacity=200_000,
        min_buffer_size=10_000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=200_000,
        device=device,
    )

    global_step = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = preprocessor.reset(obs)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = preprocessor.step(next_obs)

            agent.replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                float(done),
            )

            state = next_state
            total_reward += reward
            global_step += 1

            agent.update()

            if global_step % target_update_freq == 0:
                agent.update_target_network()

        print(f"[{env_name}] Episode {episode + 1}/{num_episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    env.close()


def main() -> None:
    """
    Entry point:
    - Train CartPole DQN
    - Train SpaceInvaders DQN
    - Train MsPacman DQN
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n=== Training CartPole DQN ===")
    train_cartpole(num_episodes=200, target_update_freq=10, device=device)

    print("\n=== Training SpaceInvaders DQN ===")
    train_atari(env_name="SpaceInvaders-v0", num_episodes=300, target_update_freq=1000, device=device)

    print("\n=== Training MsPacman DQN ===")
    train_atari(env_name="MsPacman-v0", num_episodes=300, target_update_freq=1000, device=device)


if __name__ == "__main__":
    main()
