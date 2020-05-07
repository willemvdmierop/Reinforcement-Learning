from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transform
import gym
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
import csv
from IPython import display
from IPython.display import clear_output

clear_output()
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import animation, rc
from IPython.display import Math, HTML
from pylab import rcParams
from matplotlib.animation import writers
writers.reset_available_writers()

rcParams['figure.figsize'] = 15, 9

env = gym.make("LunarLander-v2")
env.seed(0)

# Experience replay memory for training our DQN. stores the transitions that the agent observes
class ReplayMemory(object):

    def __init__(self, action_size, capacity, batch_size, seed):
        self.capacity = capacity
        self.action_size = action_size
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        # this will add an experience to memory
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# convolutional neural network with 4 outputs.
class DQN(nn.Module):

    def __init__(self, state_size, action_size, fc1_size=64, fc2_size=64):
        super(DQN, self).__init__()
        hidden_size = 30
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.out = nn.Linear(fc2_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.out(x)
        return action


class Lunar_agent:
    def __init__(self, state_size, action_size, seed, batch_size=64, gamma=0.99, learning_rate=1e-4,
                 capacity=int(1e5), update_every=4, tau=1e-3):
        self.state_size = state_size  # Observation array length = env.observation_space.shape[0]
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.capacity = capacity
        self.update_every = update_every
        self.tau = tau

        # Q-Network
        self.policy_net = DQN(self.state_size, self.action_size).to(device)
        self.target_net = DQN(self.state_size, self.action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(self.action_size, self.capacity, self.batch_size, self.seed)
        self.time_step = 0  # initialize time step for updating every predefined number of steps

    def step(self, state, action, reward, next_state, done):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # We are updating every UPDATE_EVERY time steps
        self.time_step = (self.time_step + 1) % self.update_every
        if self.time_step == 0:
            # if enough samples are available in memory, take a random subset
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.train_DQN(experiences, self.gamma)

    def get_action(self, state, epsilon=0):
        """
        This method will return the actions for given state following the current policy
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def soft_update(self, policy_net, target_net, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            policy net (PyTorch model): weights will be copied from
            target net (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
         """
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def train_DQN(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        # we need the max predicted Q values for next states from our target model
        Q_targets_net = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute the expected Q values
        Q_targets = rewards + (gamma * Q_targets_net * (1 - dones))  # done = 0 for False and 1 True
        # Q from policy
        Q_expected = self.policy_net(states).gather(1, (actions.type(torch.LongTensor)))
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------------- update target network ---------------------------------- #
        self.soft_update(policy_net=self.policy_net, target_net=self.target_net, tau=self.tau)


DQN_on = True
Double_on = True
Dueling_on = True
Noisy_on = True
PER_on = True
Multi_on = True
Rainbow_one = False
Rainbow_two = False

A3C_ON = False

path_scores = "/Users/willemvandemierop/Documents/Master AI/Pycharm/DL Optimization/Lunar_latest_code_21st_april/scores/"

######################## scores plot #########################
df = pd.read_csv(path_scores + "scores_DQN_3890_episodes.csv")
df2 = pd.read_csv(path_scores + "scores_DQN_Double_camb.csv")
df3 = pd.read_csv(path_scores + "scores_DQN_Dueling.csv")
df4 = pd.read_csv(path_scores + "scores_Noisy_correct.csv")
df5 = pd.read_csv(path_scores +  "scores_Original_DQN_PER.csv")
df6 = pd.read_csv(path_scores + "scores_3N_step_22_V1.csv")


fig, ax = plt.subplots(1,2,figsize = (14,5))
window = 50
print("last value DQN",df.iloc[-1])
print("last value double DQN",df2.iloc[-1])
rolling_mean = pd.Series(df['Scores']).rolling(window).mean()
std = pd.Series(df['Scores']).rolling(window).std()
rolling_mean2 = pd.Series(df2['Scores']).rolling(window).mean()
std2 = pd.Series(df2['Scores']).rolling(window).std()
rolling_mean3 = pd.Series(df3['Scores']).rolling(window).mean()
std3 = pd.Series(df3['Scores']).rolling(window).std()
rolling_mean4 = pd.Series(df4['Scores']).rolling(window).mean()
std4 = pd.Series(df4['Scores']).rolling(window).std()
rolling_mean5 = pd.Series(df5['Scores']).rolling(window).mean()
std5 = pd.Series(df5['Scores']).rolling(window).std()
rolling_mean6 = pd.Series(df6['Scores']).rolling(window).mean()
std6 = pd.Series(df6['Scores']).rolling(window).std()

if DQN_on:
    ax[0].plot(rolling_mean, label = 'DQN')
    ax[0].fill_between(range(len(pd.Series(df['Scores']))), rolling_mean - std, rolling_mean + std, color = 'blue', alpha = 0.1)
if Double_on:
    ax[0].plot(rolling_mean2, label = 'Double DQN')
    ax[0].fill_between(range(len(pd.Series(df2['Scores']))), rolling_mean2 - std2, rolling_mean2 + std2, color='orange',
                       alpha=0.1)
if Dueling_on:
    ax[0].plot(rolling_mean3, label = 'Dueling DQN')
    ax[0].fill_between(range(len(pd.Series(df3['Scores']))), rolling_mean3 - std3, rolling_mean3 + std3, color = 'green', alpha = 0.1)

if Noisy_on:
    ax[0].plot(rolling_mean4, label = 'Noisy DQN', color = 'red')
    ax[0].fill_between(range(len(pd.Series(df4['Scores']))), rolling_mean4 - std4, rolling_mean4 + std4, color='red',
                       alpha=0.1)
if PER_on:
    ax[0].plot(rolling_mean5, label = 'DQN PER', color = 'purple')
    ax[0].fill_between(range(len(pd.Series(df5['Scores']))), rolling_mean5 - std5, rolling_mean5 + std5, color='purple',
                       alpha=0.1)
if Multi_on:
    ax[0].plot(rolling_mean6, label = 'DQN Multi step', color = 'magenta')
    ax[0].fill_between(range(len(pd.Series(df6['Scores']))), rolling_mean6 - std6, rolling_mean6 + std6, color = 'magenta', alpha = 0.1)

ax[0].set_title('Scores moving average ({}-episode window)'.format(window))
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("Score")
ax[0].legend(loc = 'lower right')


######################## Episode length plot ###############
rolling_mean_length = pd.Series(df['Episode length']).rolling(window).mean()
std_length = pd.Series(df['Episode length']).rolling(window).std()
rolling_mean_length2 = pd.Series(df2['Episode length']).rolling(window).mean()
std_length2 = pd.Series(df2['Episode length']).rolling(window).std()
rolling_mean_length3 = pd.Series(df3['Episode length']).rolling(window).mean()
std_length3 = pd.Series(df3['Episode length']).rolling(window).std()
rolling_mean_length4 = pd.Series(df4['Episode length']).rolling(window).mean()
std_length4 = pd.Series(df4['Episode length']).rolling(window).std()
rolling_mean_length5 = pd.Series(df5['Episode length']).rolling(window).mean()
std_length5 = pd.Series(df5['Episode length']).rolling(window).std()
rolling_mean_length6 = pd.Series(df6['Episode length']).rolling(window).mean()
std_length6 = pd.Series(df6['Episode length']).rolling(window).std()


if DQN_on:
    ax[1].plot(rolling_mean_length, label = 'DQN')
    ax[1].fill_between(range(len(pd.Series(df['Episode length']))), rolling_mean_length - std, rolling_mean_length + std, color = 'blue', alpha = 0.1)

if Double_on:
    ax[1].plot(rolling_mean_length2, label = 'Double DQN')
    ax[1].fill_between(range(len(pd.Series(df2['Episode length']))), rolling_mean_length2 - std2, rolling_mean_length2 + std2, color='orange',
                       alpha=0.1)
if Dueling_on:
    ax[1].plot(rolling_mean_length3, label = 'Dueling DQN')
    ax[1].fill_between(range(len(pd.Series(df3['Episode length']))), rolling_mean_length3 - std3, rolling_mean_length3 + std3, color = 'green', alpha = 0.1)

if Noisy_on:
    ax[1].plot(rolling_mean_length4, label = 'Noisy DQN', color = 'red')
    ax[1].fill_between(range(len(pd.Series(df4['Episode length']))), rolling_mean_length4 - std4, rolling_mean_length4 + std4, color='red',
                       alpha=0.1)
if PER_on:
    ax[1].plot(rolling_mean_length5, label = 'DQN PER', color = 'purple')
    ax[1].fill_between(range(len(pd.Series(df5['Episode length']))), rolling_mean_length5 - std5, rolling_mean_length5 + std5, color='purple',
                       alpha=0.1)
if Multi_on:
    ax[1].plot(rolling_mean_length6, label = 'DQN Multi step', color = 'magenta')
    ax[1].fill_between(range(len(pd.Series(df6['Episode length']))), rolling_mean_length6 - std6, rolling_mean_length6 + std6, color = 'magenta', alpha = 0.1)

ax[1].set_title('Episode Length moving average ({}-episode window)'.format(window))
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Episode Length")
ax[1].legend(loc = 'lower right')
fig.subplots_adjust(hspace=1)
plt.show()



device = 'cpu'
# let's watch what a smart agent does.
lunar_agent_smart = Lunar_agent(state_size=env.observation_space.shape[0], action_size=4, seed=0)
lunar_agent_smart.policy_net.load_state_dict(torch.load('checkpoint_lunar_agent_5000.pth', map_location='cpu'))

'''
state = env.reset()
tot_rew = 0
done = False
while not done:
    action = lunar_agent_smart.get_action(state, epsilon=0)
    state, reward, done, info = env.step(action)
    env.state = state
    tot_rew += reward
    env.render()


def render_frames_with_env(env):

    frames = []
    lunar_agent_smart.policy_net.eval()
    for i in range(5):
        state = env.reset()
        with torch.no_grad():
            for j in range(1000):
                action = lunar_agent_smart.get_action(state, epsilon = 0)
                state = torch.from_numpy(state).float()
                state, reward, is_done, info = env.step(action)
                frames.append(env.render(mode="rgb_array") )
                if is_done:
                    break

    return frames

def create_animation(frames):
    fig = plt.figure()
    plt.axis("off")
    im = plt.imshow(frames[0], animated=True)

    def updatefig(i):
        im.set_array(frames[i])
        return im,
    ani = animation.FuncAnimation(fig, updatefig, frames=len(frames), interval=20, blit=True)
    plt.close()
    return ani

ani = create_animation(render_frames_with_env(env))
ani.save('Lunar_Lander.gif', writer=animation.PillowWriter(fps=20))

env.close()

'''