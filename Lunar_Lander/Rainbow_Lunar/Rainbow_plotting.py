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
device = 'cpu'
env = gym.make("LunarLander-v2")
env.seed(0)

# ================================= This first part creates the plots ==================================

Multi_step_on = True
Dueling_Noisy_on = True
Dueling_PER_on = True
Dueling_Noisy_multi = True
Dueling_PER_multi_on = True
Dueling_noisy_PER_on = True
Dueling_Noisy_PER_Multi_on = True
A3C_ON = False


path_scores = "/Users/willemvandemierop/Documents/Master AI/Pycharm/DL Optimization/Lunar_latest_code_27st_april/scores/Rainbow_scores/"

######################## scores plot #########################
df = pd.read_csv(path_scores + "scores_3N_step_22_V1.csv")
df2 = pd.read_csv(path_scores + "scores_Dueling_Noisy.csv")
df3 = pd.read_csv(path_scores + "scores_DuelingQN_PER.csv")
df4 = pd.read_csv(path_scores + "scores_Dueling_Noisy_multi.csv")
df5 = pd.read_csv(path_scores +  "scores_Rainbow_Dueling_PER_multi.csv")
df6 = pd.read_csv(path_scores + "scores_Rainbow_Dueling_Noisy_PER.csv")
df7 = pd.read_csv(path_scores + "scores_Rainbow_Dueling_Noisy_PER_Multi.csv")
df8 = pd.read_csv("/Users/willemvandemierop/Documents/Master AI/Pycharm/DL Optimization/Lunar_latest_code_27st_april/scores/A3C_scores/scores_A3C_Test_episodes.csv")


fig, ax = plt.subplots(1,2,figsize = (18,8))
window = 50
print("last value multi step DQN",df.iloc[-1])
print("last value Dueling_Noisy",df2.iloc[-1])
print("last value DuelingQN_PER",df3.iloc[-1])
print("last value Dueling_Noisy_multi",df4.iloc[-1])
print("last value Dueling_PER_multi",df5.iloc[-1])
print("last value Dueling_Noisy_PER",df6.iloc[-1])
print("last value Dueling_Noisy_PER_Multi",df7.iloc[-1])
print("last value A3CR",df8.iloc[-1])
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
rolling_mean7 = pd.Series(df7['Scores']).rolling(window).mean()
std7 = pd.Series(df7['Scores']).rolling(window).std()
rolling_mean8 = pd.Series(df8['Scores']).rolling(window).mean()
std8 = pd.Series(df8['Scores']).rolling(window).std()
if Multi_step_on:
    ax[0].plot(rolling_mean, label = 'DQN Multi step', color = 'blue')
    ax[0].fill_between(range(len(pd.Series(df['Scores']))), rolling_mean - std, rolling_mean + std, color = 'blue', alpha = 0.1)

if Dueling_PER_on:
    ax[0].plot(rolling_mean3, label = 'Dueling PER', color = 'purple')
    ax[0].fill_between(range(len(pd.Series(df3['Scores']))), rolling_mean3 - std3, rolling_mean3 + std3, color = 'purple', alpha = 0.1)

if Dueling_PER_multi_on:
    ax[0].plot(rolling_mean5, label = 'Dueling PER Multi step', color = 'yellow')
    ax[0].fill_between(range(len(pd.Series(df5['Scores']))), rolling_mean5 - std5, rolling_mean5 + std5, color='yellow',
                       alpha=0.1)

if Dueling_Noisy_on:
    ax[0].plot(rolling_mean2, label = 'Dueling Noisy', color = 'orange')
    ax[0].fill_between(range(len(pd.Series(df2['Scores']))), rolling_mean2 - std2, rolling_mean2 + std2, color='orange',
                       alpha=0.1)

if Dueling_noisy_PER_on:
    ax[0].plot(rolling_mean6, label = 'Dueling Noisy PER', color = 'cyan')
    ax[0].fill_between(range(len(pd.Series(df6['Scores']))), rolling_mean6 - std6, rolling_mean6 + std6, color = 'cyan', alpha = 0.1)

if Dueling_Noisy_multi:
    ax[0].plot(rolling_mean4, label = 'Dueling Noisy Multi step', color = 'red')
    ax[0].fill_between(range(len(pd.Series(df4['Scores']))), rolling_mean4 - std4, rolling_mean4 + std4, color='red',
                       alpha=0.1)

if Dueling_Noisy_PER_Multi_on:
    ax[0].plot(rolling_mean7, label='Dueling Noisy PER Multi step', color = 'green')
    ax[0].fill_between(range(len(pd.Series(df7['Scores']))), rolling_mean7 - std7, rolling_mean7 + std7, color = 'green', alpha=0.1)

if A3C_ON:
    ax[0].plot(rolling_mean8, label='A3C', color = 'blue')
    ax[0].fill_between(range(len(pd.Series(df8['Scores']))), rolling_mean8 - std8, rolling_mean8 + std8, color = 'blue', alpha=0.1)

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
rolling_mean_length7 = pd.Series(df7['Episode length']).rolling(window).mean()
std_length7 = pd.Series(df7['Episode length']).rolling(window).std()
rolling_mean_length8 = pd.Series(df8['Episode length']).rolling(window).mean()
std_length8 = pd.Series(df8['Episode length']).rolling(window).std()

if Multi_step_on:
    ax[1].plot(rolling_mean_length, label = 'DQN Multi step', color = 'blue')
    ax[1].fill_between(range(len(pd.Series(df['Episode length']))), rolling_mean_length - std, rolling_mean_length + std, color = 'blue', alpha = 0.1)


if Dueling_PER_on:
    ax[1].plot(rolling_mean_length3, label = 'Dueling PER', color = 'purple')
    ax[1].fill_between(range(len(pd.Series(df3['Episode length']))), rolling_mean_length3 - std3, rolling_mean_length3 + std3, color = 'purple', alpha = 0.1)


if Dueling_PER_multi_on:
    ax[1].plot(rolling_mean_length5, label = 'Dueling PER Multi step', color = 'yellow')
    ax[1].fill_between(range(len(pd.Series(df5['Episode length']))), rolling_mean_length5 - std5, rolling_mean_length5 + std5, color='yellow',
                       alpha=0.1)

if Dueling_Noisy_on:
    ax[1].plot(rolling_mean_length2, label = 'Dueling Noisy', color = 'orange')
    ax[1].fill_between(range(len(pd.Series(df2['Episode length']))), rolling_mean_length2 - std2, rolling_mean_length2 + std2, color='orange',
                       alpha=0.1)

if Dueling_noisy_PER_on:
    ax[1].plot(rolling_mean_length6, label = 'Dueling Noisy PER', color = 'cyan')
    ax[1].fill_between(range(len(pd.Series(df6['Episode length']))), rolling_mean_length6 - std6, rolling_mean_length6 + std6, color = 'cyan', alpha = 0.1)

if Dueling_Noisy_multi:
    ax[1].plot(rolling_mean_length4, label='Dueling Noisy Multi step', color='red')
    ax[1].fill_between(range(len(pd.Series(df4['Episode length']))), rolling_mean_length4 - std4,
                       rolling_mean_length4 + std4, color='red',
                       alpha=0.1)

if Dueling_Noisy_PER_Multi_on:
    ax[1].plot(rolling_mean_length7, label = 'Dueling Noisy PER Multi step', color = 'green')
    ax[1].fill_between(range(len(pd.Series(df7['Episode length']))), rolling_mean_length7 - std7, rolling_mean_length7 + std7, color ='green', alpha = 0.1)

if A3C_ON:
    ax[1].plot(rolling_mean_length8, label = 'A3C', color = 'blue')
    ax[1].fill_between(range(len(pd.Series(df8['Episode length']))), rolling_mean_length8 - std8, rolling_mean_length8 + std8, color = 'blue', alpha = 0.1)

ax[1].set_title('Episode Length moving average ({}-episode window)'.format(window))
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Episode Length")
ax[1].legend(loc = 'lower right')
fig.subplots_adjust(hspace=1)
plt.show()


# ================================= This second part creates the gifs ==================================
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


# new 07
class DDQN(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_size=64, fc2_size=64):
        super(DDQN, self).__init__()
        self.num_actions = action_size
        fc3_1_size = fc3_2_size = 32
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        # calculate V(s)
        self.fc3_1 = nn.Linear(fc2_size, fc3_1_size)
        self.fc4_1 = nn.Linear(fc3_1_size, 1)
        # calculate A(s,a)
        self.fc3_2 = nn.Linear(fc2_size, fc3_2_size)
        self.fc4_2 = nn.Linear(fc3_2_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        val = F.relu(self.fc3_1(x))
        val = self.fc4_1(val)

        adv = F.relu(self.fc3_2(x))
        adv = self.fc4_2(adv)

        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        action = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.num_actions)
        return action


class Lunar_agent:
    def __init__(self, state_size, action_size, seed, batch_size=64, gamma=0.99, learning_rate=1e-4,
                 capacity=int(1e5), update_every=4, tau=1e-3, beta=0.4, ):
        self.state_size = state_size  # Observation array length = env.observation_space.shape[0]
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.capacity = capacity
        self.update_every = update_every
        self.tau = tau
        self.beta = beta
        self.Loss = 0
        # Q-Network
        self.policy_net = DDQN(self.state_size, self.action_size, seed=seed).to(device)
        self.target_net = DDQN(self.state_size, self.action_size, seed=seed).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # self.importance = importance #not used
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # should increase capacity size by x10 for PER
        self.memory = ReplayMemory(self.action_size, self.capacity, self.batch_size, self.seed)
        self.time_step = 0  # initialize time step for updating every predefined number of steps

    def step(self, state, action, reward, next_state, done):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)  ##new added extra bracket

        # We are updating every UPDATE_EVERY time steps
        self.time_step = (self.time_step + 1) % self.update_every
        if self.time_step == 0:
            # if enough samples are available in memory, take a random subset
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                # can include argument for beta decay
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

    def train_DQN(self, experiences, gamma):  # may need to add beta
        states, actions, rewards, next_states, dones, indices, weights = experiences
        # we need the max predicted Q values for next states from our target model
        Q_targets_net = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute the expected Q values
        Q_targets = rewards + (gamma * Q_targets_net * (1 - dones))  # done = 0 for False and 1 True
        # Q from policy
        Q_expected = self.policy_net(states).gather(1, (actions.type(torch.LongTensor).to(device)))

        loss = F.mse_loss(Q_expected, Q_targets)
        loss = (loss * weights)
        prios = loss + 1e-5
        loss = loss.mean()  # because backward needs a tensor containing a single element
        self.Loss = loss

        self.optimizer.zero_grad()
        loss.backward()
        self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()
        # ------------------------- update target network ---------------------------------- #
        self.soft_update(policy_net=self.policy_net, target_net=self.target_net, tau=self.tau)

    def get_Loss(self):
        return self.Loss

device = 'cpu'
# let's watch what a smart agent does.
lunar_agent_smart = Lunar_agent(state_size=env.observation_space.shape[0], action_size=4, seed=0)
lunar_agent_smart.policy_net.load_state_dict(torch.load('/Users/willemvandemierop/Google Drive/DeepLearning Optimization/Checkpoints_for_report/checkpoint_Rainbow_Dueling_PER_4000.pth', map_location='cpu'))

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
    ani = animation.FuncAnimation(fig, updatefig, frames=len(frames), interval=5, blit=True)
    plt.close()
    return ani

ani = create_animation(render_frames_with_env(env))
writergif = animation.PillowWriter(fps=40)
ani.save('Lunar_Lander_smart.gif', writer=writergif)

env.close()
