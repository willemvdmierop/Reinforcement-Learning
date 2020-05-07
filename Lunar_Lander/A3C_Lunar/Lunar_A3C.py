# this code is based on following githubs:
# https://github.com/MorvanZhou/pytorch-A3C
# https://github.com/ikostrikov/pytorch-a3c

import time
from collections import deque
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transform
import gym
import torch.multiprocessing as mp
import warnings
import Lunar_A3C_model
import A3C_optim
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, shared_model, counter, lock, optimizer=None, num_steps=20, env_name="LunarLander-v2",
          lr=0.0001, max_episode_length=1000000, gamma=0.99, GAE_lambda=1,
          entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=50):
    torch.manual_seed(seed + rank)
    env = gym.make(env_name)
    env.seed(seed + rank)
    model = Lunar_A3C_model.ActorCritic(state_size=env.observation_space.shape[0], action_size=env.action_space.n,
                                        hidden_size=128)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=lr)

    model.train()
    state = env.reset()
    state = torch.from_numpy(state)
    scores_window = deque(maxlen=100)
    done = True
    episode_length = 0
    episodes = 0
    scores = []
    lengths = []
    while True:
        episodes += 1
        # sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 128)
            hx = torch.zeros(1, 128)
        else:
            cx = cx.detach()
            hx = hx.detach()
        values = []
        log_probs = []
        rewards = []
        entropies = []
        score = 0
        for step in range(num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)
            # max_episode_length = 1000 for lunar lander standar gym
            state, reward, done, _ = env.step(action.item())
            done = done or episode_length >= max_episode_length
            score += reward
            with lock:
                counter.value += 1
            if done:
                scores.append(score)
                scores_window.append(score)
                lengths.append(episode_length)
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        Gen_Adv_Est = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + gamma * values[i + 1] - values[i]
            Gen_Adv_Est = Gen_Adv_Est * gamma * GAE_lambda + delta_t

            policy_loss = policy_loss - log_probs[i] * Gen_Adv_Est.detach() - entropy_coef * entropies[i]

        optimizer.zero_grad()
        (policy_loss + value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
        if Finish.value == 1:
            scores = np.array(scores)
            lengthlist = np.array(lengths)
            df = pd.DataFrame(scores, columns=['Scores'])
            df['Episode length'] = lengthlist
            #print('\n', df.head())
            df.to_csv("A3C_scores/scores_A3C_Rank_" + str(rank) + "_episodes.csv")
            print("terminating the multiprocess")
            break


def test(rank, shared_model, counter, env_name="LunarLander-v2", max_episode_length=1000000):
    torch.manual_seed(seed + rank)
    env = gym.make(env_name)
    env.seed(seed + rank)
    model = Lunar_A3C_model.ActorCritic(state_size=env.observation_space.shape[0], action_size=env.action_space.n,
                                        hidden_size=128)

    model.eval()
    scores = []
    lengths = []
    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    prints = 0
    start_time = time.time()
    done = True
    episode_length = 0
    scores_window = deque(maxlen = 100)
    while True:
        episode_length += 1
        # sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 128)
            hx = torch.zeros(1, 128)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

        prob = F.softmax(logit, dim = -1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= max_episode_length
        reward_sum += reward

        if done:
            scores.append(reward_sum)
            lengths.append(episode_length)
            scores_window.append(reward_sum)
            if prints % 400 == 0:
                print("\rTime {}, \tnum steps {}, \tFPS {:.0f}, \tepisode reward {:.2f}, \tepisode length {}".format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    counter.value, counter.value / (time.time() - start_time),
                    reward_sum, episode_length))
            if prints % 50 == 0:
                print("\rTime {}, \tnum steps {}, \tFPS {:.0f}, \tepisode reward {:.2f}, \tepisode length {}".format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    counter.value, counter.value / (time.time() - start_time),
                    reward_sum, episode_length), end = "")

            if np.mean(scores_window) > 200:
                scores = np.array(scores)
                lengthlist = np.array(lengths)
                df = pd.DataFrame(scores, columns=['Scores'])
                df['Episode length'] = lengthlist
                df.to_csv("A3C_scores/scores_A3C_Test_episodes_lr_001_40.csv")
                print("\nAverage test scores are above 200 so we are terminating")
                Finish.value = 1
                break
            reward_sum = 0
            episode_length = 0
            state = env.reset()
            prints += 1
        state = torch.from_numpy(state)




# =========================================== let's train the A3C model ================================================
# https://pytorch.org/docs/stable/notes/multiprocessing.html
if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    env = gym.make("LunarLander-v2")
    env.seed(0)
    print("action space", env.action_space)  # actions: [do nothing, fire left, fire main, fire right]
    print("observation spac", env.observation_space)  # Box(8, )
    #############
    lr = 0.001  ##
    num_steps = 40 ## Number of forward steps in A3C
    #############
    num_processes = os.cpu_count() - 2
    print(f'we are using {num_processes} number of processes.\n')
    shared_model = Lunar_A3C_model.ActorCritic(state_size=env.observation_space.shape[0], action_size=env.action_space.n,
                                               hidden_size=128)
    shared_model.share_memory()

    optimizer = A3C_optim.SharedAdam(shared_model.parameters(), lr=lr)
    optimizer.share_memory()
    processes = []
    counter = mp.Value('i', 0)
    Finish = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target = test, args = (num_processes, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(0, num_processes):
        p = mp.Process(target = train, args = (rank, shared_model, counter, lock, optimizer, num_steps))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
