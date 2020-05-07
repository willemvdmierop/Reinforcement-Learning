import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTMCell(self.state_size, self.hidden_size)

        #self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.critic_linear = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64,1))

        self.actor_linear = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64,self.action_size))
        #self.actor_linear = nn.Linear(self.hidden_size, self.action_size)

        self.apply(weights_init)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, x):
        # hx contains the next hidden state for each element in the bach
        # cx contains the next cell state for each element in the bach
        x , (hx, cx) = x
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx,cx)
