import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CNN_encoder(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Flatten(),
        )

    def forward(self, view_state):
        x = self.net(view_state)
        return x


class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64, cnn=False):
        super(Actor, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob

class Actor_sac(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64, cnn=False):
        super(Actor_sac, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch,-1)
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        c = Categorical(action_prob)
        sampled_action = c.sample()
        greedy_action = torch.argmax(action_prob)
        return sampled_action, action_prob, greedy_action

class Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64, cnn=False):
        super(Critic, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch,-1)
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value


class Critic_sac(nn.Module):
    def __init__(self, state_space,action_space, hidden_size=64, cnn=False):
        super(Critic_sac, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.state_value = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch,-1)
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value

class CNN_Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(CNN_Actor, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2)
        # self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = 3, stride = 1)
        # self.flatten = nn.Flatten()
        self.net = Net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
        )

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, action_space)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b,1,25,25)
        x = self.net(x)
        x = torch.relu(self.linear1(x))
        action_prob = F.softmax(self.linear2(x), dim=-1)
        x = x.squeeze(0)
        return action_prob


class CNN_Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64):
        super(CNN_Critic, self).__init__()

        self.net = Net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
        )

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b,1,25,25)
        x = self.net(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        x = x.squeeze(0)
        return x


class CNN_CategoricalActor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(CNN_CategoricalActor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
        )

        self.linear1 = nn.Linear(128, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = self.net(x)
        x = F.relu(self.linear1(x))
        action_prob = F.softmax(self.linear2(x), dim=-1)
        c = Categorical(action_prob)
        sampled_action = c.sample()
        greedy_action = torch.argmax(action_prob)
        return sampled_action, action_prob, greedy_action


class CNN_Critic2(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(CNN_Critic2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
        )
        self.linear1 = nn.Linear(128, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = self.net(x)
        x = F.relu(self.linear1(x))
        return self.linear2(x)



class DDPGActor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64, cnn=False):
        super(DDPGActor, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder()
        self.fc1 = nn.Linear(state_space, 400)
        self.fc2 = nn.Linear(400, 300)
        self.action_head = nn.Linear(300, action_space)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.action_head(x)

        x[:, 0] = 50 + 150 * torch.tanh(x[:, 0])
        x[:, 1] = 30 * torch.tanh(x[:, 1])
        return x


class DDPGCritic(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64, cnn=False):
        super(DDPGCritic, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder()
        self.fc1 = nn.Linear(state_space + action_space, 400)
        self.fc2 = nn.Linear(400, 300)
        self.state_value = nn.Linear(300, 1)

    def forward(self, x, u):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.fc1(torch.cat([x, u], 1)))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value