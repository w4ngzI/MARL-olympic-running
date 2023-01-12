import os
import sys
import numpy as np
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

father_path = path.dirname(__file__)
sys.path.append(str(os.path.dirname(father_path)))

from rl_trainer.algo.network import Actor, CNN_Actor, CNN_Critic, Critic, DDPGActor, DDPGCritic
from torch.utils.tensorboard import SummaryWriter


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Args:
    gamma = 0.99
    tau = 0.005

    buffer_capacity = 1000000
    batch_size = 100

    action_space = 2
    state_space = 625

    update_iteration = 200
    update_delay = 2
    policy_noise = 0.2
    noise_clip = 0.5

args = Args()


class td3:
    
    action_space = args.action_space
    state_space = args.state_space

    batch_size = args.batch_size
    buffer_capacity = args.buffer_capacity

    update_iteration = args.update_iteration

    gamma = args.gamma
    tau = args.tau


    def __init__(
        self,
        device: str = "cpu",
        run_dir: str = None,
        writer: SummaryWriter = None,
    ):
        
        self.args = args
        self.device = device

        self.update_delay = args.update_delay

        self.actor = DDPGActor(self.state_space, self.action_space).to(device)
        self.actor_target = DDPGActor(self.state_space, self.action_space).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic_1 = DDPGCritic(self.state_space, self.action_space).to(device)
        self.critic_target_1 = DDPGCritic(self.state_space, self.action_space).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = DDPGCritic(self.state_space, self.action_space).to(device)
        self.critic_target_2 = DDPGCritic(self.state_space, self.action_space).to(device)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=1e-3)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer(self.buffer_capacity)

        self.run_dir = run_dir
        self.writer = writer
        self.IO = True if (run_dir is not None) else False

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def choose_action(self, state, train=False):
        actions = self.select_action(state)
        return [[actions[0]], [actions[1]]]

    def update(self):

        for it in range(self.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1-d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(self.device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action[:,0] = next_action[:,0].clamp(-100, 200)
            next_action[:,1] = next_action[:,1].clamp(-30, 30)


            # Compute the target Q value
            target_Q_1 = self.critic_target_1(next_state, next_action)
            target_Q_2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q_1,target_Q_2)
            target_Q = reward + (done * self.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q_1 = self.critic_1(state, action)

            # Compute critic loss
            critic_loss_1 = F.mse_loss(current_Q_1, target_Q)
            self.writer.add_scalar('Loss/critic_loss_1', critic_loss_1, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer_1.step()

            # Get current Q estimate
            current_Q_2 = self.critic_2(state, action)

            # Compute critic loss
            critic_loss_2 = F.mse_loss(current_Q_2, target_Q)
            self.writer.add_scalar('Loss/critic_loss_2', critic_loss_2, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optimizer_2.step()


            if it%self.update_delay==0:
                # Compute actor loss
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1


    def save(self, save_path, episode):
        base_path = os.path.join(save_path, "trained_model")
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_critic_path)


    def load(self, run_dir, episode):
        print(f"\nBegin to load model: ")
        print("run_dir: ", run_dir)
        base_path = os.path.dirname(os.path.dirname(__file__))
        print("base_path: ", base_path)
        algo_path = os.path.join(base_path, "models/olympics-running/ddpg")
        run_path = os.path.join(algo_path, run_dir)
        run_path = os.path.join(run_path, "trained_model")
        model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
        print(f"Actor path: {model_actor_path}")
        print(f"Critic path: {model_critic_path}")

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=self.device)
            critic = torch.load(model_critic_path, map_location=self.device)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f"Model not founded!")