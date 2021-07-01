import numpy as np
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
    import EngineConfigurationChannel
from collections import deque

import random

# set param's
state_size = 9
action_size = 3

load_model = False
train_mode = True

discount_factor = 0.99
actor_lr = 1e-4
critic_lr = 5e-4

batch_size = 8
mem_maxlen = 50000

tau = 1e-3

mu = 0
theta = 1e-3
sigma = 2e-3

start_train_episode = 5
run_episode = 500
test_episode = 100

env_config = {"gridSize": 5, "numPlusGoals": 1, "numExGoals": 1}
VISUAL_OBS = 0
GOAL_OBS = 1
VECTOR_OBS = 2
OBS = VECTOR_OBS

game = "Drone"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"

date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/DDPG/{date_time}"
load_path = f"./saved_models/{game}/DDPG/20210217000848"

# set device ( GPU vs CPU )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = np.ones(action_size) * mu

    def sample(self):
        dx = theta * (mu - self.X) + sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X


class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.pi = torch.nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.pi(x))


class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(state_size, 256)
        self.fc2 = torch.nn.Linear(256+action_size, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class DDPGAgent():
    def __init__(self):
        self.actor_local = Actor().to(device)
        self.actor_target = Actor().to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(), lr=actor_lr)

        self.critic_local = Critic().to(device)
        self.critic_target = Critic().to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(), lr=critic_lr)

        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)
        self.writer = SummaryWriter(save_path)

    def get_action(self, state):

        pi = self.actor_local(torch.FloatTensor(state).to(device))
        action = torch.multinomial(pi, num_samples=1).cpu().numpy()

        noise = self.OU.sample()

        return action + noise if train_mode else action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        mini_batch = random.sample(self.memory, batch_size)
        states = np.asarray([sample[0]
                            for sample in mini_batch])
        actions = np.asarray([sample[1]
                             for sample in mini_batch])
        rewards = np.asarray([sample[2]
                             for sample in mini_batch])
        next_states = np.asarray([sample[3]
                                 for sample in mini_batch])
        dones = np.asarray([sample[4] for sample in mini_batch])

        # states = torch.from_numpy(states.astype(np.double))
        # actions = torch.from_numpy(actions.astype(np.double))
        # rewards = torch.from_numpy(rewards.astype(np.double))
        # next_states = torch.from_numpy(next_states.astype(np.double))
        # dones = torch.from_numpy(dones.astype(np.double))

        # states = (torch.from_numpy(
        #     states.astype(np.double))).float()

        # actions = (torch.from_numpy(
        #     actions.astype(np.double))).float()

        # rewards = (torch.from_numpy(
        #     rewards.astype(np.double))).float()

        # next_states = (torch.from_numpy(
        #     next_states.astype(np.double))).float()

        # dones = (torch.from_numpy(
        #     dones.astype(np.double))).float()

        states = np.squeeze(states.astype('float32'))
        actions = np.squeeze(actions.astype('float32'))
        rewards = np.squeeze(rewards.astype('float32'))
        next_states = np.squeeze(next_states.astype('float32'))
        dones = np.squeeze(dones.astype('float32'))

        # print('------------------')
        # print(states.shape)
        # print(actions.shape)
        #  print(rewards.shape)
        # print(next_states.shape)
        #  print(dones.shape)
        # print('------------------')

        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones)

        actions_next = self.actor_target(next_states)

        Q_targets_next = self.critic_target(next_states, actions_next)

        Q_targets = rewards + (discount_factor * Q_targets_next * (1 - dones))

        Q_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target_actor_actions = self.actor_target(next_states)
        # target_critic_predict_qs = self.critic_target(
        #     next_states, target_actor_actions)

        # target_qs = np.asarray([reward + discount_factor * (1 - done) * target_critic_predict_q
        #                         for reward, target_critic_predict_q, done in zip(rewards, target_critic_predict_qs, dones)])

        # expected_qs = self.critic_local(states, actions)

        # critic_loss = F.mse_loss(expected_qs, target_qs)

        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        # expected_action = self.actor_local(states)
        # actor_loss = -self.critic_local(states, expected_action).mean()

        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)


if __name__ == '__main__':

    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=None, side_channels=[
                           engine_configuration_channel])

    agent = DDPGAgent()
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
    dec, term = env.get_steps(behavior_name)

    step = 0

    for episode in range(1000):

        state = dec.obs[0]
        episode_rewards = 0
        done = False

        while not done:
            step += 1

            action = agent.get_action(state)

            # print(state)

            action_tuple = ActionTuple()
            action_tuple.add_continuous(action)
            env.set_actions(behavior_name, action_tuple)
            env.step()

            dec, term = env.get_steps(behavior_name)

            done = len(term.agent_id) > 0
            reward = term.reward if done else dec.reward
            next_state = term.obs[0] if done else dec.obs[0]

            episode_rewards += reward[0]

            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

            if episode > start_train_episode and train_mode:
                agent.train_model()

        print(f'ep : {episode} / reward : {episode_rewards}')

    env.close()
