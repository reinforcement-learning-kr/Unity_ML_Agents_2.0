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
import tensorflow as tf

import random

# set param's
state_size = 9
action_size = 3

load_model = False
train_mode = True

batch_size = 128
mem_maxlen = 50000
discount_factor = 0.99
actor_lr = 1e-4
critic_lr = 5e-4
tau = 1e-3

mu = 0
theta = 1e-3
sigma = 2e-3

run_step = 50000 if train_mode else 0
test_step = 10000
train_start_step = 5000

print_interval = 10
save_interval = 100

# run_episode = 10000
# test_episode = 100

game = "Drone"
os_name = platform.system()
# if os_name == 'Windows':
#     env_name = f"../envs/{game}_{os_name}/{game}"
# elif os_name == 'Darwin':
#     env_name = f"../envs/{game}_{os_name}"
env_name = 'D:/unityProjects/DroneBuild/windows/Drone'

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
        self.X = np.ones((1, action_size), dtype=np.float32) * mu

    def sample(self):
        dx = theta * (mu - self.X) + sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X


class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.mu = torch.nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.mu(x))


class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(state_size, 256)
        self.fc2 = torch.nn.Linear(256+action_size, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=-1)
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

        if load_model == True:
            self.actor_local.load_state_dict(torch.load(load_path+'/actor'))
            self.critic_local.load_state_dict(torch.load(load_path+'/critic'))

        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)
        self.writer = tf.summary.create_file_writer(save_path)

    def get_action(self, state):

        action_ = self.actor_local(torch.FloatTensor(state))
        # action = self.actor_local(torch.FloatTensor(state).to(device))
        # action = torch.multinomial(pi, num_samples=1).cpu().numpy()

        action = action_.detach().numpy()
        noise = self.OU.sample()

        return action + noise if train_mode else action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        mini_batch = random.sample(self.memory, batch_size)
        states = np.stack([b[0] for b in mini_batch], axis=0)
        actions = np.stack([b[1] for b in mini_batch], axis=0)
        rewards = np.stack([b[2] for b in mini_batch], axis=0)
        next_states = np.stack([b[3] for b in mini_batch], axis=0)
        dones = np.stack([b[4] for b in mini_batch], axis=0).astype(np.float32)

        states, actions, rewards, next_states, dones = map(lambda x: torch.FloatTensor(
            x).to(device), [states, actions, rewards, next_states, dones])

        next_actions = self.actor_target(next_states)
        next_q = self.critic_target(next_states, next_actions)
        target_q = rewards + (1 - dones) * discount_factor * next_q
        q = self.critic_local(states, actions)
        critic_loss = F.mse_loss(target_q, q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, local_model, target_model):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self):
        print("... Save Model ...")
        torch.save(self.actor_local.state_dict(), save_path+'/actor')
        torch.save(self.critic_local.state_dict(), save_path+'/critic')

    def write_summray(self, score, actor_loss, critic_loss, step):
        with self.writer.as_default():
            tf.summary.scalar("run/score", score, step=step)
            tf.summary.scalar("model/actor_loss", actor_loss, step=step)
            tf.summary.scalar("model/critic_loss", critic_loss, step=step)


if __name__ == '__main__':

    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name, side_channels=[
                           engine_configuration_channel])

    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=10.0)
    dec, term = env.get_steps(behavior_name)

    step = 0

    agent = DDPGAgent()

    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0

    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()

            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(
                time_scale=1.0)

        state = dec.obs[0]
        # state = torch.FloatTensor(state).to(device)

        action = agent.get_action(state)

        # env.set_actions(behavior_name, action)
        # env.step()

        action_tuple = ActionTuple()
        action_tuple.add_continuous(action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name)
        done = len(term.agent_id) > 0
        reward = term.reward[0] if done else dec.reward[0]

        # if done:
        #     next_state = term.obs[0]
        #     next_state = torch.FloatTensor(next_state).to(device)

        # else:
        #     next_state = dec.obs[0]
        #     next_state = torch.FloatTensor(next_state).to(device)

        next_state = term.obs[0] if done else dec.obs[0]
        score += reward

        if train_mode and len(state) > 0:
            agent.append_sample(state[0], action[0], [
                                reward], next_state[0], [done])

        if train_mode and step > max(batch_size, train_start_step):
            actor_loss, critic_loss = agent.train_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        if done:
            episode += 1
            scores.append(score)
            score = 0

            if episode % print_interval == 0:
                mean_score = torch.mean(torch.FloatTensor(scores))
                mean_actor_loss = torch.mean(torch.FloatTensor(actor_losses))
                mean_critic_loss = torch.mean(torch.FloatTensor(critic_losses))
                agent.write_summray(
                    mean_score, mean_actor_loss, mean_critic_loss, step)
                actor_losses, critic_losses, scores = [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")

            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()
