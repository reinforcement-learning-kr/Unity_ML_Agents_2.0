import torch
import torch.nn as nn
import torch.optim as optim

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel\
                             import EnvironmentParametersChannel

torch.autograd.set_detect_anomaly(True)

# Centralized Critic Model
class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_agents):
        super(CentralizedCritic, self).__init__()
        self.fc1 = nn.Linear(num_agents * (state_dim + action_dim), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Decentralized Actor Model
class DecentralizedActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DecentralizedActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# Initialize models and optimizers
num_agents = 3
state_dim = 24
action_dim = 4
hidden_dim = 128

actors = [DecentralizedActor(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
critic = CentralizedCritic(state_dim, action_dim, hidden_dim, num_agents)

actor_optimizers = [optim.Adam(actor.parameters(), lr=0.001) for actor in actors]
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# Dummy data for example
states = torch.rand((1, num_agents * state_dim))
actions = torch.rand((1, num_agents * action_dim))

# Forward pass and training code
predicted_actions = torch.cat([actor(state) for actor, state in zip(actors, states.split(state_dim, dim=1))], dim=1)
value = critic(states, actions)

# Compute counterfactual baseline
counterfactual_value = critic(states, predicted_actions.detach())

# Compute advantage
advantage = value - counterfactual_value

# Update actor and critic
log_probs = [-torch.log(predicted_action) for predicted_action in predicted_actions.split(action_dim, dim=1)]
actor_losses = [log_prob * advantage.clone().detach() for log_prob in log_probs]
actor_losses = [actor_loss.mean() for actor_loss in actor_losses]
critic_loss = advantage ** 2

# Update actors
for actor_optimizer in actor_optimizers:
    actor_optimizer.zero_grad()

actor_losses = []
for i, (actor, actor_optimizer, state) in enumerate(zip(actors, actor_optimizers, states.split(state_dim, dim=1))):
    predicted_action = actor(state)
    log_prob = -torch.log(predicted_action)
    actor_loss = (log_prob * advantage.clone().detach()).mean()
    actor_losses.append(actor_loss)

# 모든 actor_loss를 더하고 backward() 호출
total_actor_loss = sum(actor_losses)
total_actor_loss.backward()

# 각 actor의 optimizer로 step() 호출
for actor_optimizer in actor_optimizers:
    actor_optimizer.step()

# Update critic
critic_optimizer.zero_grad()
critic_loss.backward()
critic_optimizer.step()
