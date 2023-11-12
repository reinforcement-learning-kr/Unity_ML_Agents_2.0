import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel\
                             import EnvironmentParametersChannel

"""
note that this code is tested on "Cooperative Push Block" environment with one environment instance. (not distributed environment)

observation of 3 agents --> (3, 20, 20)
also it uses 6 detectable tags: wall, agent, goal, blockSmall, blockLarge, blockVeryLarge --> (3, 20, 20, 6)

we have 2 options:
1. use CNN to extract features from grid observation
2. vectorize grid observation and use MLP

[NOTE]
* dec.obs에 에이전트 id대로 순차적으로 observation이 들어가 있는 것을 전제로 함
    -> dictionary로 agent id와 observation을 매핑하면 될 듯

* MHA를 사용할 지, 단순히 Attention Mechanism을 사용할 지 고민

* distributed envrionment을 가정해야 하나?
"""

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
    
# Attention Mechanism 추가
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        scores = torch.tanh(self.attention(x))
        attention_weights = torch.softmax(scores, dim=1)
        return attention_weights

# Centralized Critic Model에 Attention 추가
class CentralizedCriticWithAttention(CentralizedCritic):
    def __init__(self, state_dim, action_dim, hidden_dim, num_agents):
        super(CentralizedCriticWithAttention, self).__init__(state_dim, action_dim, hidden_dim, num_agents)
        self.attention = Attention(hidden_dim)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = torch.relu(self.fc1(x))
        
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        x = self.fc2(x)
        return x
    
class CentralizedCriticWithMultiheadAttention(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_agents, num_heads):
        super(CentralizedCriticWithMultiheadAttention, self).__init__()
        self.fc1 = nn.Linear(num_agents * (state_dim + action_dim), hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = torch.relu(self.fc1(x)).unsqueeze(0)  # Add time dimension for MultiheadAttention
        x, _ = self.multihead_attn(x, x, x)
        x = x.squeeze(0)  # Remove time dimension
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
    
class DecentralizedActorWithAttention(DecentralizedActor):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DecentralizedActorWithAttention, self).__init__(state_dim, action_dim, hidden_dim)
        self.attention = Attention(hidden_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        
        # Attention weights 계산
        attention_weights = self.attention(x)
        
        # Attention 적용
        x = x * attention_weights

        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# parameters
num_agents = 3 * 2             # 3 agents, 2 distributed environments
state_dim = 20 * 20 * 6        # 20x20 grid, 6 tags for a single agent
action_dim = 7
hidden_dim = 128

"""
actors = [DecentralizedActor(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
# 기존의 CentralizedCritic을 CentralizedCriticWithAttention으로 교체
critic = CentralizedCriticWithAttention(state_dim, action_dim, hidden_dim, num_agents)

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
"""

if __name__ == '__main__':
    
    engine_configuration_channel = EngineConfigurationChannel()
    env_name = None
    env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel])
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

    dec, term = env.get_steps(behavior_name)

    actors = [DecentralizedActor(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
    critic = CentralizedCriticWithAttention(state_dim, action_dim, hidden_dim, num_agents)

    actor_optimizers = [optim.Adam(actor.parameters(), lr=0.001) for actor in actors]
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

    num_episodes = 10

    # 에피소드 시작
    for episode in range(num_episodes):
        env.reset()
        done = False

        while not done:
            dec, term = env.get_steps(behavior_name)

            # agent_id와 observation을 매핑하는 dictionary 생성
            agent_obs_map = {agent_id: obs for agent_id, obs in zip(dec.agent_id, dec.obs[0])}

            actions = []
            for agent_id, actor in zip(dec.agent_id, actors):
                obs_agent = torch.tensor(agent_obs_map[agent_id], dtype=torch.float32).view(1, -1)
                action = actor(obs_agent).detach()
                action = torch.argmax(action, dim=1).item()
                actions.append(action)
                # print(f"Agent {agent_id} chose action {action}.")

            actions = np.array(actions, dtype=np.int32).reshape(-1, 1)
            
            action_tuple = ActionTuple()
            action_tuple.add_discrete(actions)
            env.set_actions(behavior_name, action_tuple)

            env.step()

            dec, term = env.get_steps(behavior_name)
            next_obs = dec.obs[0]
            rewards = dec.reward
            done = term.interrupted[0] if term.interrupted.size > 0 else False

        print(f"Episode {episode + 1} finished.")
