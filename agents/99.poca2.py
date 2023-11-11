import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlagents_envs.environment import UnityEnvironment, ActionTuple

"""
1. 우선, Centralized Critic의 결과물을 내기 위해 현재 살아있는 에이전트들에 대한 
RSA 블록의 출력값을 평균내는 방식으로 구현했음.

2. 상태와 행동을 입력으로 받는 Embedding Network도 구현했음.
"""

state_size = 6
action_size = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingNetwork(nn.Module):
    def __init__(self, observation_dim, embedding_dim):
        super(EmbeddingNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear_q = nn.Linear(observation_dim, embedding_dim)
        self.linear_k = nn.Linear(observation_dim, embedding_dim)
        self.linear_v = nn.Linear(observation_dim, embedding_dim)

    def forward(self, observation):
        observation = torch.tensor(observation).float().to(device)
        observation = torch.flatten(observation, start_dim=1)
        Q = self.linear_q(observation).view(-1, 3, self.embedding_dim) # seq_length를 3으로 설정
        K = self.linear_k(observation).view(-1, 3, self.embedding_dim)
        V = self.linear_v(observation).view(-1, 3, self.embedding_dim)
        return Q, K, V

class EmbeddingNetwork2(nn.Module):
    def __init__(self, observation_dim, action_dim, embedding_dim):
        super(EmbeddingNetwork2, self).__init__()
        self.embedding_dim = embedding_dim

        # 각각의 q, k, v를 위한 레이어
        self.linear_q = nn.Linear(observation_dim + action_dim, embedding_dim)
        self.linear_k = nn.Linear(observation_dim + action_dim, embedding_dim)
        self.linear_v = nn.Linear(observation_dim + action_dim, embedding_dim)

    def forward(self, observation, action):
        # 관찰과 행동을 결합
        combined = torch.cat([observation, action], dim=-1)

        # Q, K, V 생성
        Q = self.linear_q(combined).view(-1, 3, self.embedding_dim)
        K = self.linear_k(combined).view(-1, 3, self.embedding_dim)
        V = self.linear_v(combined).view(-1, 3, self.embedding_dim)

        return Q, K, V

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, Q, K, V):
        attn_output, _ = self.multihead_attn(Q, K, V)
        return attn_output

class CentralizedCritic(nn.Module):
    def __init__(self, observation_dim, num_heads, hidden_dim):
        super(CentralizedCritic, self).__init__()

        # Embedding Network 
        self.embedding_network = EmbeddingNetwork(observation_dim, hidden_dim)

        # Multi-Head Attention
        self.multi_head_attention = MultiHeadAttention(hidden_dim, num_heads)

        # Critic Network
        self.critic_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, observation):
        Q, K, V = self.embedding_network(observation)
        attn_output = self.multi_head_attention(Q, K, V)
        values = self.critic_network(attn_output)

        # in here, the shape of output should be (num_active_agents, 1)
        # thus, we can use the mean of the output as the value of the state
        value = values.mean()

        return value

class Agent(nn.Module):
    def __init__(self, observation_dim, action_dim, num_heads, hidden_dim):
        super(Agent, self).__init__()
        
        # Embedding Network
        self.embedding_network = EmbeddingNetwork(observation_dim, hidden_dim)
        
        # Multi-Head Attention
        self.multi_head_attention = MultiHeadAttention(hidden_dim, num_heads)
        
        # Action Network
        self.action_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, observation):
        # Embed the observation into Q, K, V
        Q, K, V = self.embedding_network(observation)

        # Apply Multi-Head Attention
        attn_output = self.multi_head_attention(Q, K, V)
        
        # Generate Action Probabilities
        action_prob = F.softmax(self.action_network(attn_output), dim=-1)
        
        return action_prob

    def act(self, observation):
        # Get action probabilities
        action_prob = self.forward(observation)
        
        # squeeze the batch dimension
        action_prob = action_prob.squeeze()

        # Sample action from the probabilities
        # action = torch.multinomial(action_prob, 1).item()
        action = torch.multinomial(action_prob, 1)

        return action


if __name__ == '__main__':
    # Unity 환경 경로 설정
    env_name = './envs/EscapeRoom_Windows/EscapeRoom.exe'
    env = UnityEnvironment(file_name=None)
    env.reset()

    # 유니티 브레인 설정
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    observation_dim = 315
    action_dim = 7
    num_heads = 4
    hidden_dim = 64
    agent = Agent(observation_dim, action_dim, num_heads, hidden_dim).to(device)

    # 에피소드 시작
    for episode in range(10): # 10 에피소드 실행, 원하는 에피소드 수로 변경
        env.reset()
        dec, term = env.get_steps(behavior_name)
        done = False

        while not done:
            # 관찰 가져오기
            observation = dec.obs[0]

            # 행동 선택
            action = agent.act(observation)

            action = action.cpu().numpy()
            action_tuple = ActionTuple()
            action_tuple.add_discrete(action)

            # 행동 수행
            env.set_actions(behavior_name, action_tuple)

            # 다음 스텝
            env.step()
            dec, term = env.get_steps(behavior_name)

            # 종료 조건 확인, 세 명의 에이전트가 종료되면 done이 True가 됨
            if len(term) > 0:
                done = term.interrupted[0]
            

    env.close()
