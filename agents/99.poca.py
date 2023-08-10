import numpy as np
import torch
import torch.nn.functional as F

from mlagents_envs.environment import UnityEnvironment

state_size = 6
action_size = 7

discount_factor = 0.99
learning_rate = 3e-4
n_step = 128
batch_size = 128
n_epoch = 3
_lambda = 0.95
epsilon = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(torch.nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_size),
            torch.nn.Softmax(dim=-1)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(28800, 128), # 중앙화된 가치 추정
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, state, global_state):
        policy = self.actor(state)
        value = self.critic(global_state)
        return policy, value

class MAPocaAgent:
    def __init__(self):
        self.networks = [ActorCritic().to(device) for _ in range(num_agents)]
        self.optimizers = [torch.optim.Adam(network.parameters(), lr=learning_rate) for network in self.networks]
        self.memory = [[] for _ in range(num_agents)]

    def get_actions(self, states):
        actions = []
        for network, state in zip(self.networks, states):
            for agent_state in state: # 각 에이전트의 상태를 처리
                state_tensor = torch.FloatTensor(agent_state).to(device) # agent_state의 차원이 20인지 확인
                global_state_tensor = torch.FloatTensor(states).view(-1).to(device)
                policy, _ = network(state_tensor, global_state_tensor)
                action = torch.multinomial(policy, num_samples=1).cpu().numpy()
                actions.append(action)
        return actions


    # 리플레이 메모리에 데이터 추가
    def append_sample(self, states, actions, rewards, next_states, dones):
        for i in range(num_agents):
            self.memory[i].append((states[i], actions[i], rewards[i], next_states[i], dones[i]))

    # 학습 수행
    def train_model(self):
        # 각 에이전트별로 학습 수행
        for i in range(num_agents):
            states, actions, rewards, next_states, dones = zip(*self.memory[i])
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            global_states = states.view(-1)
            global_next_states = next_states.view(-1)

            # 현재 정책 및 가치 계산
            policies, values = self.networks[i](states, global_states)
            next_policies, next_values = self.networks[i](next_states, global_next_states)

            # Advantage 계산
            advantages = rewards + (1 - dones) * discount_factor * next_values - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            # PPO 손실 계산
            old_policies = policies.detach()
            ratio = (policies / old_policies).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(values, rewards + (1 - dones) * discount_factor * next_values)

            # 업데이트
            loss = actor_loss + critic_loss
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

            self.memory[i] = [] # 메모리 초기화

if __name__ == '__main__':
    # Unity 환경 경로 설정
    env_name = None
    env = UnityEnvironment(file_name=env_name)
    env.reset()

    # 유니티 브레인 설정
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    dec, term = env.get_steps(behavior_name)
    num_agents = len(dec)

    # MA-POCA 에이전트 초기화
    agent = MAPocaAgent()

    # 학습 설정
    run_step = 2000000
    print_interval = 10

    # 학습 루프
    for step in range(run_step):
        env.reset()
        dec, term = env.get_steps(behavior_name)
        done = [False] * num_agents
        episode_rewards = np.zeros(num_agents)

        while not any(done):
            states = dec.obs[0]
            actions = agent.get_actions(states)
            env.set_actions(behavior_name, actions)
            env.step()

            # 환경으로부터 얻는 정보
            dec, term = env.get_steps(behavior_name)
            next_states = dec.obs[0]
            rewards = dec.reward
            for id in term.agent_id:
                _id = list(term.agent_id).index(id)
                done[id] = True
                next_states[id] = term.obs[0][_id]
                rewards[id] = term.reward[_id]

            # 메모리에 샘플 추가
            agent.append_sample(states, actions, rewards, next_states, done)

            episode_rewards += rewards

        # 에이전트 학습
        agent.train_model()

        # 게임 진행 상황 출력
        if step % print_interval == 0:
            print(f"Step: {step} / Episode Rewards: {episode_rewards}")

    env.close()
