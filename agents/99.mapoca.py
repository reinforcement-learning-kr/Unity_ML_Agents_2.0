# 라이브러리 불러오기
import numpy as np
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel
# 파라미터 값 세팅 
actor_state_size = 651
action_size = 5
num_agents = 3
critic_state_size = actor_state_size * num_agents

load_model = False
train_mode = True

discount_factor = 0.99
learning_rate = 3e-4
n_step = 128
batch_size = 128
n_epoch = 3
_lambda = 0.95
epsilon = 0.2

run_step = 2000000 if train_mode else 0
test_step = 100000

print_interval = 10
save_interval = 100

# 유니티 환경 경로 
game = "EscapeRoom"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/MAPOCA/{date_time}"
load_path = f"./saved_models/{game}/MAPOCA/20230728125435"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor 클래스 -> Actor Network 정의 
class Actor(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.d1 = torch.nn.Linear(actor_state_size, 128)
        self.d2 = torch.nn.Linear(128, 128)
        self.pi = torch.nn.Linear(128, action_size)
        
    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return F.softmax(self.pi(x), dim=-1)

# Critic 클래스 -> Critic Network 정의 
class Critic(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)
        # Residual Self Attention
        self.query = torch.nn.Linear(critic_state_size, 128)
        self.key = torch.nn.Linear(critic_state_size, 128)
        self.value = torch.nn.Linear(critic_state_size, 128)
        self.MHA = torch.nn.MultiheadAttention(128, 4)
        self.out = torch.nn.Linear(128, critic_state_size)

        # Linear Layer
        self.d1 = torch.nn.Linear(critic_state_size, 128)
        self.d2 = torch.nn.Linear(128, 128)
        self.v = torch.nn.Linear(128, 1)
        
    def forward(self, x):
        q = self.query(x).unsqueeze(dim=0)
        k = self.key(x).unsqueeze(dim=0)
        v = self.value(x).unsqueeze(dim=0)
        attn_output, attn_output_weights = self.MHA(q, k, v)
        x = self.out(attn_output.squeeze(dim=0)) + x
        
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return self.v(x)

# MAPOCAAgent 클래스 -> MAPOCA 알고리즘을 위한 다양한 함수 정의 
class MAPOCAAgent:
    def __init__(self):
        self.actors = [Actor().to(device) for _ in range(num_agents)]
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=learning_rate) for actor in self.actors]
        self.critic = Critic().to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.memory = list()
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            for i in range(num_agents):
                self.actors[i].load_state_dict(checkpoint[f"actor_{i}"])
                self.actor_optimizers[i].load_state_dict(checkpoint[f"actor_optimizer_{i}"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    # 정책을 통해 행동 결정 
    def get_action(self, states, active_agents, training=True):
        actions = []
        for i, (state, active_agent) in enumerate(zip(states, active_agents)):
            # 네트워크 모드 설정
            self.actors[active_agent].train(training)

            # 네트워크 연산에 따라 행동 결정
            pi = self.actors[active_agent](torch.FloatTensor(state).to(device))
            action = torch.multinomial(pi, num_samples=1).cpu().numpy()
            actions.append(action)
        return np.array(actions).reshape((len(active_agents), 1))

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 학습 수행
    # def train_model(self):
    #     for i in range(num_agents):
    #         self.actors[i].train()
    #     self.critic.train()

    #     state      = np.stack([m[0] for m in self.memory], axis=0)
    #     action     = np.stack([m[1] for m in self.memory], axis=0)
    #     reward     = np.stack([m[2] for m in self.memory], axis=0)
    #     next_state = np.stack([m[3] for m in self.memory], axis=0)
    #     done       = np.stack([m[4] for m in self.memory], axis=0)
    #     self.memory.clear()

    #     state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
    #                                                     [state, action, reward, next_state, done])
    #     # prob_old, adv, ret 계산 
    #     with torch.no_grad():
    #         pi_old, value = self.network(state)
    #         prob_old = pi_old.gather(1, action.long())

    #         _, next_value = self.network(next_state)
    #         delta = reward + (1 - done) * discount_factor * next_value - value
    #         adv = delta.clone()
    #         adv, done = map(lambda x: x.view(n_step, -1).transpose(0,1).contiguous(), [adv, done])
    #         for t in reversed(range(n_step-1)):
    #             adv[:, t] += (1 - done[:, t]) * discount_factor * _lambda * adv[:, t+1]
    #         adv = adv.transpose(0,1).contiguous().view(-1, 1)
            
    #         ret = adv + value

    #     # 학습 이터레이션 시작
    #     actor_losses, critic_losses = [], []
    #     idxs = np.arange(len(reward))
    #     for _ in range(n_epoch):
    #         np.random.shuffle(idxs)
    #         for offset in range(0, len(reward), batch_size):
    #             idx = idxs[offset : offset + batch_size]

    #             _state, _action, _ret, _adv, _prob_old =\
    #                 map(lambda x: x[idx], [state, action, ret, adv, prob_old])
                
    #             pi, value = self.network(_state)
    #             prob = pi.gather(1, _action.long())

    #             # 정책신경망 손실함수 계산
    #             ratio = prob / (_prob_old + 1e-7)
    #             surr1 = ratio * _adv
    #             surr2 = torch.clamp(ratio, min=1-epsilon, max=1+epsilon) * _adv
    #             actor_loss = -torch.min(surr1, surr2).mean()

    #             # 가치신경망 손실함수 계산
    #             critic_loss = F.mse_loss(value, _ret).mean()

    #             total_loss = actor_loss + critic_loss

    #             self.optimizer.zero_grad()
    #             total_loss.backward()
    #             self.optimizer.step()

    #             actor_losses.append(actor_loss.item())
    #             critic_losses.append(critic_loss.item())

    #     return np.mean(actor_losses), np.mean(critic_losses)

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        obj = {}
        for i in range(num_agents):
            obj[f"actor_{i}"] = self.actors[i].state_dict()
            obj[f"actor_optimizer_{i}"] = self.actor_optimizers[i].state_dict()
        obj["critic"] = self.critic.state_dict()
        obj["critic_optimizer"] = self.critic_optimizer.state_dict()
        torch.save(obj, save_path+'/ckpt')

    # 학습 기록 
    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)

# Main 함수 -> 전체적으로 PPO 알고리즘을 진행 
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel],
                           base_port=7777)
    env.reset()

    # 유니티 behavior 설정 
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    
    dec, term = env.get_steps(behavior_name)
    # MAPOCA 클래스를 agent로 정의 
    agent = MAPOCAAgent()
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    agents_id, active_agents, term_agents = dec.agent_id, list(range(num_agents)), 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
        
        states = dec.obs[0]
        actions = agent.get_action(states, active_agents, train_mode)
        actions_tuple = ActionTuple()
        actions_tuple.add_discrete(actions)
        env.set_actions(behavior_name, actions_tuple)
        env.step()
        # 환경으로부터 얻는 정보
        dec, term = env.get_steps(behavior_name)
        next_states = dec.obs[0]
        for term_agent_id in term.agent_id:
            active_agents.remove(list(agents_id).index(term_agent_id))
            term_agents += 1
        done = term_agents == num_agents
        reward = dec.reward
        score += sum(reward)

        # if train_mode:
        #     for id in range(num_worker):
        #         agent.append_sample(state[id], action[id], [reward[id]], next_state[id], [done[id]])
        #     # 학습수행
        #     if (step+1) % n_step == 0:
        #         actor_loss, critic_loss = agent.train_model()
        #         actor_losses.append(actor_loss)
        #         critic_losses.append(critic_loss)

        if done:
            episode +=1
            scores.append(score)
            agents_id, active_agents, term_agents, score = dec.agent_id, list(range(num_agents)), 0, 0

        #     # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
        #     if episode % print_interval == 0:
        #         mean_score = np.mean(scores)
        #         mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
        #         mean_critic_loss = np.mean(critic_losses)  if len(critic_losses) > 0 else 0
        #         agent.write_summary(mean_score, mean_actor_loss, mean_critic_loss, step)
        #         actor_losses, critic_losses, scores = [], [], []

        #         print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
        #               f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}" )

        #     # 네트워크 모델 저장 
        #     if train_mode and episode % save_interval == 0:
        #         agent.save_model()
    env.close()
