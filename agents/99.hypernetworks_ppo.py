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
from mlagents_envs.side_channel.environment_parameters_channel\
                             import EnvironmentParametersChannel
# 파라미터 값 세팅 
state_size = 126 # Ray(19 * 6 = 114) & position(3) & rotation(3) & velocity(3) & ball velocity(3)
action_size = 4 # Rotate(2) & Move(2)
goal_size = 2 # goal_signal

GOAL_OBS = 0
RAY_OBS = 1
VECTOR_OBS = 2

load_model = False
train_mode = True

discount_factor = 0.99
learning_rate = 3e-4
n_step = 512
batch_size = 256
n_epoch = 3
_lambda = 0.95
epsilon = 0.2

run_step = 2500000 if train_mode else 0
test_step = 10000

print_interval = 10
save_interval = 100

# 유니티 환경 경로 
game = "TwoMissions"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/HyperPPO/{date_time}"
load_path = f"./saved_models/{game}/HyperPPO/20220803093412"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HyperNetwork 클래스 -> HyperActorCritic의 마지막 Layer hn 생성 
class HyperNetwork(torch.nn.Module):
    def __init__(self, input_unit_size, action_size, hyper_input_size, **kwargs):
        super(HyperNetwork, self).__init__(**kwargs)
        self.input_unit_size = input_unit_size
        self.action_size = action_size
        self.hyper_input_size = hyper_input_size

        self.d1 = torch.nn.Linear(self.hyper_input_size, 256)
        self.d2 = torch.nn.Linear(256, 256)
        self.pi = torch.nn.Linear(256, self.input_unit_size * self.action_size)
        self.v = torch.nn.Linear(256, self.input_unit_size)

    def forward(self, x, h):
        h = F.relu(self.d1(h))
        h = F.relu(self.d2(h))
        target_weights_pi = F.tanh(self.pi(h))
        target_weights_v = F.tanh(self.v(h))

        x = x.unsqueeze(dim=1)
        target_weights_pi = target_weights_pi.view(-1, self.input_unit_size, self.action_size)
        result_pi = torch.bmm(x, target_weights_pi)
        result_pi = result_pi.squeeze(dim = 1)
        target_weights_v = target_weights_v.view(-1, self.input_unit_size, 1)
        result_v = torch.bmm(x, target_weights_v)

        return F.softmax(result_pi, dim=1), result_v.squeeze(dim=1)
    
# HyperActorCritic 클래스 -> Hypernetwork를 적용한 Actor, Critic Network 정의 
class HyperActorCritic(torch.nn.Module):
    def __init__(self, **kwargs):
        super(HyperActorCritic, self).__init__(**kwargs)
        self.d1 = torch.nn.Linear(state_size, 256)
        self.d2 = torch.nn.Linear(256, 256)
        self.hn = HyperNetwork(256, action_size, goal_size)
        
    def forward(self, x, h):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return self.hn(x, h)

# HyperPPOAgent 클래스 -> HyperPPOAgent 알고리즘을 위한 다양한 함수 정의 
class HyperPPOAgent:
    def __init__(self):
        self.network = HyperActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = list()
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    # 정책을 통해 행동 결정 
    def get_action(self, state, goal_signal, training=True):
        # 네트워크 모드 설정
        self.network.train(training)

        # 네트워크 연산에 따라 행동 결정
        pi, _ = self.network(torch.FloatTensor(state).to(device), torch.FloatTensor(goal_signal).to(device))
        action = torch.multinomial(pi, num_samples=1).cpu().numpy()
        return action

    # 리플레이 메모리에 데이터 추가 (상태, 목표신호, 행동, 보상, 다음 상태, 다음 목표신호, 게임 종료 여부)
    def append_sample(self, state, goal_signal, action, reward, next_state, next_goal_signal, done):
        self.memory.append((state, goal_signal, action, reward, next_state, next_goal_signal, done))

    # 학습 수행
    def train_model(self):
        self.network.train()

        state      = np.stack([m[0] for m in self.memory], axis=0)
        goal_signal= np.stack([m[1] for m in self.memory], axis=0)
        action     = np.stack([m[2] for m in self.memory], axis=0)
        reward     = np.stack([m[3] for m in self.memory], axis=0)
        next_state = np.stack([m[4] for m in self.memory], axis=0)
        next_goal_signal = np.stack([m[5] for m in self.memory], axis=0)
        done       = np.stack([m[6] for m in self.memory], axis=0)
        self.memory.clear()

        state, goal_signal, action, reward, next_state, next_goal_signal, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, goal_signal, action, reward, next_state, next_goal_signal, done])
        # prob_old, adv, ret 계산 
        with torch.no_grad():
            pi_old, value = self.network(state, goal_signal)
            prob_old = pi_old.gather(1, action.long())

            _, next_value = self.network(next_state, next_goal_signal)
            delta = reward + (1 - done) * discount_factor * next_value - value
            adv = delta.clone()
            adv, done = map(lambda x: x.view(n_step, -1).transpose(0,1).contiguous(), [adv, done])
            for t in reversed(range(n_step-1)):
                adv[:, t] += (1 - done[:, t]) * discount_factor * _lambda * adv[:, t+1]
            adv = adv.transpose(0,1).contiguous().view(-1, 1)
            
            ret = adv + value

        # 학습 이터레이션 시작
        actor_losses, critic_losses = [], []
        idxs = np.arange(len(reward))
        for _ in range(n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), batch_size):
                idx = idxs[offset : offset + batch_size]

                _state, _goal_signal, _action, _ret, _adv, _prob_old =\
                    map(lambda x: x[idx], [state, goal_signal, action, ret, adv, prob_old])
                
                pi, value = self.network(_state, _goal_signal)
                prob = pi.gather(1, _action.long())

                # 정책신경망 손실함수 계산
                ratio = prob / (_prob_old + 1e-7)
                surr1 = ratio * _adv
                surr2 = torch.clamp(ratio, min=1-epsilon, max=1+epsilon) * _adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # 가치신경망 손실함수 계산
                critic_loss = F.mse_loss(value, _ret).mean()

                total_loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path+'/ckpt')

    # 학습 기록 
    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)

# Main 함수 -> 전체적으로 HyperPPO 알고리즘을 진행 
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    environment_parameters_channel = EnvironmentParametersChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel,
                                          environment_parameters_channel])
    env.reset()

    # 유니티 behavior 설정 
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    dec, term = env.get_steps(behavior_name)
    num_worker = len(dec)

    # HyperPPOAgent 클래스를 agent로 정의 
    agent = HyperPPOAgent()
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        goal_signal = dec.obs[GOAL_OBS]
        state = np.concatenate([dec.obs[RAY_OBS], dec.obs[VECTOR_OBS]], axis=-1)
        action = agent.get_action(state, goal_signal, train_mode)
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        # 환경으로부터 얻는 정보
        dec, term = env.get_steps(behavior_name)
        done = [False] * num_worker
        next_goal_signal = dec.obs[GOAL_OBS]
        next_state = np.concatenate([dec.obs[RAY_OBS], dec.obs[VECTOR_OBS]], axis=-1)
        reward = dec.reward
        for id in term.agent_id:
            _id = list(term.agent_id).index(id)
            done[id] = True
            next_goal_signal[id] = term.obs[GOAL_OBS][_id]
            next_state[id] = np.concatenate([term.obs[RAY_OBS][_id], term.obs[VECTOR_OBS][_id]], axis=-1)
            reward[id] = term.reward[_id]
        score += reward[0]

        if train_mode:
            for id in range(num_worker):
                agent.append_sample(state[id], goal_signal[id], action[id], [reward[id]], next_state[id], next_goal_signal[id], [done[id]])
            # 학습수행
            if (step+1) % n_step == 0:
                actor_loss, critic_loss = agent.train_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        if done[0]:
            episode +=1
            scores.append(score)
            score = 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
                mean_critic_loss = np.mean(critic_losses)  if len(critic_losses) > 0 else 0
                agent.write_summary(mean_score, mean_actor_loss, mean_critic_loss, step)
                actor_losses, critic_losses, scores = [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                      f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}" )

            # 네트워크 모델 저장 
            if train_mode and episode % save_interval == 0:
                agent.save_model()
    env.close()
