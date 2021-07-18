# 라이브러리 불러오기
import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel

# DQN을 위한 파라미터 값 세팅 
state_size = 8
action_size = 3

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.9
learning_rate = 0.00025

run_step = 1000000 if train_mode else 0
test_step = 100000
train_start_step = 100000
target_update_step = 10000

print_interval = 10
save_interval = 100

epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.1
explore_step = run_step * 0.8
eplsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0.

# 유니티 환경 경로 
game = "Pong"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/ADQN/{date_time}"
load_path = f"./saved_models/{game}/ADQN/20210710003410"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN 클래스 -> Deep Q Network 정의 
class DQN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.q = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 정의 
class DQNAgent:
    def __init__(self, id):
        self.network = DQN().to(device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init
        self.save_path = f"{save_path}/{id}"
        self.load_path = f"{load_path}/{id}"
        self.writer = SummaryWriter(self.save_path)

        if load_model == True:
            print(f"... Load Model from {self.load_path}/ckpt")
            checkpoint = torch.load(self.load_path+'/ckpt', map_location=device)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
    # Epsilon greedy 기법에 따라 행동 결정 
    def get_action(self, state, training=True):
        #  네트워크 모드 설정
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_eval

        # 랜덤하게 행동 결정
        if epsilon > random.random():  
            action = np.random.randint(0, action_size, size=(state.shape[0],1))
        # 네트워크 연산에 따라 행동 결정
        else:
            q = self.network(torch.FloatTensor(state).to(device))
            action = torch.argmax(q, axis=-1, keepdim=True).data.cpu().numpy()
        return action

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 학습 수행
    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state      = np.stack([b[0] for b in batch], axis=0)
        action     = np.stack([b[1] for b in batch], axis=0)
        reward     = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done       = np.stack([b[4] for b in batch], axis=0)

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])

        eye = torch.eye(action_size).to(device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            next_q = self.target_network(next_state)
            target_q = reward + next_q.max(1, keepdims=True).values * ((1 - done) * discount_factor)

        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 엡실론 감소
        self.epsilon = max(epsilon_min, self.epsilon - eplsilon_delta)

        return loss.item()

    # 타겟 네트워크 업데이트
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    # 네트워크 모델 저장 
    def save_model(self):
        print(f"... Save Model to {self.save_path}/ckpt...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, self.save_path+'/ckpt')

    # 학습 기록 
    def write_summray(self, score, loss, epsilon, step):
            self.writer.add_scalar("run/score", score, step)
            self.writer.add_scalar("model/loss", loss, step)
            self.writer.add_scalar("model/epsilon", epsilon, step)

# Main 함수 -> 전체적으로 DQN 알고리즘을 진행 
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 브레인 설정 
    behavior_name_list = list(env.behavior_specs.keys())
    behavior_A = behavior_name_list[0]
    behavior_B = behavior_name_list[1]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    dec_A, term_A = env.get_steps(behavior_A)
    dec_B, term_B = env.get_steps(behavior_B)

    # DQNAgent 클래스를 agent로 정의 
    agent_A = DQNAgent("A")
    agent_B = DQNAgent("B")
    
    losses_A, losses_B, scores_A, scores_B, episode, score_A, score_B = [], [], [], [], 0, 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent_A.save_model()
                agent_B.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        state_A = dec_A.obs[0]
        state_B = dec_B.obs[0]
        action_A = agent_A.get_action(state_A, train_mode)
        action_B = agent_B.get_action(state_B, train_mode)
        action_tuple_A, action_tuple_B = map(lambda x: ActionTuple(discrete=x), [action_A, action_B])
        env.set_actions(behavior_A, action_tuple_A)
        env.set_actions(behavior_B, action_tuple_B)
        env.step()

        dec_A, term_A = env.get_steps(behavior_A)
        dec_B, term_B = env.get_steps(behavior_B)
        done_A = len(term_A.agent_id) > 0
        done_B = len(term_B.agent_id) > 0
        next_state_A = term_A.obs[0] if done_A else dec_A.obs[0]
        next_state_B = term_B.obs[0] if done_B else dec_B.obs[0]
        reward_A = term_A.reward if done_A else dec_A.reward
        reward_B = term_B.reward if done_B else dec_B.reward
        score_A += reward_A[0]
        score_B += reward_B[0]

        if train_mode:
            agent_A.append_sample(state_A[0], action_A[0], reward_A, next_state_A[0], [done_A])
            agent_B.append_sample(state_B[0], action_B[0], reward_B, next_state_B[0], [done_B])
    
        if train_mode and step > max(batch_size, train_start_step) :
            # 학습 수행
            loss_A = agent_A.train_model()
            loss_B = agent_B.train_model()
            losses_A.append(loss_A)
            losses_B.append(loss_B)

            # 타겟 네트워크 업데이트 
            if step % target_update_step == 0:
                agent_A.update_target()
                agent_B.update_target()

        if done_A or done_B:
            episode +=1
            scores_A.append(score_A)
            scores_B.append(score_B)
            score_A = score_B = 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
            if episode % print_interval == 0:
                mean_score_A = np.mean(scores_A)
                mean_score_B = np.mean(scores_B)
                mean_loss_A = np.mean(losses_A)
                mean_loss_B = np.mean(losses_B)
                agent_A.write_summray(mean_score_A, mean_loss_A, agent_A.epsilon, step)
                agent_B.write_summray(mean_score_B, mean_loss_B, agent_B.epsilon, step)
                losses_A, losses_B, scores_A, scores_B = [], [], [], []

                print(f"{episode} Episode / Step: {step} / "  +\
                      f"A Score: {mean_score_A:.2f} / B Score: {mean_score_B:.2f} / " +\
                      f"A Loss: {mean_loss_A:.4f} / B Loss: {mean_loss_B:.4f} / Epsilon: {agent_A.epsilon:.4f}")

            # 네트워크 모델 저장 
            if train_mode and episode % save_interval == 0:
                agent_A.save_model()
                agent_B.save_model()

    env.close()