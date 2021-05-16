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
#파라미터 값 세팅 
state_size = 6
action_size = 4 

load_model = False
train_mode = True

discount_factor = 0.9
learning_rate = 0.00025

run_step = 30000 if train_mode else 0
test_step = 3000

print_interval = 10
save_interval = 100

# 그리드월드 환경 설정 (게임판 크기=5, + 목적지 수=1, X 목적지 수=1)
env_config = {"gridSize": 5, "numPlusGoals": 1, "numExGoals": 1}
VISUAL_OBS = 0
VECTOR_OBS = 1
OBS = VECTOR_OBS

# 유니티 환경 경로 
game = "GridWorld"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/A2C/{date_time}"
load_path = f"./saved_models/{game}/A2C/20210217000848"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A2C 클래스 -> Actor Network, Critic Network 정의 
class A2C(torch.nn.Module):
    def __init__(self, **kwargs):
        super(A2C, self).__init__(**kwargs)
        self.d1 = torch.nn.Linear(state_size, 128)
        self.d2 = torch.nn.Linear(128, 128)
        self.pi = torch.nn.Linear(128, action_size)
        self.v = torch.nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return F.softmax(self.pi(x), dim=1), self.v(x)

# A2CAgent 클래스 -> A2C 알고리즘을 위한 다양한 함수 정의 
class A2CAgent:
    def __init__(self):
        self.a2c = A2C().to(device)
        self.optimizer = torch.optim.Adam(self.a2c.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.a2c.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def get_action(self, state, training=True):
        #  네트워크 모드 설정
        self.a2c.train(training)

        # 네트워크 연산에 따라 행동 결정
        pi, _ = self.a2c(torch.FloatTensor(state).to(device))
        action = np.array([np.random.choice(action_size, p=p, size=(1)) for p in pi.data.cpu().numpy()])
        return action

    def train_model(self, state, action, reward, next_state, done):
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])
        pi, value = self.a2c(state)

        #가치신경망
        with torch.no_grad():
            _, next_value = self.a2c(next_state)
            target_value  = reward + (1-done) * discount_factor * next_value
        critic_loss = F.mse_loss(target_value, value)

        #정책신경망
        eye = torch.eye(action_size).to(device)
        one_hot_action = eye[action.view(-1).long()]
        advantage = (target_value - value).detach()
        actor_loss = -(torch.log((one_hot_action * pi).sum(1))*advantage).mean()
        total_loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt...")
        torch.save({
            "network" : self.a2c.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path+'/ckpt')

        # 학습 기록 
    def write_summray(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)

# Main 함수 -> 전체적으로 A2C 알고리즘을 진행 
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 브레인 설정 
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    dec, term = env.get_steps(behavior_name)

    # A2C 클래스를 agent로 정의 
    agent = A2CAgent()
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        state = dec.obs[OBS]
        action = agent.get_action(state, train_mode)
        real_action = action + 1
        action_tuple = ActionTuple()
        action_tuple.add_discrete(real_action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        #환경으로부터 얻는 정보
        dec, term = env.get_steps(behavior_name)
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        next_state = term.obs[OBS] if done else dec.obs[OBS]
        score += reward[0]

        if train_mode and len(state) > 0:
            #학습수행
            actor_loss, critic_loss = agent.train_model(state, action[0], [reward], next_state, [done])
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        if done:
            episode +=1
            scores.append(score)
            score = 0

          # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
                mean_critic_loss = np.mean(critic_losses)  if len(critic_losses) > 0 else 0
                agent.write_summray(mean_score, mean_actor_loss, mean_critic_loss, step)
                actor_losses, critic_losses, scores = [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                      f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")

            # 네트워크 모델 저장 
            if train_mode and episode % save_interval == 0:
                agent.save_model()
    env.close()