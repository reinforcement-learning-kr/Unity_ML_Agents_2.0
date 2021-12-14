import numpy as np
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.buffer import BufferKey, ObservationKeyPrefix

# Behavioral Cloning을 위한 파라미터 값 세팅
state_size = 12 * 4
action_size = 1

load_model = False
train_mode = True

batch_size = 128
learning_rate = 3e-4
discount_factor = 0.9

train_epoch = 500
test_step = 10000

print_interval = 10
save_interval = 100

# 유니티 환경 경로
game = "Kart"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/BC/{date_time}"
load_path = f"./saved_models/{game}/BC/20211128183733"

# Demonstration 경로
demo_path = "../demo/KartAgent.demo"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor 클래스 -> Behavioral Cloning Actor 클래스 정의
class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.mu = torch.nn.Linear(128, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.mu(x))

# BCAgent 클래스 -> Behavioral Cloning 알고리즘을 위한 다양한 함수 정의
class BCAgent():
    def __init__(self):
        self.actor = Actor().to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(save_path)
        
        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    # 행동 결정
    def get_action(self, state, training=False):
        #  네트워크 모드 설정
        self.actor.train(training)
        action = self.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
        return action
    
    def train_model(self, state, action):
        self.actor.train(True)
        losses = []
        
        rand_idx = torch.randperm(len(state))
        for iter in range(int(np.ceil(len(state)/batch_size))):
            _state = state[rand_idx[iter*batch_size: (iter+1)*batch_size]]
            _action = action[rand_idx[iter*batch_size: (iter+1)*batch_size]]
            
            action_pred = self.actor(_state)
            loss = F.mse_loss(_action, action_pred).mean()
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        
        return np.mean(losses)
        
    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "actor" : self.actor.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path+'/ckpt')

    # 학습 기록
    def write_summray(self, loss, epoch):
        self.writer.add_scalar("model/loss", loss, epoch)
        
# Main 함수 -> 전체적으로 BC 알고리즘을 진행
if __name__ == '__main__':
    # BCAgent 클래스를 agent로 정의
    agent = BCAgent()
    
    if train_mode:
        # Demonstration 정보 가져오기
        brain_params, demo_buffer = demo_to_buffer(demo_path,1)
        print("Brain Params", brain_params)
        print(demo_buffer._fields.keys())
        
        demo_to_tensor = lambda key: torch.FloatTensor(demo_buffer[key]).to(device)
        state = demo_to_tensor((ObservationKeyPrefix.OBSERVATION, 0))
        action = demo_to_tensor(BufferKey.CONTINUOUS_ACTION)
        reward = demo_to_tensor(BufferKey.ENVIRONMENT_REWARDS)
        done = demo_to_tensor(BufferKey.DONE)
        
        ret = reward.clone()
        for t in reversed(range(len(ret) - 1)):
            ret[t] += (1. - done[t]) * (discount_factor * ret[t+1])
        
        # return이 0보다 큰 (state, action) pair만 학습에 사용.
        state, action = map(lambda x: x[ret > 0], [state, action])
        
        losses = []
        for epoch in range(1, train_epoch+1):
            loss = agent.train_model(state, action)
            agent.write_summray(loss, epoch)
            losses.append(loss)
            
            # 텐서 보드에 손실함수 값 기록
            if epoch % print_interval == 0:
                mean_loss = np.mean(losses)
                print(f"{epoch} Epoch / Loss: {mean_loss:.8f}" )
                agent.write_summray(mean_loss, epoch)
                losses = []
            
            if epoch % save_interval == 0:
                agent.save_model()
                
    # 빌드 환경에서 Play 시작
    print("PLAY START")

    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 브레인 설정
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
    dec, term = env.get_steps(behavior_name)
    
    # TEST 시작
    scores, episode, score = [], 0, 0
    for step in range(test_step):
        state = dec.obs[0]
        action = agent.get_action(state, False)
        action_tuple = ActionTuple()
        action_tuple.add_continuous(action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name)
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        next_state = term.obs[0] if done else dec.obs[0]
        score += reward[0]

        if done:
            episode += 1
            scores.append(score)
            score = 0

            # 게임 진행 상황 출력
            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} ")

    env.close()
