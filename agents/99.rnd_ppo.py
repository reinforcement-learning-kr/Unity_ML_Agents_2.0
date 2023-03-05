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
state_size = [3*2, 32, 32]
action_size = 8

load_model = False
train_mode = True

discount_factor = 0.9999
learning_rate = 3e-4
n_step = 4096
batch_size = 512
n_epoch = 3
_lambda = 0.95
epsilon = 0.2

rnd_learning_rate = 1e-4
rnd_strength = 0.1
rnd_discount_factor = 0.99
rnd_feature_size = 128

run_step = 5000000 if train_mode else 0
test_step = 10000

print_interval = 5
save_interval = 50

# 유니티 환경 경로 
game = "Maze"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/RNDPPO/{date_time}"
load_path = f"./saved_models/{game}/RNDPPO/20220502131128"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPONetwork 클래스 -> Actor Network, Critic Network 정의 
class PPONetwork(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PPONetwork, self).__init__(**kwargs)
        self.conv1 = torch.nn.Conv2d(in_channels=state_size[0], out_channels=16,
                                     kernel_size=8, stride=4)
        dim1 = ((state_size[1] - 8)//4 + 1, (state_size[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32,
                                     kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.flat = torch.nn.Flatten()
        self.d1 = torch.nn.Linear(32*dim2[0]*dim2[1], 512)
        self.d2 = torch.nn.Linear(512, 512)
        self.pi = torch.nn.Linear(512, action_size)
        self.v = torch.nn.Linear(512, 1)
        self.v_i = torch.nn.Linear(512, 1)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.flat(x)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return F.softmax(self.pi(x), dim=-1), self.v(x)
    
    def get_vi(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.flat(x)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return self.v_i(x)
    
# RNDNetwork 클래스 
class RNDNetwork(torch.nn.Module):
    def __init__(self, is_predictor, **kwargs):
        super(RNDNetwork, self).__init__(**kwargs)
        self.is_predictor = is_predictor
        self.conv1 = torch.nn.Conv2d(in_channels=state_size[0], out_channels=16,
                                     kernel_size=8, stride=4)
        dim1 = ((state_size[1] - 8)//4 + 1, (state_size[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32,
                                     kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.flat = torch.nn.Flatten()
        if is_predictor:
            self.d1 = torch.nn.Linear(32*dim2[0]*dim2[1], 128)
            self.d2 = torch.nn.Linear(128, 128)
            self.v = torch.nn.Linear(128, rnd_feature_size)
        else:
            self.v = torch.nn.Linear(32*dim2[0]*dim2[1], rnd_feature_size)
            
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.flat(x)
        if self.is_predictor:
            x = F.relu(self.d1(x))
            x = F.relu(self.d2(x))
        return self.v(x)
    
# codes modified from https://github.com/openai/random-network-distillation
class RunningMeanStd(torch.nn.Module):
    def __init__(self, shape, epsilon=1e-4):
        super(RunningMeanStd, self).__init__()
        self.mean = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.var = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.count = torch.nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def update(self, x):
        batch_mean, batch_std, batch_count = x.mean(axis=0), x.std(axis=0), x.shape[0]
        batch_var = torch.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count.data = new_count
    
    def normalize(self, x):
        return torch.clip((x - self.mean) / (torch.sqrt(self.var) + 1e-7), min=-5.0, max=5.0)
        
# RNDPPOAgent 클래스 -> RND PPO 알고리즘을 위한 다양한 함수 정의 
class RNDPPOAgent:
    def __init__(self):
        # RND 모듈에 사용되는 예측, 랜덤 네트워크 선언 
        self.random_network = RNDNetwork(is_predictor=False).to(device)
        self.predictor_network = RNDNetwork(is_predictor=True).to(device)
        self.rnd_optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=rnd_learning_rate)
        
        # 관측, 내적 보상 RunningMeanStd 선언
        raw_state_size = state_size[1:] + state_size[:1]
        self.obs_rms = RunningMeanStd(raw_state_size).to(device)
        self.ri_rms = RunningMeanStd(1).to(device)

        self.network = PPONetwork().to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = list()
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    # 정책을 통해 행동 결정
    def get_action(self, state, training=True):
        # 네트워크 모드 설정
        self.network.train(training)
        
        # 네트워크 연산에 따라 행동 결정
        pi, _ = self.network(torch.FloatTensor(state).to(device))
        action = torch.multinomial(pi, num_samples=1).cpu().numpy()
        return action

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 학습 수행
    def train_model(self):
        self.network.train()
        self.predictor_network.train()
        self.random_network.train(False)

        state      = np.stack([m[0] for m in self.memory], axis=0)
        action     = np.stack([m[1] for m in self.memory], axis=0)
        reward     = np.stack([m[2] for m in self.memory], axis=0)
        next_state = np.stack([m[3] for m in self.memory], axis=0)
        done       = np.stack([m[4] for m in self.memory], axis=0)
        self.memory.clear()

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])
        
        # obs_rms update
        self.obs_rms.update(state)
        
        # prob_old, adv, ret 계산
        with torch.no_grad():
            # obs normalize
            normalized_next_state = self.obs_rms.normalize(next_state)
            target = self.random_network(normalized_next_state)
            pred = self.predictor_network(normalized_next_state)
            reward_i = torch.sum(torch.square(pred - target), dim = 1, keepdim=True)
            
            # ri_rms update
            self.ri_rms.update(reward_i)
            
            # ri normalize
            reward_i /= torch.sqrt(self.ri_rms.var) + 1e-7
            
            pi_old, value = self.network(state)
            prob_old = pi_old.gather(1, action.long())
            value_i = self.network.get_vi(state)
            
            _, next_value = self.network(next_state)
            delta = reward + (1 - done) * discount_factor * next_value - value
            
            next_value_i = self.network.get_vi(next_state)
            delta_i = reward_i + rnd_discount_factor * next_value_i - value_i
            
            adv, adv_i = delta.clone(), delta_i.clone()
            adv, adv_i, done = map(lambda x: x.view(n_step, -1).transpose(0,1).contiguous(), [adv, adv_i, done])
            for t in reversed(range(n_step-1)):
                adv[:, t] += (1 - done[:, t]) * discount_factor * _lambda * adv[:, t+1]
                adv_i[:, t] += rnd_discount_factor * _lambda * adv_i[:, t+1]
           
            # adv standardization
            adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True) + 1e-7)
            adv_i = (adv_i - adv_i.mean(dim=1, keepdim=True)) / (adv_i.std(dim=1, keepdim=True) + 1e-7)
                
            adv, adv_i  = map(lambda x: x.transpose(0,1).contiguous().view(-1, 1), [adv, adv_i])
            
            ret = adv + value
            ret_i = adv_i + value_i
            adv = adv + rnd_strength * adv_i
            
        # 학습 이터레이션 시작
        actor_losses, critic_losses, rnd_losses = [], [], []
        idxs = np.arange(len(reward))
        for _ in range(n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), batch_size):
                idx = idxs[offset : offset + batch_size]

                _state, _next_state, _action, _ret, _ret_i, _adv, _prob_old =\
                    map(lambda x: x[idx], [state, next_state, action, ret, ret_i, adv, prob_old])
                
                pi, value = self.network(_state)
                prob = pi.gather(1, _action.long())

                # 정책신경망 손실함수 계산
                ratio = prob / (_prob_old + 1e-7)
                surr1 = ratio * _adv
                surr2 = torch.clamp(ratio, min=1-epsilon, max=1+epsilon) * _adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # 가치신경망 손실함수 계산
                critic_loss_e = F.mse_loss(value, _ret).mean()
                
                # 내적 가치 신경망 손실함수 계산
                value_i = self.network.get_vi(_state)
                critic_loss_i = F.mse_loss(value_i, _ret_i).mean()
                
                critic_loss = critic_loss_e + critic_loss_i

                total_loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # RND 신경망 손실함수 계산
                _normalized_next_state = self.obs_rms.normalize(_next_state)
                with torch.no_grad():
                    target = self.random_network(_normalized_next_state)
                pred = self.predictor_network(_normalized_next_state)
                rnd_loss = torch.sum(torch.square(pred - target), dim = 1).mean()
                
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                rnd_losses.append(rnd_loss.item())
                
        return np.mean(actor_losses), np.mean(critic_losses), np.mean(rnd_losses)

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path+'/ckpt')

    # 학습 기록 
    def write_summary(self, score, actor_loss, critic_loss, rnd_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)
        self.writer.add_scalar("model/rnd_loss", rnd_loss, step)

# Main 함수 -> 전체적으로 RND PPO 알고리즘을 진행 
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

    # PPO 클래스를 agent로 정의 
    agent = RNDPPOAgent()
    actor_losses, critic_losses, rnd_losses, scores, episode, score = [], [], [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
        
        state = dec.obs[0]
        action = agent.get_action(state, train_mode)

        action_branches = np.array(
          [[0,0,1], [0,0,2], [0,1,0],
           [0,1,1], [0,1,2], [1,0,1],
           [1,0,2], [1,1,0]])
        branch_action = action_branches[action.squeeze()]
       
        action_tuple = ActionTuple()
        action_tuple.add_discrete(branch_action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        # 환경으로부터 얻는 정보
        dec, term = env.get_steps(behavior_name)
        done = [False] * num_worker
        next_state = dec.obs[0]
        reward = dec.reward
        for id in term.agent_id:
            _id = list(term.agent_id).index(id)
            done[id] = True
            next_state[id] = term.obs[0][_id]
            reward[id] = term.reward[_id]
        score += sum(reward)/len(reward)

        if train_mode:
            for id in range(num_worker):
                agent.append_sample(state[id], action[id], [reward[id]], next_state[id], [done[id]])

                # 학습수행  
                if len(agent.memory) / num_worker == n_step:
                    actor_loss, critic_loss, rnd_loss = agent.train_model()
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    rnd_losses.append(rnd_loss)

        if done[0]:
            episode +=1
            scores.append(score)
            score = 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_rnd_loss = np.mean(rnd_losses) if len(rnd_losses) > 0 else 0
                mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
                mean_critic_loss = np.mean(critic_losses)  if len(critic_losses) > 0 else 0
                agent.write_summary(mean_score, mean_actor_loss, mean_critic_loss, mean_rnd_loss, step)
                actor_losses, critic_losses, rnd_losses, scores = [], [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                    f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f} / RND loss: {mean_rnd_loss:.4f}" )

            # 네트워크 모델 저장 
            if train_mode and episode % save_interval == 0:
                agent.save_model()
    env.close()
