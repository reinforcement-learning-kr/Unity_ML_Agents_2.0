# 라이브러리 불러오기
import numpy as np
import random
import datetime
import platform
import tensorflow as tf
from collections import deque
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel

# DQN을 위한 파라미터 값 세팅 
state_size = [64, 84, 3]
action_size = 4 

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.99
learning_rate = 3e-4

run_step = 10000 if train_mode else 0
test_step = 10000
train_start_step = 5000
target_update_step = 500

print_interval = 10
save_interval = 100

epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.1
explore_step = run_step * 0.8
eplsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0.

# 그리드월드 환경 설정 (게임판 크기=5, 목적지 수=1, 장애물 수=1)
env_config = {"gridSize": 5, "numGoals": 1, "numObstacles": 1}
VISUAL_OBS = 0
VECTOR_OBS = 1
OBS = VISUAL_OBS

# 유니티 환경 경로 
game = "GridWorld"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/DQN/{date_time}"
load_path = f"./saved_models/{game}/DQN/20201113003549"

# DQN 클래스 -> Deep Q Network 정의 
class DQN(tf.keras.Model):
    def __init__(self, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(32, 8, activation='relu',
                                            strides=[4,4], padding="same")
        self.conv2 = tf.keras.layers.Conv2D(64, 4, activation='relu',
                                            strides=[4,4], padding="same")
        self.conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu',
                                            strides=[1,1], padding="same")
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(512, activation='relu')
        self.q = tf.keras.layers.Dense(action_size, activation='linear')

    def call(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.q(x)

# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 정의 
class DQNAgent:
    def __init__(self):
        self.network = DQN(name='Q')
        self.target_network = DQN(name='target')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init

        if load_model == True:
            self.network.load_weights(load_path+'/model')
        self.update_target()
        self.writer = tf.summary.create_file_writer(save_path)

    # Epsilon greedy 기법에 따라 행동 결정
    def get_action(self, state, training=True):
        # 랜덤하게 행동 결정
        if self.epsilon > random.random():  
            action = np.random.randint(1, action_size, size=(1,1))
        # 네트워크 연산에 따라 행동 결정
        else:
            q = self.network(state, training=training)                     
            action = tf.argmax(q, axis=-1)[..., tf.newaxis].numpy()
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
        done       = np.stack([b[4] for b in batch], axis=0).astype(np.float32)

        # 엡실론 감소
        self.epsilon = max(epsilon_min, self.epsilon - eplsilon_delta)
        
        return self.train_step(state, action, reward, next_state, done)

    @tf.function
    def train_step(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape:
            action_one_hot = tf.one_hot(tf.squeeze(action), action_size)
            q = tf.reduce_sum(self.network(state) * action_one_hot, axis=1, keepdims=True)
            next_q = tf.reduce_max(self.target_network(next_state), axis=1, keepdims=True)
            target_q = reward + (1 - done)*discount_factor*next_q
            delta = abs(target_q - q)
            huber = delta + tf.cast(delta > 1, tf.float32)*(delta**2 - delta)
            loss = tf.reduce_mean(huber)
        grads = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
        return loss

    # 타겟 네트워크 업데이트
    def update_target(self):
        self.target_network.set_weights(self.network.get_weights())

    # 엡실론 설정
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    # 네트워크 모델 저장 
    def save_model(self):
        print("... Save Model ...")
        self.network.save_weights(save_path+'/model')

    # 학습 기록 
    def write_summray(self, score, loss, epsilon, step):
        with self.writer.as_default():
            tf.summary.scalar("run/score", score, step=step)
            tf.summary.scalar("model/loss", loss, step=step)
            tf.summary.scalar("model/epsilon", epsilon, step=step)

# Main 함수 -> 전체적으로 DQN 알고리즘을 진행 
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                        #    no_graphics=True,
                           side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 브레인 설정 
    group_name = list(env.behavior_specs.keys())[0]
    group_spec = env.behavior_specs[group_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    dec, term = env.get_steps(group_name)

    # DQNAgent 클래스를 agent로 정의 
    agent = DQNAgent()
    
    losses, scores, episode, score = [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
                agent.set_epsilon(epsilon_eval)
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        state = dec.obs[OBS]
        action = agent.get_action(state, train_mode)
        real_action = action + 1
        env.set_actions(group_name, real_action)
        env.step()

        dec, term = env.get_steps(group_name)
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        next_state = term.obs[OBS] if done else dec.obs[OBS]
        score += reward[0]

        if train_mode and len(state) > 0:
            agent.append_sample(state[0], action[0], reward, next_state[0], [done])

        if train_mode and step > max(batch_size, train_start_step) :
            # 학습 수행
            loss = agent.train_model()
            losses.append(loss)

            # 타겟 네트워크 업데이트 
            if step % target_update_step == 0:
                agent.update_target()

        if done:
            episode +=1
            scores.append(score)
            score = 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
            if episode % print_interval == 0:
                mean_score = tf.reduce_mean(scores)
                mean_loss = tf.reduce_mean(losses)
                agent.write_summray(mean_score, mean_loss, agent.epsilon, step)
                losses, scores = [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                      f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

            # 네트워크 모델 저장 
            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()