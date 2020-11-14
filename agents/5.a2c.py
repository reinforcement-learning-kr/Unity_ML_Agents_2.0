# 라이브러리 불러오기
import numpy as np
import datetime
import tensorflow as tf
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel

# DQN을 위한 파라미터 값 세팅 
state_size = 6
action_size = 4 

load_model = False
train_mode = True

discount_factor = 0.99
learning_rate = 5e-4

run_step = 50000 if train_mode else 0
test_step = 10000

print_interval = 10
save_interval = 100

# 소코반 환경 설정 (게임판 크기=5, 초록색 +의 수=1, 박스의 수=1)
env_config = {"gridSize": 5, "numGoals": 1, "numBoxes": 1}
VISUAL_OBS = 0
VECTOR_OBS = 1
OBS = VECTOR_OBS

# 유니티 환경 경로 
game = "GridWorld"
env_name = f"../envs/{game}/{game}"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/A2C/{date_time}"
load_path = f"./saved_models/{game}/A2C/20201114230509"

# A2C 클래스 -> Actor, Critic 정의 
class A2C(tf.keras.Model):
    def __init__(self, **kwargs):
        super(A2C, self).__init__(**kwargs)
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')
        self.pi = tf.keras.layers.Dense(action_size, activation='softmax')
        self.v = tf.keras.layers.Dense(action_size, activation='linear')

    def call(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        x = self.d1(x)
        x = self.d2(x)
        return self.pi(x), self.v(x)

# A2CAgent 클래스 -> A2C 알고리즘을 위한 다양한 함수 정의 
class A2CAgent:
    def __init__(self):
        self.a2c = A2C()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        if load_model == True:
            self.a2c.load_weights(load_path+'/a2c')
        self.writer = tf.summary.create_file_writer(save_path)

    def get_action(self, state, training=True):
        # 네트워크 연산에 따라 행동 결정
        pi, _ = self.a2c(state, training=training)
        action = np.random.choice(action_size, p=pi.numpy()[0], size=(1,1))
        return action

    # 학습 수행
    def train_model(self, state, action, reward, next_state, done):
        action     = tf.convert_to_tensor(action)
        reward     = tf.convert_to_tensor(reward)
        done       = tf.convert_to_tensor(done, tf.float32)
        return self.train_step(state, action, reward, next_state, done)

    @tf.function
    def train_step(self, state, action, reward, next_state, done):
        with tf.GradientTape(persistent=True) as tape:
            pi, value = self.a2c(state)
            _, next_value = self.a2c(next_state)
            target = tf.stop_gradient(reward + (1 - done) * discount_factor * next_value)
            critic_loss = tf.keras.losses.mean_squared_error(target, value)

            action_one_hot = tf.one_hot(tf.squeeze(action), action_size)
            advantage = tf.stop_gradient(target - value)
            actor_loss = tf.reduce_mean(-tf.math.log(tf.reduce_sum(action_one_hot*pi, axis=1))*advantage)

            total_loss = critic_loss + actor_loss
        
        grads = tape.gradient(total_loss, self.a2c.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.a2c.trainable_weights))
        return actor_loss, critic_loss

    # 네트워크 모델 저장 
    def save_model(self):
        print("... Save Model ...")
        self.a2c.save_weights(save_path+'/a2c')

    # 학습 기록 
    def write_summray(self, score, actor_loss, critic_loss, step):
        with self.writer.as_default():
            tf.summary.scalar("run/score", score, step=step)
            tf.summary.scalar("model/actor_loss", actor_loss, step=step)
            tf.summary.scalar("model/critic_loss", critic_loss, step=step)

# Main 함수 -> 전체적으로 A2C 알고리즘을 진행 
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

    # A2CAgent 클래스를 agent로 정의 
    agent = A2CAgent()
    
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        if  step == run_step:
            if train_mode:
                agent.save_model()
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
        reward = term.reward[0] if done else dec.reward[0]
        next_state = term.obs[OBS] if done else dec.obs[OBS]
        score += reward

        if train_mode and len(state) > 0:
            # 학습 수행 
            actor_loss, critic_loss = agent.train_model(state, action[0], [reward], next_state, [done])
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        if done:
            episode +=1
            scores.append(score)
            score = 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
            if episode % print_interval == 0:
                mean_score = tf.reduce_mean(scores)
                mean_actor_loss = tf.reduce_mean(actor_losses)
                mean_critic_loss = tf.reduce_mean(critic_losses)
                agent.write_summray(mean_score, mean_actor_loss, mean_critic_loss, step)
                actor_losses, critic_losses, scores = [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                      f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")

            # 네트워크 모델 저장 
            if train_mode and episode % save_interval == 0:
                agent.save_model()
    env.close()