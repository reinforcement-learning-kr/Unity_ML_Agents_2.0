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

# DDPG를 위한 파라미터 값 세팅
state_size = 9
action_size = 3

load_model = False
train_mode = True

batch_size = 128
mem_maxlen = 50000
discount_factor = 0.99
actor_lr = 1e-4
critic_lr = 5e-4
tau = 1e-3

# OU noise 파라미터
mu = 0
theta = 1e-3
sigma = 2e-3

run_step = 50000 if train_mode else 0
test_step = 10000
train_start_step = 5000

print_interval = 10
save_interval = 100

# 유니티 환경 경로
game = "Drone"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/DDPG/{date_time}"
load_path = f"./saved_models/{game}/DDPG/20201008205356"

# OU_noise 클래스 -> ou noise 정의 및 파라미터 결정


class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = tf.ones((1, action_size), dtype=tf.float32) * mu

    def sample(self):
        dx = theta * (mu - self.X) + sigma * tf.random.normal([len(self.X)])
        self.X += dx
        return self.X

# Actor 클래스 -> DDPG Actor 클래스 정의


class Actor(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.mu = tf.keras.layers.Dense(action_size, activation='tanh')

    def call(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.mu(x)

# Critic 클래스 -> DDPG Critic 클래스 정의


class Critic(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.q = tf.keras.layers.Dense(1)

    def call(self, x1, x2):
        x1, x2 = map(lambda x: tf.convert_to_tensor(x, tf.float32), [x1, x2])
        x = tf.concat([self.fc1(x1), x2], axis=-1)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.q(x)

# DDPGAgent 클래스 -> DDPG 알고리즘을 위한 다양한 함수 정의


class DDPGAgent:
    def __init__(self):
        self.actor = Actor(name="actor")
        self.critic = Critic(name="critic")
        self.target_actor = Actor(name="target_actor")
        self.target_critic = Critic(name="target_critic")
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.ou = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)

        if load_model == True:
            self.actor.load_weights(load_path+'/actor')
            self.critic.load_weights(load_path+'/critic')
        self.writer = tf.summary.create_file_writer(save_path)
        self.first_update = True

    def get_action(self, state, training=True):
        action = self.actor(state, training=training)
        # OU noise 기법에 따라 행동 결정
        noise = self.ou.sample()
        return action + noise if training else action

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 학습 수행
    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state = np.stack([b[0] for b in batch], axis=0)
        action = np.stack([b[1] for b in batch], axis=0)
        reward = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done = np.stack([b[4] for b in batch], axis=0).astype(np.float32)
        actor_loss, critic_loss = self.train_step(
            state, action, reward, next_state, done)

        # 타겟 네트워크 업데이트
        self.update_target()

        return actor_loss, critic_loss

    @tf.function
    def train_step(self, state, action, reward, next_state, done):
        # Critic 업데이트
        with tf.GradientTape() as tape:
            next_action = self.target_actor(next_state)
            next_q = self.target_critic(next_state, next_action)
            target_q = reward + (1-done)*discount_factor*next_q
            q = self.critic(state, action)
            critic_loss = tf.keras.losses.mean_squared_error(target_q, q)

        critic_grads = tape.gradient(
            critic_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_weights))
        with tf.GradientTape() as tape:
            q = self.critic(state, self.actor(state))
            actor_loss = -tf.reduce_mean(q)

        # Actor 업데이트
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_weights))
        return actor_loss, critic_loss

    # 타겟 네트워크 업데이트
    def update_target(self):
        if self.first_update:
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())
            self.first_update = False
        else:
            for t_a, a in zip(self.target_actor.trainable_weights, self.actor.trainable_weights):
                t_a.assign((1-tau)*t_a + tau*a)
            for t_c, c in zip(self.target_critic.trainable_weights, self.critic.trainable_weights):
                t_c.assign((1-tau)*t_c + tau*c)

    # 네트워크 모델 저장
    def save_model(self):
        print("... Save Model ...")
        self.actor.save_weights(save_path+'/actor')
        self.critic.save_weights(save_path+'/critic')

    # 학습 기록
    def write_summray(self, score, actor_loss, critic_loss, step):
        with self.writer.as_default():
            tf.summary.scalar("run/score", score, step=step)
            tf.summary.scalar("model/actor_loss", actor_loss, step=step)
            tf.summary.scalar("model/critic_loss", critic_loss, step=step)


# Main 함수 -> 전체적으로 DDPG 알고리즘을 진행
if __name__ == "__main__":
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

    # DDPGAgent 클래스를 agent로 정의
    agent = DDPGAgent()

    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(
                time_scale=1.0)

        state = dec.obs[0]
        action = agent.get_action(state, train_mode).numpy()
        env.set_actions(group_name, action)
        env.step()

        dec, term = env.get_steps(group_name)
        done = len(term.agent_id) > 0
        reward = term.reward[0] if done else dec.reward[0]
        next_state = term.obs[0] if done else dec.obs[0]
        score += reward

        if train_mode and len(state) > 0:
            agent.append_sample(state[0], action[0], [
                                reward], next_state[0], [done])

        if train_mode and step > max(batch_size, train_start_step):
            # 학습 수행
            actor_loss, critic_loss = agent.train_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        if done:
            episode += 1
            scores.append(score)
            score = 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록
            if episode % print_interval == 0:
                mean_score = tf.reduce_mean(scores)
                mean_actor_loss = tf.reduce_mean(actor_losses)
                mean_critic_loss = tf.reduce_mean(critic_losses)
                agent.write_summray(
                    mean_score, mean_actor_loss, mean_critic_loss, step)
                actor_losses, critic_losses, scores = [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +
                      f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")

            # 네트워크 모델 저장
            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()
