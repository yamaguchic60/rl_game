import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np



# 環境の作成
env = gym.make("CartPole-v1",render_mode="human")


# モデルの読み込み
model = PPO.load("./train_data/ppo_cartpole", env=env)

obs = env.reset()[0]
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()  # エージェントの動きを表示
env.close()
