import gymnasium as gym
from stable_baselines3 import PPO

# 環境の作成
env = gym.make("CartPole-v1",render_mode="human")

# エージェントの作成 (PPOアルゴリズム)
model = PPO("MlpPolicy", env, verbose=1)

# エージェントのトレーニング
model.learn(total_timesteps=10000)


import numpy as np

def evaluate_agent(env, model, num_episodes=10):
    all_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()[0]  # 初期観測を取得
        done = False
        total_reward = 0
        while not done:
            # 最適なアクションを選択
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        all_rewards.append(total_reward)
    print(f"平均報酬: {np.mean(all_rewards)}")

# 評価の実行
evaluate_agent(env, model)

# モデルの保存
model.save("./train_data/ppo_cartpole")

# モデルの読み込み
model = PPO.load("./train_data/ppo_cartpole", env=env)

obs = env.reset()[0]
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()  # エージェントの動きを表示
env.close()
