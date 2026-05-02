import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

env = make_atari_env("ALE/Qbert-v5", n_envs=1, seed=42, 
                     env_kwargs={"render_mode": "human"})
env = VecFrameStack(env, n_stack=4)

model = PPO.load("qbert_ppo_5m", env=env)

obs = env.reset()
total_reward = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]
    if done[0]:
        print(f"Game over! Score: {total_reward}")
        total_reward = 0
        obs = env.reset()
