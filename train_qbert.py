import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

env = make_atari_env("ALE/Qbert-v5", n_envs=4, seed=42)
env = VecFrameStack(env, n_stack=4)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./qbert_tensorboard/",
    n_steps=64,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    learning_rate=2.5e-4,
)

print("Training started!")
model.learn(total_timesteps=5_000_000)
model.save("qbert_ppo_5m")
print("Done! Run watch_qbert.py to see what it learned.")
