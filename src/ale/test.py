import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", frameskip=4, render_mode = "human")
env.reset(seed=42)
