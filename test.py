import gymnasium as gym
import ale_py

def main():
    gym.register_envs(ale_py)
    env = gym.make('ALE/Breakout-v5')
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    

if __name__ == "__main__":
    main()