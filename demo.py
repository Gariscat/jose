import gymnasium as gym
from stable_baselines3 import SAC, DDPG

env_name = 'Ant-v4'

env = gym.make(
    env_name,
    render_mode="human",
    max_episode_steps=1000,
    exclude_current_positions_from_observation=False
)
print(f"Observation shape: {env.observation_space.shape}")
print(f"Action shape: {env.action_space.shape}")


model = SAC("MlpPolicy", env, verbose=1,
    seed=101,
)
model.learn(total_timesteps=int(5e5), log_interval=10)
model.save(f"ckpt/{env_name}")

del model # remove to demonstrate saving and loading
""""""
model = SAC.load(f"ckpt/{env_name}")


for i in range(10):
    print(f'eval episode: {i}')
    obs, info = env.reset()
    while True:
        env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

env.close()
