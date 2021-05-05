import gym
import numpy as np
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines.ddpg import OrnsteinUhlenbeckActionNoise,AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import DDPG
from stable_baselines.common import set_global_seeds
set_global_seeds(75)
env = gym.make('Hopper-v2')
env.seed(75)
# vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
n_actions = env.action_space.shape[-1]
action_noise =OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))
model = DDPG(LnMlpPolicy, env, param_noise=None,batch_size=64,buffer_size=1000000,enable_popart=False, action_noise=action_noise, verbose=4,seed=75,n_cpu_tf_sess=1)
model=model.load(r"/home/mohit/Downloads/stable-baselines/results_mohit/ddpg/Hopper-v2/None/75/best_model.pkl")
env_id = 'Hopper-v2'

env = DummyVecEnv([lambda: gym.make(env_id)])
def evaluate(model, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs = env.reset()
  for i in range(num_steps):
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)
      # here, action, rewards and dones are arrays
      # because we are using vectorized env
      obs, rewards, dones, info = env.step(action)
      
      # Stats
      episode_rewards[-1] += rewards[0]
      if dones[0]:
          obs = env.reset()
          episode_rewards.append(0.0)
  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
  return mean_100ep_reward

avg=0.0

for i in range(0,100):
	mean_reward_before_train = evaluate(model, num_steps=1000)
	avg=avg+mean_reward_before_train
avg=avg/100
print("Average:",avg)
