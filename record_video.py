import gym
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
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
n_actions = env.action_space.shape[-1]
action_noise =OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))
model = DDPG(LnMlpPolicy, env, param_noise=None,batch_size=64,buffer_size=1000000,enable_popart=False, action_noise=action_noise, verbose=4,seed=75,n_cpu_tf_sess=1)
model=model.load(r"/home/mohit/Downloads/stable-baselines/results_mohit/ddpg/Hopper-v2/None/75/best_model.pkl")
env_id = 'Hopper-v2'
video_folder = 'home/'
video_length = 1000

env = DummyVecEnv([lambda: gym.make(env_id)])

obs = env.reset()

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="random-agent-{}".format(env_id))

env.reset()
for _ in range(video_length + 1):
    action, _states = model.predict(obs)
      # here, action, rewards and dones are arrays
      # because we are using vectorized env
    obs, rewards, dones, info = env.step(action)
# Save the video
env.close()
