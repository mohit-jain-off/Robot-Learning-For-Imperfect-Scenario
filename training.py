# Ddpg stable_baseline
#not now 1221,5000,2048,6557,7331
sed=[69,75,100,343]
for difsed in range(4):
    seed=sed[difsed]
    import gym
    import random
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from stable_baselines.ddpg.policies import LnMlpPolicy
    from stable_baselines.bench import Monitor
    from stable_baselines.results_plotter import load_results, ts2xy
    from stable_baselines import DDPG
    from stable_baselines.ddpg import OrnsteinUhlenbeckActionNoise,AdaptiveParamNoiseSpec
    from datetime import datetime, timedelta
    import time
    from mpi4py import MPI
    from stable_baselines import logger
    from stable_baselines.common import set_global_seeds
    set_global_seeds(seed)
    best_mean_reward, n_steps = -np.inf, 0
    def callback(_locals, _globals):
        global log_dir
        #print("4 my callbck ",np.random.get_state()[1][0])
        global n_steps, best_mean_reward
        # Print stats every 1000 calls3
        if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                #print(x[-1], 'timesteps')
                #print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    #print("Saving new best model")
                    _locals['self'].save(log_dir + 'best_model.pkl')
        n_steps += 1
        # Returning False will stop training early
        return True

    
    path="/home/mohit/Downloads/stable-baselines/results_mohit/ddpg"
    #"InvertedPendulum-v2","Hopper-v2",int(1e6),int(1e6), "Humanoid-v2"
    environment=["Humanoid-v2"]
    ts=[int(1e6),int(1e6)]
    for name in range(len(environment)):
        exp="None"
        log_dir = path+"/"+environment[name]+"/"+exp+"/"+str(seed)+"/"
        os.makedirs(log_dir, exist_ok=True)
        env = gym.make(environment[name])
        env.seed(seed)
        env =  Monitor(env, log_dir, allow_early_resets=True)
        n_actions = env.action_space.shape[-1]
        action_noise =OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))
        param_noise = None
        #param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
        model = DDPG(LnMlpPolicy, env, param_noise=param_noise,batch_size=64,buffer_size=1000000,enable_popart=False, action_noise=action_noise, verbose=4,seed=seed,n_cpu_tf_sess=1)
        model.learn(total_timesteps=ts[name], callback=callback,log_dir =log_dir)
        model.save(log_dir + "ddpg_"+environment[name])
        print("seed  ",np.random.get_state()[1][0])
        #env = Monitor(env, log_dir, allow_early_resets=True)
        n_actions = env.action_space.shape[-1]
        start_time=time.time()
        print(str(seed),'start',start_time,'elapsed',timedelta(seconds=time.time()-start_time),"\n")
        file=open(path+"/time.txt","a")
        file.write(str(seed)+environment[name]+exp+' start '+str(start_time)+' elapsed '+str(timedelta(seconds=time.time()-start_time))+"\n")
        file.close()
