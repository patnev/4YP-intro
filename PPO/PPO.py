from spinup import ppo_pytorch
import torch
import gym
from spinup.algos.pytorch.ppo.core import MLPActorCritic
import gym_bike



env_fn = lambda : gym.make('bike-v3')
# env = gym.make('CartPole-v0')
# observation_space = env.observation_space.shape[0]
# action_space = env.action_space.n
ac_kwargs = dict(hidden_sizes=[64,64])

logger_kwargs = dict( exp_name='PPO_Cartpole_1.2', output_dir='/Users/admin/Documents/4YP-intro/results/PPO_Cartpole_1.2')



ppo_pytorch(env_fn, 
                  ac_kwargs=ac_kwargs, seed=0, steps_per_epoch=4000, epochs=5000, gamma=0.99, 
                  clip_ratio=0.2, pi_lr=0.0003, vf_lr=0.001, train_pi_iters=80, train_v_iters=80, 
                  lam=0.97, max_ep_len=1000, target_kl=0.1, logger_kwargs=logger_kwargs, save_freq=10)

