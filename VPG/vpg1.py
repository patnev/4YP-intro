#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque
import gym_bike

v_dif_f = 0.001
draft_f = 0.01
f_3 = 5
f_4 = 1

alpha_dis = 0.999
alpha = 1
pre_v = 0
a_prev =0

def reward_exp(obs):
    
    a = abs(pre_v - obs[1]) *f_3
    a2 = abs(a_prev - a)*f_4
    #v_dif = (abs(obs[1] - 32))*v_dif_f
    draft = -abs(obs[0]-obs[2]- 60)*draft_f
    
    return (5- draft )/10, a

# define policy network
class policy_net(nn.Module):
    def __init__(self, nS, nH, nA): # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
        super(policy_net, self).__init__()
        self.h = nn.Linear(nS, nH)
        self.out = nn.Linear(nH, nA)

    # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.softmax(self.out(x), dim=1)
        return x

# create environment
env = gym.make("bike-v3")
# instantiate the policy
policy = policy_net(env.observation_space.shape[0], 20, env.action_space.n)
# create an optimizer
optimizer = torch.optim.Adam(policy.parameters())
# initialize gamma and stats
gamma=0.99
c=0
n_episode = 1
returns = deque(maxlen=100)
render_rate = 200 # render every render_rate episodes
while True:
    pre_v =0
    a_prev =0
    rewards = []
    actions = []
    states  = []
    alpha *= alpha_dis
    # reset environment
    state = env.reset()
    while True:
        # render episode every render_rate epsiodes
        if n_episode%render_rate==0:
            env.render()

        # calculate probabilities of taking each action
        probs = policy(torch.tensor(state).unsqueeze(0).float())
        # sample an action from that set of probs
        sampler = Categorical(probs)
        action = sampler.sample()

        # use that action in the environment
        new_state, act_reward, done, info = env.step(action.item())
        rew_exp , a_prev = reward_exp(new_state)
        reward = rew_exp*alpha + (1-alpha)*act_reward
        
        # store state, action and reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        c +=1
        pre_v = new_state[1]

        state = new_state
        if done:
            
            break

    # preprocess rewards
    rewards = np.array(rewards)
    # calculate rewards to go for less variance
    R = torch.tensor([np.sum(rewards[i:]*(gamma**np.array(range(i, len(rewards))))) for i in range(len(rewards))])
    # or uncomment following line for normal rewards
    #R = torch.sum(torch.tensor(rewards))

    # preprocess states and actions
    states = torch.tensor(states).float()
    actions = torch.tensor(actions)

    # calculate gradient
    probs = policy(states)
    sampler = Categorical(probs)
    log_probs = -sampler.log_prob(actions)   # "-" because it was built to work with gradient descent, but we are using gradient ascent
    pseudo_loss = torch.sum(log_probs * R) # loss that when differentiated with autograd gives the gradient of J(θ)
    # update policy weights
    optimizer.zero_grad()
    pseudo_loss.backward()
    optimizer.step()

    # calculate average return and print it out
    returns.append(np.sum(rewards))
    print("Episode: {:6d}\tAvg. Return: {:6.2f},,,{}".format(n_episode, np.mean(returns),c))
    n_episode += 1
    c=0

# close environment
env.close()


    
    







