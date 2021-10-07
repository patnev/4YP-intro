import gym_bike
import gym
env = gym.make('bike-v3')

    
   
for i in range(1000):
    env.render()
    if i % 10 :
        
        obs, rew,done, _ =env.step(1) # take a random action
    else:
        obs, rew,done, _ =env.step(1) # take a random action
    print(rew)
    if done == True:
        break
    
env.close()



