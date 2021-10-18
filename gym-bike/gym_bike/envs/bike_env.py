import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import cv2
import numpy as np

draft_f = 0.01
gamma = 0.999
def reward_exp(obs):
    
    # a = abs(pre_v - obs[1]) *f_3
    # a2 = abs(a_prev - a)*f_4
    # #v_dif = (abs(obs[1] - 32))*v_dif_f
    draft = -abs(obs[2]-obs[0]- 60)*draft_f
    
    return (5+draft )/10

class BikeEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  
  def __init__(self):
      self.agent = Cyclist(40000,1000,0,0.1,80,0.4)
      self.control = Cyclist(40000,1000,0,0.1,80,0.4)
      self.episode_count = 0
      self.time = 0
      
      self.observation_shape = (600, 1600, 3)
      self.canvas = np.ones(self.observation_shape) * 1
      high = np.array([10000,1000,10000,1000],dtype=np.float32)
      low = np.array([0,0,0,0],dtype=np.float32)
      self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
      self.dt = 0.5
      self.max_power = 990
      #self.action_space = spaces.Box(
       #     low=0, high=self.max_power, shape=(1,), dtype=np.float32
       # )
      self.action_space = spaces.Discrete(2)
      self.sep = 0
      self.state = None
      
   
  def step(self, a_action):
     self.sep = abs(self.control.pos - self.agent.pos) 
     
     #assert self.action_space.contains(a_action), "Invalid Action"
     a_action = a_action * self.max_power
     
     c_action = self.controllerAction()
        
     self.control.modelUpdate(self.sep, c_action, self.isLeading(self.control,self.agent), self.dt)
     self.agent.modelUpdate(self.sep, a_action, self.isLeading(self.agent,self.control), self.dt)
     self.time += self.dt
     
     
     
     
     self.state = [self.agent.pos,self.agent.vel,self.control.pos,self.control.vel]
     act_reward = 0
     if self.agent.pos >= 1450:
         done = True
         if self.agent.pos>= self.control.pos:
             act_reward = 1
     else:
        done = False
    
     #reward = - abs(self.control.pos - self.agent.pos -50)
     alpha = gamma**self.episode_count
     rew_exp  = reward_exp(self.state)
     reward = rew_exp*alpha + (1-alpha)*act_reward
     
     return np.array(self.state, dtype=np.float32),reward,done,{}
 
    
  def reset(self):
         self.episode_count += 1

         self.agent.pos = 0
         self.agent.vel = 0.1
         self.agent.energy = 40000
         
         self.control.pos = 0
         self.control.vel = 0.1
         self.control.energy = 40000
         self.canvas = np.ones(self.observation_shape) * 1
         
         self.time = 0
         self.state = [self.agent.pos,self.agent.vel,self.control.pos,self.control.vel]

         
         return np.array(self.state,dtype=np.float32)
     
    
  def render(self, mode='human'):
    self.canvas = np.ones(self.observation_shape) * 1  
    if  self.sep <= 50:
        cv2.rectangle(self.canvas,( int(self.control.pos),200), (int(self.control.pos + 50),250), (0,0,0), -1)
        cv2.rectangle(self.canvas,( int(self.agent.pos),250), (int(self.agent.pos + 50),300), (0,0,200), -1)
        
    else:
        cv2.rectangle(self.canvas,( int(self.control.pos),200), (int(self.control.pos + 50),250), (0,0,0), -1)
        cv2.rectangle(self.canvas,( int(self.agent.pos),200), (int(self.agent.pos + 50),250), (0,0,200), -1)
        
    cv2.line(self.canvas,(1500,0),(1500,600),(0,200,0),3)

   
    
    text_a= 'agent : pos: {:.1f} | vel: {:.1f}| energy{:.1f}' .format(self.agent.pos, self.agent.vel, self.agent.energy)
    text_c= 'contol : pos: {:.1f} | vel: {:.1f}| energy{:.1f}'.format(self.control.pos, self.control.vel, self.control.energy)
    text_d = 'step: {:}'.format(self.episode_count)
    # Put the info on canvas 
    self.canvas = cv2.putText(self.canvas, text_c, (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL ,  
               0.8, (0,0,0), 1, cv2.LINE_AA)
    self.canvas = cv2.putText(self.canvas, text_a, (10,100), cv2.FONT_HERSHEY_COMPLEX_SMALL ,  
               0.8, (0,0,0), 1, cv2.LINE_AA)
    self.canvas = cv2.putText(self.canvas, text_d, (10,600), cv2.FONT_HERSHEY_COMPLEX_SMALL ,  
               0.8, (0,0,0), 1, cv2.LINE_AA)
    
    cv2.imshow("test",self.canvas)
    cv2.waitKey(1)
    
    return 
    
  def close(self):
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    
  def controllerAction(self):
      
      return self.control.max_power
    #should be "is the bike not draughting?"
  def isLeading(self,b1,b2):
      if b1.pos +50 >= b2.pos:
          return True
      else:
          return False





class Cyclist:
    
    p = 1.225
    
    def __init__(self,init_energy,max_power,init_pos,init_vel,mass,ca):
        self.energy = init_energy
        self.max_power = max_power
        self.energy_left = init_energy
        self.pos = init_pos
        self.vel = init_vel
        self.mass = mass
        self.ca = ca
        self.time = 0
        
    def cReduct(self,sep):
        
        #seperation in m with 25 in the scale ratio
        return min(0.48 + 0.2/(25)*sep,1)
    
    def airRes(self):
        a = 0.5*self.ca*self.p*self.vel
        return a
        
    def modelUpdate(self,sep,p,leading,dt):
        self.time += dt
        #solution to F - f(c,v) = m a
        
        if p>self.max_power:
            p = self.max_power
        if self.energy <= 0:
            p = 0
        else:
            self.energy -= p*dt
        if leading == True:
            self.pos = (((p/self.vel)-self.airRes())/self.mass)*dt**2 +self.vel*dt + self.pos
            self.vel = (((p/self.vel)-self.airRes())/self.mass)*dt + self.vel
        else:
            self.pos = (((p/self.vel)-self.airRes()*self.cReduct(sep))/self.mass)*dt**2 +self.vel*dt + self.pos
            self.vel = (((p/self.vel)-self.airRes()*self.cReduct(sep))/self.mass)*dt + self.vel
        
        
# agent = Cyclist(10000,1000,0,0.3,80,0.4)
# pos = []
# vel = []
# for i in range(2000):
#     pos.append(agent.pos)
#     vel.append(agent.vel)
#     agent.modelUpdate(10, 1000, True, 0.01)
    
# #plt.plot(pos)
# plt.plot(vel)
# plt.show()
# print(vel)




        
