import matplotlib.pyplot as plt
import numpy as np
import math
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os

class MountainCar(object):
    def __init__(self,alpha=0.01, gamma=0.999, epsilon= 0.1, Lambda=0.55):
        super(MountainCar, self).__init__()          
        self.alpha       = alpha    #learning rate
        self.gamma       = gamma    #discount factor
        self.epsilon     = epsilon  #probability of a random action selection
        self.Lambda      = Lambda 
        self.actionlist  = self.BuildActionList()   # the list of actions        
        self.nactions    = self.actionlist.shape[0] # number of actions      
        self.num_rbf = 4 * np.ones(self.nactions).astype(int)
        self.width = 1. / (self.num_rbf - 1.)
        self.rbf_sigma = self.width[0] / 2.
        self.num_ind = np.prod(self.num_rbf)
        self.rbf_den = 2 * self.rbf_sigma ** 2
        self.centres = self.BuildCenters()
        
    def BuildActionList(self):
        return np.array([-1.0 , 0.0 , 1.0])

    def normalize_state(self, _s):
        xbar = np.zeros((2, 2))
        xbar[0, :] = (0.5, 0.07)
        xbar[1, :] = (-1.5, -0.07)
        _y = np.zeros(len(_s))
        for _i in range(len(_s)):
            _y[_i] = (_s[_i] - xbar[0, _i]) / (xbar[1, _i] - xbar[0, _i])
        return _y

    def BuildCenters(self):
        c = np.zeros((self.num_ind, 2))
        for i in range(self.num_rbf[0]):
            for j in range(self.num_rbf[1]):
                c[i*self.num_rbf[1] + j, :] = (i * self.width[1], j * self.width[0])
        return c

    def GetReward(self, x ):
        # MountainCarGetReward returns the reward at the current state
        # x: a vector of position and velocity of the car
        # r: the returned reward.
        # f: true if the car reached the goal, otherwise f is false
            
        position = x[0]
        # bound for position; the goal is to reach position = 0.45
        bpright  = 0.45

        r = -1
        f = False
        
        
        if  position >= bpright:
            r = 100
            f = True
        
        return r,f

    
    def DoAction(self, force, x ):
        #MountainCarDoAction: executes the action (a) into the mountain car
        # a: is the force to be applied to the car
        # x: is the vector containning the position and speed of the car
        # xp: is the vector containing the new position and velocity of the car

        position = x[0]
        speed    = x[1] 

        # bounds for position
        bpleft=-1.5 

        # bounds for speed
        bsleft=-0.07 
        bsright=0.07
         
        speedt1= speed + (0.001*force) + (-0.0025 * math.cos( 3.0*position) )	 
        speedt1= speedt1 * 0.999 # thermodynamic law, for a more real system with friction.

        if speedt1<bsleft: 
            speedt1=bsleft 
        elif speedt1>bsright:
            speedt1=bsright    

        post1 = position + speedt1 

        if post1<=bpleft:
            post1=bpleft
            speedt1=0.0
            
        xp = np.array([post1,speedt1])
        return xp


    def GetInitialState(self):
        initial_position = -0.5
        initial_speed    =  0.0    
        return  np.array([initial_position,initial_speed])

    def phi(self, _state):
        _phi = np.zeros(self.num_ind)
        for _k in range(self.num_ind):
            _phi[_k] = np.exp(-np.linalg.norm(_state - self.centres[_k, :]) ** 2 / self.rbf_den)
        return _phi
        
    def GetBestAction(self, s, theta):
        #GetBestAction return the best action for state (s)
        #Q: the Qtable
        #the current state
        #has structure  Q(states,actions)
        
        #a = argmax(self.Q[s,:].flat)
        a  = np.argmax(self.action_values(s, theta))    
        return a
    
    def action_values(self, _activations, _theta):
        _val = np.dot(_theta.T, _activations)
        return _val

    def action_value(self, _activations, _action, _theta):
        _val = np.dot(_theta[:, _action], _activations)
        return _val

    def epsilon_greedy(self, _epsilon, _vals):
        _rand = np.random.random()
        if _rand < 1. - _epsilon:
            _action = _vals.argmax()
        else:
            _action = random.randint(0, 2)
        #print(int(_action))
        return int(_action)

    def SARSAEpisode(self, maxsteps, theta):
        # do one episode with sarsa learning
        # maxstepts: the maximum number of steps per episode
        # Q: the current QTable
        # alpha: the current learning rate
        # gamma: the current discount factor
        # epsilon: probablity of a random action
        # statelist: the list of states
        # actionlist: the list of actions
        x_points = []
        y_points=[]
        e = np.zeros((self.num_ind, self.nactions))
        x                = self.GetInitialState()
        steps            = 0
        total_reward     = 0
        
        # convert the continous state variables to an index of the statelist
        
        s  = self.phi(self.normalize_state(x))
        vals = self.action_values(s, theta)
        #print(vals)
        # selects an action using the epsilon greedy selection strategy
        a = self.epsilon_greedy(self.epsilon, vals)

        for i in range(maxsteps):
                        
            # convert the index of the action into an action value
            action = self.actionlist[a]    
            
            # do the selected action and get the next car state    
            xp     = self.DoAction( action , x )    
            
            # observe the reward at state xp and the final state flag
            r, isfinal    = self.GetReward(xp)
            total_reward = total_reward + r

            # convert the continous state variables in [xp] to an index of the statelist
            sp     = self.phi(self.normalize_state(xp))

            new_vals = self.action_values(sp, theta)
            # select action prime
            ap     = self.epsilon_greedy(self.epsilon, new_vals)
            #print(ap)

            Q = self.action_value(s, a, theta)
            cost = -Q
            y_points.append(cost)
            Q_new = self.action_value(sp, ap, theta)

            if isfinal==True:
                target = r - Q
            else:
                target = r + self.gamma * Q_new - Q

            e[:, a] = s

            # Update the Qtable, that is,  learn from the experience
            for k in range(self.num_ind):
                for a in range(self.nactions):
                    theta[k, a] += self.alpha * target * e[k, a]
            
            e *= self.gamma * self.Lambda

            #update the current variables
            s = sp.copy()
            a = ap
            x = xp.copy()
                
            
            #increment the step counter.
            steps = steps+1
            
            # if reachs the goal breaks the episode
            if isfinal==True:
                break
        
        return total_reward, steps, theta
  


def MountainCarDemo(maxepisodes):
    MC  = MountainCar()
    maxsteps = 1000
    theta = np.zeros((64,3))
    xpoints=[]
    ypoints=[]

    for i in range(maxepisodes):    
        total_reward,steps, theta_prime = MC.SARSAEpisode( maxsteps, theta)    
        MC.epsilon *= 0.9999
        theta = theta_prime

        '''for j in range(len(x)):
            X.append(x[j][0]) 
            Y.append(x[j][1]) 
        if i==0 or i==12 or i==104 or i==1000 or i==9000:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            #ax.plot_trisurf(X, Y, Z , cmap=cm.viridis, linewidth=0.2)            
            surf=ax.plot_trisurf(X, Y, Z, cmap=cm.viridis, linewidth=0.2)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            fig.colorbar( surf, shrink=0.5, aspect=5)
            filename=os.getcwd() + '/ANIMATION/Volcano_step'+str(i)+'.png'
            plt.savefig(filename, dpi=96)

        #ax.scatter(X, Y, Z, c='skyblue', s=60)
        ax.view_init(30,angle)
        #os.getcwd() + '/train/actor_params.pth'
        filename=os.getcwd() + '/ANIMATION/Volcano_step'+str(angle)+'.png'
        plt.savefig(filename, dpi=96)
        plt.gca()'''

        
        #plt.show()

        print ('Espisode: ',i,'  Steps:',steps,'  Reward:',str(total_reward),' epsilon: ',str(MC.epsilon))
        xpoints.append(i)
        ypoints.append(-total_reward)
        #MC.epsilon = MC.epsilon * 0.99
        
        #xpoints.append(i)
        #ypoints.append(-total_reward)

    plt.plot(xpoints, ypoints)
    plt.xlabel("Episodes")
    plt.ylabel("-Rewards")
    plt.show()


                
if __name__ == '__main__':
    MountainCarDemo(2000)              
