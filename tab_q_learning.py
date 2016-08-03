import numpy as np
from grid_env import *
import matplotlib.pyplot as plt
epsilon = .005
alpha = .01
gamma = 1.0
side_size = 25
env = FourRooms(side_size)
num_states = side_size**2
num_actions = env.action_space.n
Q = np.random.rand(side_size,side_size,num_actions)*0.0
row,col = env.reset()
cumr = 0.0
cume = 0.0
refresh = int(1e4)
for i in range(int(1e6)):
    if np.random.rand() < epsilon:
        a = env.action_space.sample()
    else:
        a = np.argmax(Q[row,col])
    (new_row,new_col),r,done,_ = env.step(a)
    cumr+=r
    if done:
        cume+=1
        target = r
        new_row,new_col = env.reset()
    else:
        target = r+gamma*max(Q[new_row,new_col])
    Q[row,col,a] = (1-alpha)*Q[row,col,a] + alpha*(target)
    row,col = new_row,new_col
    if i % refresh == 0:
        print('{},{:0.2f},{:0.1f}'.format(i,cumr/(cume+1e-10),refresh/(cume+1e-10)))
        #print('{},{:0.3f}'.format(i,cumr/(refresh+1e-10)))
        cumr = 0.0
        cume = 0.0

