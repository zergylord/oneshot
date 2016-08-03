import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import time
from sklearn.neighbors import NearestNeighbors
from ops import *
from grid_env import *
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
gamma = 1.0
epsilon = .005
num_neighbors = 1
env = FourRooms(25)
#env = gym.make('Pong-v0')
#env = gym.make('Frostbite-v0')
#env = gym.make('SpaceInvaders-v0')
n_act = env.action_space.n
print(n_act)
knn = []
for a in range(n_act):
    knn.append(NearestNeighbors(n_neighbors=num_neighbors,metric='euclidean'))#,algorithm='brute'))
s_dim = env.observation_space.shape[0]
cur_time = time.clock()
s = env.reset()
#env.render()
def process_obs(obs):
    return obs
def get_action(rep):
    #global knn,n_act
    max_q_val = float("-inf")
    act = 0
    tied = []
    nearby = []
    exact_match = {}
    for a in range(n_act):
        nearby.append([])
        dists,inds = knn[a].kneighbors(np.expand_dims(rep,0))
        dists = dists[0]
        inds = inds[0]
        nearby[a] = inds
        if 0.0 in dists:
            hit_ind = inds[dists == 0.0][0]
            #print('hit!',hit_ind,R[a][hit_ind])
            q_val = R[a][hit_ind]
            exact_match[a] = hit_ind
        else:
            q_val = np.mean(R[a][inds])
        if q_val > max_q_val:
            max_q_val = q_val
            act = a
            tied = []
            a_not_in_tied = True
        elif a != act and q_val == max_q_val:
            if not tied:
                tied.append(act)
            tied.append(a)
            act = a
    if tied:
        act = np.random.choice(tied) 
    return act,nearby,exact_match



mem_size = int(1e6)
rep_dim = 2 #64
M = np.random.randn(s_dim,rep_dim)
S = np.zeros((n_act,mem_size,rep_dim))
R = np.zeros((n_act,mem_size,))
last_used = np.tile(np.asarray(range(-mem_size,0)),[n_act,1])
mem_ind = np.zeros((n_act,),dtype=int)
total_hits = 0
Ret = 0
cumr = 0.0
episodes = 0.0
warming = True
refresh = int(1e4)
plt.ion()
r_hist = []
def get_state(obs):
    return obs
    #return np.matmul(obs,M)
def episode_reset():
    global cur_step,episode_states,episode_matches,episode_match_inds,episode_actions,episode_rewards
    cur_step = 0
    episode_states = []
    episode_matches = []
    episode_match_inds = []
    episode_actions = []
    episode_rewards = []
episode_reset()
for i in range(int(1e7)):
    obs = process_obs(s)
    #----action selection--------------
    if not warming:
        action,nearby,match = get_action(get_state(obs))
        for a in range(n_act):
            last_used[a,nearby[a]] = i

    if np.random.rand() < epsilon or warming:
        action = env.action_space.sample()
    if not warming and action in match:
        episode_matches.append(True)
        episode_match_inds.append(match[action])
        total_hits+=1
    else:
        episode_matches.append(False)
    
    episode_actions.append(action)
    episode_states.append(obs)
    reward = 0.0
    s,r,done,_ = env.step(action)
    cur_step+=1
    if cur_step >= 1e3:
        done = True
    reward+=r
    episode_rewards.append(reward)
    #env.render()
    Ret+=reward

    #-------------------end of episode processing---------------------------
    if done:
        episodes+=1
        #print(i,'done!',Ret)
        cumr+=Ret
        if warming:
            if i > 250:
                warming = False
        s = env.reset()
        episode_rets = np.asarray(compute_return(episode_rewards,gamma))
        episode_states = np.asarray(episode_states)
        episode_actions = np.asarray(episode_actions)
        episode_matches = np.asarray(episode_matches)
        if np.any(episode_matches):
            #update matched return estimates
            match_act_inds = episode_actions[episode_matches]
            R[match_act_inds,episode_match_inds] = np.maximum(R[match_act_inds,episode_match_inds],episode_rets[episode_matches])
            if not np.all(episode_matches):
                #remove matches from list to add to memory
                neg = np.logical_not(episode_matches)
                episode_rets = episode_rets[neg]
                episode_actions = episode_actions[neg]
                episode_states = episode_states[neg]
                add_memories = True
            else:
                add_memories = False
        else:
            add_memories = len(episode_states)>0
        if add_memories:
            #-----add stuff to memory------------
            episode_reps = np.asarray(episode_states)
            for a in range(n_act):
                mask = episode_actions==a
                n_reps = len(episode_actions[mask])
                if n_reps > 0:
                    replace_these = np.argpartition(last_used[a],n_reps-1)[:n_reps]
                    last_used[a][replace_these] = i
                    S[a,replace_these] = episode_reps[mask]
                    R[a,replace_these] = episode_rets[mask]
                    if mem_ind[a] + n_reps < mem_size:
                        mem_ind[a] += n_reps
                    else:
                        mem_ind[a] = mem_size
                    knn[a].fit(S[a][:mem_ind[a]])

        episode_reset()
        episode_actions = []
        Ret = 0
    if i >0 and i % refresh == 0:
        '''
        for a in range(n_act):
            print(unique_rows(S[a]))
        '''
        plt.clf()
        #plt.plot(last_used[0])
        r_hist.append(cumr/(episodes+1e-10))
        plt.plot(r_hist)
        plt.pause(.1)
        print(i,'reward per episode: ',cumr/(episodes+1e-10),'steps per episode: ',refresh/(episodes+1e-10),'hit %: ',total_hits/refresh, 'time: ', time.clock()-cur_time)
        total_hits = 0.0
        episodes = 0.0
        cumr = 0.0
        cur_time = time.clock()
