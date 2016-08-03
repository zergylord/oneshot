'''
currently set to a single goal a [0,0] since the state space doesn't currently contain goal info
'''
import numpy as np

class ActionSpace(object):
    def __init__(self,size):
        self.n = size
    def sample(self):
        return np.random.randint(self.n)
class ObservationSpace(object):
    def __init__(self,shape):
        self.shape = (*shape,)
class FourRooms(object):
    action_space = ActionSpace(4)
    observation_space = ObservationSpace([2])
    def __init__(self,size,center_passage = False):
        if size % 2 == 0:
            raise NameError('size should be odd')
        if size < 3:
            raise NameError('room should be size 3 or larger')
        self.size = size
        self.half = int(size/2) #round down
        #-----setup walls------------
        if center_passage:
            if size < 5:
                raise NameError('room should be size 5 or larger')
            #vertical walls
            bad_rows = list(range(0,self.half-1)) + list(range(self.half+2,self.size))
            bad_cols = [self.half]*(self.size-3)
            #horizontal walls
            bad_rows += [self.half]*(self.size-3)
            bad_cols += list(range(0,self.half-1)) + list(range(self.half+2,self.size))
            self.bad_inds = np.ndarray.tolist(np.concatenate((np.asarray((bad_rows,)),np.asarray((bad_cols,))),0).T) 
        else:
            #vertical walls
            bad_rows = list(range(1,self.size-1))
            bad_cols = [self.half]*(self.size-2)
            #horizontal walls
            bad_rows += [self.half]*(self.size-2)
            bad_cols += list(range(1,self.size-1))
            self.bad_inds = np.ndarray.tolist(np.concatenate((np.asarray((bad_rows,)),np.asarray((bad_cols,))),0).T) 
        #-----setup starting pos-----
        start_rows = list(range(0,self.half))*2 + list(range(self.half+1,self.size))*2
        start_cols = (list(range(0,self.half)) + list(range(self.half+1,self.size)) )*2
        self.start_inds = np.ndarray.tolist(np.concatenate((np.asarray((start_rows,)),np.asarray((start_cols,))),0).T) 
        self.goal_ind = [0,0]
        self.reset()
    def reset(self):
        self.cur_ind = list(self.start_inds[np.random.randint(len(self.start_inds))])
        '''
        self.goal_ind = list(self.start_inds[np.random.randint(len(self.start_inds))])
        if self.goal_ind in self.bad_inds:
            print(self.goal_ind)
            raise NameError('poo')
        '''
        return list(self.cur_ind)
    def get_image(self):
        grid = np.zeros((self.size,self.size))
        for i in self.bad_inds:
            grid[i[0],i[1]] += -1
        grid[self.cur_ind[0],self.cur_ind[1]] += 2
        grid[self.goal_ind[0],self.goal_ind[1]] += 1
        return grid.astype(int)

    def step(self,a):
        if not (a in range(4)):
            raise NameError('action must be an int between 0 and 3')
        r = -.01
        term = False
        self.prev_ind = list(self.cur_ind)
        if a == 0: #up
            self.cur_ind[0]-=1
        elif a == 1: #left
            self.cur_ind[1]-=1
        elif a ==2: #down
            self.cur_ind[0]+=1
        elif a == 3: #right
            self.cur_ind[1]+=1
        if (self.cur_ind in self.bad_inds 
                or self.cur_ind[0] < 0
                or self.cur_ind[1] < 0
                or self.cur_ind[0] >= self.size
                or self.cur_ind[1] >= self.size):
            #print('hit wall!')
            self.cur_ind = self.prev_ind
            r = -1.0
            term = True
        elif self.cur_ind == self.goal_ind:
            #print('you win!')
            r = 1.0
            term = True
        return list(self.cur_ind),r,term,'stupid value'
    def test_animation(self,steps):
        import os,sys,time
        os.system('setterm -cursor off')
        tot_r = 0
        for i in range(steps):
            s,r,term = env.step(np.random.randint(4))
            tot_r+=r
            time.sleep(.01)
            os.system('clear')
            sys.stdout.write(str(env.get_image())+"\r")
            if term:
                env.reset()
        os.system('setterm -cursor on')

'''
env = FourRooms(15)
env.test_animation(1000)
for i in range(10000):
    env.reset()
'''

