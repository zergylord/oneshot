#TODO: add some dummy data in S,R,mem_ind, and see if it runs
'''
matching network architecture applied to model-free episodic control.

Works like matching network, except using returns instead of labels, and
instead of inference from a few random examples, inference 
via cosine similarity is performed across a bank of the N most recently used states.
Stores seperate memory bank for each action, so inference must be performed several times
per action selection.
'''
from grid_env import * 
from ops import *
import numpy as np
import time
cur_time = time.time()
env = FourRooms(9)#size of one side of square image
mb_dim = 32 #training examples per minibatch
hid_dim = 10
out_dim = 64
x_dim = env.observation_space.shape[0]  
n_act = env.action_space.n
eps = 1e-10 #term added for numerical stability of log computations
tie = False #tie the weights of the query network to the labeled network
x_i_learn = True #toggle learning for the query network
learning_rate = 1e-1
warming_steps = 250
'''
setup episodic memory
'''
mem_size = int(1e3)
S = np.zeros((n_act,mem_size,out_dim))
R = np.zeros((n_act,mem_size,))
#usage normally strictly positive, initialization ensures memory bank fills up in order
usage = np.tile(np.asarray(range(-mem_size,0)),[n_act,1])
mem_ind = np.zeros((n_act,),dtype=int)



'''
randomly select a state from *one* of the action memory banks, and try
and predict its Return based on other random samples from the same bank.

Perhaps the samples chosen should be weighted by their usage?
'''
def get_minibatch():
    a = np.random.randint(n_act)
    mb_x_i = np.zeros((mb_dim,n_samplies,x_dim))
    mb_y_i = np.zeros((mb_dim,n_samples))
    mb_x_hat = np.zeros((mb_dim,x_dim))
    mb_y_hat = np.zeros((mb_dim,))
    
    for i in range(mb_dim):
        example_inds = np.random.choice(mem_ind[a],n_samples,False)
        mb_x_hat[i] = S[a][example_inds[0]]
        mb_y_hat[i] = R[a][example_inds[0]]
        mb_x_i[i] = S[a][example_inds[1:]]
        mb_y_i[i] = R[a][example_inds[1:]]
    return mb_x_i,mb_y_i,mb_x_hat,mb_y_hat



                



import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/oneshot_logs', 'Summaries directory')
if tf.gfile.Exists(FLAGS.summary_dir):
    tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)


'''
    basic feedforward network
'''
def make_net(inp,scope,reuse=False,stop_grad=False):
    with tf.variable_scope(scope) as varscope:
        if reuse: varscope.reuse_variables()
        hid = linear(inp,hid_dim,'ff0',tf.nn.relu)
        output = linear(hid,out_dim,'ff1')
    if stop_grad:
        return tf.stop_gradient(output)
    else:
        return output
'''
    assemble a computational graph for processing minibatches of the n_samples labeled examples and one unlabeled sample.
    All labeled examples use the same convolutional network, whereas the unlabeled sample defaults to using different parameters.
    After using the convolutional networks to encode the input, the pairwise cos similarity is computed. The normalized version of this
    is used to weight each label's contribution to the queried label prediction.
'''
'''
inference graph
'''
x_hat = tf.placeholder(tf.float32,shape=[1,x_dim])
y_hat = tf.placeholder(tf.float32,shape=[1])
x_i = tf.placeholder(tf.float32,shape=[n_act,None,1,x_dim])
y_i = tf.placeholder(tf.float32,shape=[n_act,None])
scope = 'encode_x'
x_hat_encode = make_net(x_hat,scope)
#x_hat_inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(x_hat_encode),1,keep_dims=True),eps,float("inf")))
cos_sim_list = []
cos_sim = []
for a in range(n_act):
    cos_sim_list.append([])
    if not tie:
        scope = 'encode_x_i_' + str(a)
    for i in range(mem_ind[a]):
        x_i_encode = make_net(x_i[a,i],scope,tie or i > 0,not x_i_learn)
        x_i_inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(x_i_encode),1,keep_dims=True),eps,float("inf")))
        dotted = tf.squeeze(
            tf.matmul(tf.expand_dims(x_hat_encode,0),tf.expand_dims(x_i_encode,1)))
        cos_sim_list[a].append(dotted
                *x_i_inv_mag)
                #*x_hat_inv__mag
    cos_sim.append(tf.concat(1,cos_sim_list[a]))
    tf.histogram_summary('cos sim_'+str(a),cos_sim[a])
    weighting = tf.nn.softmax(cos_sim[a])
    return_pred.append(tf.squeeze(tf.matmul(tf.expand_dims(weighting,0),tf.expand_dim(y_i[a],1))))
    tf.histogram_summary('return pred' + str(a),return_pred[a])
q_values = tf.concat(return_pred,1)

'''
train graph
x_hat = tf.placeholder(tf.float32,shape=[None,x_dim])
y_hat = tf.placeholder(tf.float32,shape=[None])
x_i = tf.placeholder(tf.float32,shape=[None,None,x_dim])
y_i = tf.placeholder(tf.float32,shape=[None,None])

top_k = tf.nn.in_top_k(label_prob,y_hat_ind,1)
acc = tf.reduce_mean(tf.to_float(top_k))
tf.scalar_summary('train avg accuracy',acc)
correct_prob = tf.reduce_sum(tf.log(tf.clip_by_value(label_prob,eps,1.0))*y_hat,1)
loss = tf.reduce_mean(-correct_prob,0)
tf.scalar_summary('loss',loss)
optim = tf.train.GradientDescentOptimizer(learning_rate)
#optim = tf.train.AdamOptimizer(learning_rate)
grads = optim.compute_gradients(loss)
grad_summaries = [tf.histogram_summary(v.name,g) if g is not None else '' for g,v in grads]
train_step = optim.apply_gradients(grads)

#testing stuff
test_acc = tf.reduce_mean(tf.to_float(top_k))
'''

'''
    End of the construction of the computational graph. The remaining code runs training steps.
'''

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter(FLAGS.summary_dir,sess.graph)
sess.run(tf.initialize_all_variables())
for i in range(int(1e7)):
    s,r,done,_ = env.step(env.action_space.sample())
    if i >= warming_steps:
        qvals,summary,sim = sess.run([q_values,merged,cos_sim],feed_dict=feed_dict)
    if done:
        :
    '''
    execute training step:

    mb_x_i,mb_y_i,mb_x_hat,mb_y_hat = get_minibatch()
    feed_dict = {x_hat: mb_x_hat,
                y_hat_ind: mb_y_hat,
                x_i: mb_x_i,
                y_i_ind: mb_y_i}
    _,mb_loss,summary,ans = sess.run([train_step,loss,merged,cos_sim],feed_dict=feed_dict)
    if i % int(1e2) == 0:
        print(i,'loss: ',mb_loss,'time: ',time.time()-cur_time)
        cur_time = time.time()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        writer.add_run_metadata(run_metadata, 'step%d' % i)
    writer.add_summary(summary,i)
    '''
