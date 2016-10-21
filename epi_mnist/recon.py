'''
predict a MNIST digit by comparing it to the most recent n_mem digits.
Each digit is only encoded once. Synthetic gradients are used to update
based on the predicted effect of future comparisons.

A non-parametric algorithm with O(1) time complexity, and O(k*n) space complexity,
where k is the embedding dimensionality, and n is n_mem.

non-minibatch
'''
import tensorflow as tf
import numpy as np
from ops import *
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
plt.ion()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

in_dim = 784
hid_dim = 256
embed_dim = 64
out_dim = 10
n_mem = 128
eps = 1e-10

def make_encoder(inp,scope,tied=False):
    with tf.variable_scope('classifier'):
        with tf.variable_scope(scope,reuse=tied):
            hid1 = linear(inp,hid_dim,'hid1',tf.nn.relu)
            hid2 = linear(hid1,embed_dim,'hid2')
    return hid2

x_ = tf.placeholder(tf.float32,shape=[1,in_dim])
#memories
mem_embed_ = tf.placeholder(tf.float32,shape=[n_mem,embed_dim])
mem_y_ = tf.placeholder(tf.float32,shape=[n_mem,out_dim])

'''
encode->decode x_hat for z and recon_loss
'''
#encoder
embed = make_encoder(x_,'encoder')
#decoder
with tf.variable_scope('helper'):
    hid3 = linear(embed,hid_dim,'hid3',tf.nn.relu)
    recon = linear(hid3,in_dim,'recon',tf.nn.sigmoid)
helper_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='helper')
recon_loss = tf.reduce_mean(-tf.reduce_sum(x_*tf.log(tf.clip_by_value(recon,1e-10,1))+(1-x_)*tf.log(tf.clip_by_value(1-recon,1e-10,1)),1))
'''
decode->encode mem_embed_ for passing class_loss gradient
'''
with tf.variable_scope('helper'):
    hid3 = linear(mem_embed_,hid_dim,'hid3',tf.nn.relu,tied=True)
    mem_recon = linear(hid3,in_dim,'recon',tf.nn.sigmoid,tied=True)
fake_z = make_encoder(mem_recon,'encoder',True)
classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='classifier')
'''
#encode to store
hid1_store = linear(x_,hid_dim,'hid1_store',tf.nn.relu)
embed_store = linear(hid1,embed_dim,'embed_store')
'''

mem_inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(mem_embed_),1,keep_dims=True),eps,float("inf")))
#inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(embed)),eps,float("inf")))
#cos_sim comparison
cos_sim = tf.transpose(tf.matmul(mem_embed_,tf.transpose(embed))*mem_inv_mag)
weighting = tf.nn.softmax(cos_sim) #a [1,n_mem] shaped tensor
label_prob = tf.squeeze(tf.matmul(weighting,mem_y_))
#label_prob = tf.Print(label_prob,[label_prob,cos_sim])
#supervised loss
y_ = tf.placeholder(tf.float32,shape=[1,out_dim])
acc = tf.to_float(tf.nn.in_top_k(tf.expand_dims(label_prob,0),tf.arg_max(y_,1),1))[0]
class_loss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(label_prob,eps,1)))

'''
crazy stuff
'''

z_grad = tf.gradients(class_loss,mem_embed_)[0]
syn_loss = tf.reduce_mean(tf.batch_matmul(tf.expand_dims(fake_z,1),tf.expand_dims(tf.stop_gradient(z_grad),2)))
loss = syn_loss
optim = tf.train.AdamOptimizer(1e-4)
train_step = optim.minimize(loss,var_list=classifier_vars)
helper_optim = tf.train.AdamOptimizer(1e-4)
helper_train_step = optim.minimize(recon_loss,var_list=helper_vars)


sess = tf.Session()
sess.run(tf.initialize_all_variables())
refresh = int(1e3)
cumacc = 0.0
cumloss = np.zeros((3,))
cur_los = np.zeros((3,))
from collections import deque
M = {'embed': deque(np.zeros((n_mem,embed_dim))),'label': deque(np.zeros((n_mem,out_dim)))}
for i in range(n_mem):
    cur_input,cur_output = mnist.train.next_batch(1)
    M['embed'].popleft()
    M['embed'].append(sess.run(embed,feed_dict={x_:cur_input})[0])
    M['label'].popleft()
    M['label'].append(np.copy(cur_output)[0])
acc_hist = []
loss_hist = []
rho = .999
for i in range(int(1e7)):
    cur_input,cur_output = mnist.train.next_batch(1)
    _,_,*cur_loss,cur_acc,cur_embed = sess.run([train_step,helper_train_step,class_loss,syn_loss,recon_loss,acc,embed],feed_dict={x_:cur_input,y_:cur_output,mem_embed_:M['embed'],mem_y_:M['label']})
    #_,*cur_loss,cur_acc,cur_embed = sess.run([train_step,class_loss,syn_loss,recon_loss,acc,embed],feed_dict={x_:cur_input,y_:cur_output,mem_embed_:M['embed'],mem_y_:M['label']})
    M['embed'].popleft()
    M['embed'].append(cur_embed[0])
    M['label'].popleft()
    M['label'].append(np.copy(cur_output)[0])
    cumloss*=rho
    cumacc*=rho
    cumloss+=np.asarray(cur_loss)*(1-rho)
    cumacc+=cur_acc*(1-rho)
    if (i+1) % refresh == 0: 
        acc_hist.append(cumacc)
        loss_hist.append(cumloss[0])
        plt.clf()
        #plt.ylim((0,1000))
        time_list = list(range(len(acc_hist)))
        plt.plot(time_list,np.asarray(acc_hist),time_list,np.asarray(loss_hist)/max(np.asarray(loss_hist)))
        plt.pause(.1)
        print(i+1,*(cumloss),cumacc)
        #cumloss[:] = 0
        #cumacc = 0.0
