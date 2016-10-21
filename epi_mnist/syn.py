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
def make_syn_net(inputs,tie_weights=False):
    syn_hid1 = linear(tf.concat(1,inputs),hid_dim,'syn_hid1',tf.nn.relu,tied=tie_weights)
    syn_out = linear(syn_hid1,embed_dim,'syn_out',bias_value=0.0,init=tf.constant_initializer(),tied=tie_weights)
    #syn_out = linear(tf.concat(1,inputs),embed_dim,'syn_out',bias_value=0.0,init=tf.constant_initializer(),tied=tie_weights)
    return syn_out

#encoder
x_ = tf.placeholder(tf.float32,shape=[1,in_dim])
hid1 = linear(x_,hid_dim,'hid1',tf.nn.relu)
embed = linear(hid1,embed_dim,'embed')
'''
#encode to store
hid1_store = linear(x_,hid_dim,'hid1_store',tf.nn.relu)
embed_store = linear(hid1,embed_dim,'embed_store')
'''

#memories
mem_embed_ = tf.placeholder(tf.float32,shape=[n_mem,embed_dim])
mem_y_ = tf.placeholder(tf.float32,shape=[n_mem,out_dim])
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

#sythetic gradients
''' full context syn net
mem_embed_blob = tf.reshape(mem_embed_,[1,-1])
mem_y_blob = tf.reshape(mem_y_,[1,-1])
syn_out = tf.stop_gradient(make_syn_net([mem_embed_blob,mem_y_blob,embed,y_]))
'''
syn_out = tf.stop_gradient(make_syn_net([embed,y_]))
syn_loss = tf.squeeze(tf.matmul(embed,tf.transpose(syn_out)))

'''synthetic gradient gradients
super inefficient right now. Shouldn't need to recompute syn grads for 
everything in memory, should be storing this state instead
'''
'''full context syn net
syn_out_copy = make_syn_net([tf.tile(mem_embed_blob,[n_mem,1]),tf.tile(mem_y_blob,[n_mem,1]),mem_embed_,mem_y_],True)
mem_embed_grad = tf.gradients(class_loss,mem_embed_)[0]
'''
syn_out_copy = make_syn_net([mem_embed_,mem_y_],True)
mem_embed_grad = tf.gradients(class_loss,mem_embed_)[0]
grad_loss = tf.reduce_mean(tf.reduce_sum(tf.square(syn_out_copy
    -tf.stop_gradient(mem_embed_grad)),1))

#loss = class_loss
#loss = grad_loss + class_loss + syn_loss
loss = grad_loss + syn_loss
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

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
for i in range(int(1e5)):
    cur_input,cur_output = mnist.train.next_batch(1)
    _,*cur_loss,cur_acc,cur_embed = sess.run([train_step,class_loss,syn_loss,grad_loss,acc,embed],feed_dict={x_:cur_input,y_:cur_output,mem_embed_:M['embed'],mem_y_:M['label']})
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
        #plt.plot(time_list,np.log(np.asarray(acc_hist)),time_list,np.log(np.asarray(loss_hist)))
        plt.plot(time_list,np.asarray(acc_hist),time_list,np.asarray(loss_hist)/max(np.asarray(loss_hist)))
        plt.pause(.1)
        print(i+1,*(cumloss),cumacc)
        #cumloss[:] = 0
        #cumacc = 0.0
