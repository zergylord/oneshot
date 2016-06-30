import tensorflow as tf
import numpy as np
import time
mb_dim = 32
mem_dim = int(1e2)
enc_dim = 100
x_hat_encode = tf.ones(shape=[mb_dim,enc_dim])
x_i_encode = tf.get_variable('i',shape=[mb_dim,mem_dim,enc_dim])
x_hat_inv_mag = tf.rsqrt(tf.reduce_sum(tf.square(x_hat_encode),1))
x_i_inv_mag = tf.rsqrt(tf.reduce_sum(tf.square(x_i_encode),2))
cos_sim = tf.squeeze(tf.batch_matmul(x_i_encode,tf.expand_dims(x_hat_encode,-1)))*tf.expand_dims(x_hat_inv_mag,-1)*x_i_inv_mag
net_cost = -tf.reduce_mean(tf.reduce_sum(cos_sim,1))
train_step = tf.train.AdamOptimizer().minimize(net_cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(1000):
    cur_time = time.clock()
    _,loss = sess.run([train_step,net_cost])
    print(loss,time.clock()-cur_time)
print(sess.run(cos_sim))
#output = sess.run(cos_sim)
#print(time.clock()-cur_time,output.shape)
