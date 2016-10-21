import numpy as np
import tensorflow as tf
def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    #print('You have opted to use the orthogonal_initializer function')
    def _initializer(shape, dtype=tf.float32,partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape) #this needs to be corrected to float32
        #print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer
eps = 1e-10
def compute_return(rewards,gamma):
    length = len(rewards)
    R = np.zeros((length,))
    for t in reversed(range(length)):
        R[:t+1] = R[:t+1]*gamma + rewards[t]
    return list(R)
import tensorflow as tf
def linear(in_,out_dim,name,activation_fn=None,bias=True,bias_value=None,init=orthogonal_initializer(),tied=False):
    in_dim = in_.get_shape()[1]
    with tf.variable_scope(name,reuse=tied):
        W = tf.get_variable('W',[in_dim,out_dim],tf.float32,initializer=init)
        #W = tf.random_uniform([in_dim,out_dim], minval=-0.05, maxval=0.05)
        out = tf.matmul(in_,W)
        if bias:
            b = tf.get_variable('b',[out_dim],tf.float32)
            #b = tf.constant(0.01, shape=[out_dim])
            if bias_value != None:
                print('manual bias')
                b = b.assign(tf.constant(bias_value,shape=[out_dim]))
            out = out + b
        if activation_fn != None:
            '''
            mean,variance = tf.nn.moments(out,[0,1])
            beta = tf.Variable(tf.constant(0.0,shape=[out_dim]))
            gamma = tf.Variable(tf.constant(1.0,shape=[out_dim]))
            out = tf.nn.batch_normalization(out,mean,variance,beta,gamma,eps)
            '''
            out = activation_fn(out)
    return out
