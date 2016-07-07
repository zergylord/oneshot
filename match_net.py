import numpy as np
import time
cur_time = time.time()
mb_dim = 32
x_dim = 28
y_dim = 5 #5 possible classes
hid_dim = 200
enc_dim = 100
n_samples_per_class = 5 #5 samples of each class
n_samples = y_dim*n_samples_per_class

data = np.load('data.npy')
data = np.reshape(data,[-1,20,28,28])

def get_minibatch():
    mb_x_i = np.zeros((mb_dim,n_samples,x_dim,x_dim,1))
    mb_y_i = np.zeros((mb_dim,n_samples))
    mb_x_hat = np.zeros((mb_dim,x_dim,x_dim,1))
    mb_y_hat = np.zeros((mb_dim,))
    for i in range(mb_dim):
        ind = 0
        pinds = np.random.permutation(n_samples)
        classes = np.random.choice(data.shape[0],y_dim,False)
        x_hat_class = np.random.randint(y_dim)
        for j,cur_class in enumerate(classes): #each class
            example_inds = np.random.choice(data.shape[1],n_samples_per_class,False)
            for eind in example_inds:
                mb_x_i[i,pinds[ind],:,:,0] = data[cur_class][eind]
                mb_y_i[i,pinds[ind]] = j
                ind +=1
            if j == x_hat_class:
                mb_x_hat[i,:,:,0] = data[cur_class][np.random.choice(data.shape[1])]
                mb_y_hat[i] = j
    return mb_x_i,mb_y_i,mb_x_hat,mb_y_hat



                



import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/oneshot_logs', 'Summaries directory')
if tf.gfile.Exists(FLAGS.summary_dir):
    tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)


x_hat = tf.placeholder(tf.float32,shape=[None,x_dim,x_dim,1])
x_i = tf.placeholder(tf.float32,shape=[None,n_samples,x_dim,x_dim,1])
y_i_ind = tf.placeholder(tf.int32,shape=[None,n_samples])
y_i = tf.one_hot(y_i_ind,y_dim)
y_hat_ind = tf.placeholder(tf.int32,shape=[None])
y_hat = tf.one_hot(y_hat_ind,y_dim)
with tf.variable_scope('encode_x_hat'):
    cur_input = x_hat
    cur_filters = 1
    for i in range(4):
        with tf.variable_scope('conv'+str(i)):
            W = tf.get_variable('W',[3,3,cur_filters,64])
            cur_filters = 64
            conv = tf.nn.relu(tf.nn.conv2d(cur_input,W,strides=[1,1,1,1],padding='SAME'))
            cur_input = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    x_hat_encode = tf.squeeze(cur_input,[1,2])
x_hat_inv_mag = tf.rsqrt(tf.reduce_sum(tf.square(x_hat_encode),1,keep_dims=True))
cos_sim_list = []
with tf.variable_scope('encode_x_i') as varscope:
    for i in range(n_samples):
        if i > 0: varscope.reuse_variables()
        cur_input = x_hat
        cur_filters = 1
        for i in range(4):
            with tf.variable_scope('conv'+str(i)):
                W = tf.get_variable('W',[3,3,cur_filters,64])
                cur_filters = 64
                conv = tf.nn.relu(tf.nn.conv2d(cur_input,W,strides=[1,1,1,1],padding='SAME'))
                cur_input = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        x_i_encode = tf.squeeze(cur_input,[1,2])
        x_i_inv_mag = tf.rsqrt(tf.reduce_sum(tf.square(x_i_encode),1,keep_dims=True))
        dotted = tf.squeeze(
            tf.batch_matmul(tf.expand_dims(x_hat_encode,1),tf.expand_dims(x_i_encode,2)),[1,])
        cos_sim_list.append(dotted
                *x_hat_inv_mag
                *x_i_inv_mag)
cos_sim = tf.concat(1,cos_sim_list)
tf.histogram_summary('cos sim',cos_sim)
weighting = tf.nn.softmax(cos_sim)
label_prob = tf.squeeze(tf.batch_matmul(tf.expand_dims(weighting,1),y_i))
tf.histogram_summary('label prob',label_prob)

eps = 0
loss = tf.reduce_mean(-tf.reduce_sum(tf.log(eps+tf.gather(label_prob,y_hat_ind)),1))
tf.scalar_summary('loss',loss)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
optim = tf.train.AdamOptimizer(1e-2)
grads = optim.compute_gradients(loss)
grad_summaries = [tf.histogram_summary(v.name,g) for g,v in grads]
train_step = optim.apply_gradients(grads)

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter(FLAGS.summary_dir,sess.graph)
sess.run(tf.initialize_all_variables())
for i in range(int(1e5)):
    mb_x_i,mb_y_i,mb_x_hat,mb_y_hat = get_minibatch()
    feed_dict = {x_hat: mb_x_hat,
                y_hat_ind: mb_y_hat,
                x_i: mb_x_i,
                y_i_ind: mb_y_i}
    _,mb_loss,summary = sess.run([train_step,loss,merged],feed_dict=feed_dict)
    if i % int(1e2) == 0:
        print(i,mb_loss,'time: ',time.time()-cur_time)
        cur_time = time.time()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        writer.add_run_metadata(run_metadata, 'step%d' % i)
    writer.add_summary(summary,i)




