import numpy as np
import os 
import time 
import tqdm
import tensorflow as tf
import config as cfg
slim = tf.contrib.slim
###
##  Construct the capsule network
### 
def CheckData(X):
    if np.isnan(X).any():
        X[np.isnan(X)] = np.nanmean(X)
    return X
config = cfg.Config90()
x = tf.placeholder(tf.float32,shape=[None,1,90,3])
y = tf.placeholder(tf.float32,shape=[None,config.num_label])

###################
###  simple network ######
###################
def depth_convolution(x,channels,kernel,strides):
    x1 = slim.conv2d(x,channels[0],1,stride=1,padding="VALID",activation_fn=tf.nn.relu,weights_initializer=tf.random_uniform_initializer(maxval=0.01,minval=0.001) )
    x2 = slim.conv2d(x1,channels[1],kernel,stride=strides,padding="VALID",activation_fn=tf.nn.relu,weights_initializer=tf.random_uniform_initializer(maxval=0.01,minval=0.001))
    return x2
def Totalcount():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
        # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print("The Total params:",total_parameters)
top1 = depth_convolution(x,[16,32],[1,9],1)
top1 = tf.nn.dropout(top1,0.5)
#top2 = depth_convolution(top1,[32,64],[1,9],1)
top2 = slim.max_pool2d(top1,kernel_size=[1,9], stride=2,padding="VALID")
top2 = tf.nn.dropout(top2,0.5)
v1 = slim.flatten(top2)
v2 =  slim.fully_connected(v1,256,activation_fn=tf.nn.relu,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
v2 = tf.nn.dropout(v2,0.5)
v3 = slim.fully_connected(v2,config.num_label, activation_fn=None)
loss =tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=v3))

train_op = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(v3,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_x = CheckData(np.load(cfg.save_train_90_size))
train_y = np.load(cfg.save_train_label_90).astype(np.float32)
test_x = CheckData(np.load(cfg.save_test_90_size))
test_y = np.load(cfg.save_test_label_90).astype(np.float32)
batch_size = config.batch_size
total_train_batch = train_x.shape[0] // batch_size
total_test_batch = test_x.shape[0] // batch_size
epoch = 120
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver2 = tf.train.Saver(tf.global_variables())
Totalcount()
last = 0.0
for i in range(epoch):
        for b in range(total_train_batch):
            offset = (b * batch_size) % (train_y.shape[0] -batch_size)
            batch_x = train_x[offset:(offset+batch_size),:,:,:]
            batch_y = train_y[offset:(offset+batch_size),:]
            _ , losses = sess.run( [train_op,loss],feed_dict={x:batch_x,y:batch_y} )
        if i % 2 == 0:
            acc = sess.run(accuracy,feed_dict={x:test_x,y:test_y})
            print(acc,losses)         
            if acc > 0.94 and acc > last :
                last = acc 
                saver2.save(sess,"./mobile_data/model.ckpt")
sess.close()       

