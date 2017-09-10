# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 07:10:14 2017

@author: User
"""
#%%

import os
import os.path
import math

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

import cifar10_input

#%%

BATCH_SIZE = 128
learning_rate = 0.05
MAX_STEP = 1000
Train = True

img_width = 32
img_height = 32
img_depth = 3
img_pixel = img_width * img_height * img_depth
label_bytes = 1
image_bytes = img_width * img_height * img_depth

#%%

def inference(images):
   
    #conv1, [5, 5, 3, 96], The first two dimensions are the patch size,
    #the next is the number of input channels, 
    #the last is the number of output channels
    
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [3, 3, 3, 96],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32)) 
        biases = tf.get_variable('biases', 
                                 shape=[96],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    
    
    #pool1 and norm1   
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')
    
    
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,96, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    
    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')
    
    
    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    
    
    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[384,192],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[192],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
     
        
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[192, 10],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[10],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    
    return softmax_linear

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.int64)
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='xentropy_per_example')
        
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss

def train():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    data_dir = 'G:/Git/CIFAR10/data/cifar-10-batches-bin/'
    log_dir = 'G:/Git/CIFAR10/logs/'
    
    images, labels = cifar10_input.read_cifar10(data_dir=data_dir,
                                                is_train=True,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
    
    logits = inference(images)
    loss = losses(logits, labels)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            
            _, loss_value = sess.run([train_op, loss])
            
            if step % 50 == 0:
                print('Step: %d , loss: %.4f' % (step, loss_value))
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        coord.request_stop()
    
    coord.join(threads)
    sess.close()

#%%
def evaluation():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    data_dir = 'G:/Git/CIFAR10/data/cifar-10-batches-bin/'
    log_dir = 'G:/Git/CIFAR10/logs/'
    
    images, labels = cifar10_input.read_cifar10(data_dir=data_dir,
                                                is_train=True,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
    
    with tf.Graph().as_default():
       
        
        logit = inference(images)
        loss = losses(logit, labels)
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        
        x = tf.placeholder(tf.float32, shape=[img_width, img_height, img_depth])
        
        saver = tf.train.Saver(tf.global_variables())
       
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        i = 0
        
        with tf.Session() as sess:
            try:
                while not coord.should_stop() and i < 1:    
                    print("Reading......")
                    ckpt = tf.train.get_checkpoint_state(log_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print("Loading sucess, global_step is %s" % global_step)
                    else:
                        print("No checkpoint file found")
                    
                    img, prediction = sess.run([images, labels], feed_dict={x:images})
                    for j in np.arange(2):
                        max_index = np.argmax(prediction)
                        print("Max_Index: %s" % max_index)
                        plt.imshow(img[j,:,:,:])
                        plt.show()
                    i+=1
            except tf.errors.OutOfRangeError:
                print("done")
            finally:
                coord.request_stop()
            coord.join(threads)
        
        
