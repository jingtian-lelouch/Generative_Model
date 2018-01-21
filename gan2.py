# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:18:28 2018

@author: xiong
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
############Load Data
mnist = input_data.read_data_sets("MNIST_data/")
x_train = mnist.train.images[:55000,:]

##### Hyperparameters
z_dimensions = 100
batch_size = 16

########Layers
def conv2d(x, W):
  return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def discriminator(x_image):

    #First Conv and Pool Layers
    with tf.variable_scope("hidden_layer1"):
        W_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = avg_pool_2x2(h_conv1)

    #Second Conv and Pool Layers
    with tf.variable_scope("hidden_layer2"):
        W_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = avg_pool_2x2(h_conv2)

    #First Fully Connected Layer
    with tf.variable_scope("hidden_layer3"):
        W_fc1 = tf.get_variable('d_wfc1', [7 * 7 * 16, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #Second Fully Connected Layer
    with tf.variable_scope("hidden_layer4"):
        W_fc2 = tf.get_variable('d_wfc2', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))
        y_conv=(tf.matmul(h_fc1, W_fc2) + b_fc2)
    return y_conv

def generator(z, batch_size, z_dim):
    g_dim = 64 #Number of filters of first layer of generator 
    c_dim = 1 #Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
    s = 28 #Output size of the image
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16) #We want to slowly upscale the image, so these values will help
                                                              #make that change gradual.
    with tf.variable_scope("hidden_layer0"):
        h0 = tf.reshape(z, [batch_size, s16+1, s16+1, 25])
        h0 = tf.nn.relu(h0)
    #Dimensions of h0 = batch_size x 2 x 2 x 25

    #First DeConv Layer
    with tf.variable_scope("hidden_layer1"):
        output1_shape = [batch_size, s8, s8, g_dim*4]
        W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, 
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv1
        H_conv1 = tf.layers.batch_normalization(inputs = H_conv1, center=True, scale=True, training=phase)
        H_conv1 = tf.nn.relu(H_conv1)
    #Dimensions of H_conv1 = batch_size x 3 x 3 x 256

    #Second DeConv Layer
    with tf.variable_scope("hidden_layer2"):
        output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim*2]
        W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, 
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv2
        H_conv2 = tf.layers.batch_normalization(inputs = H_conv2, center=True, scale=True, training=phase)
        H_conv2 = tf.nn.relu(H_conv2)
    #Dimensions of H_conv2 = batch_size x 6 x 6 x 128

    #Third DeConv Layer
    with tf.variable_scope("hidden_layer3"):  
        output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim*1]
        W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, 
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv3
        H_conv3 = tf.layers.batch_normalization(inputs = H_conv3, center=True, scale=True, training=phase)
        H_conv3 = tf.nn.relu(H_conv3)
    #Dimensions of H_conv3 = batch_size x 12 x 12 x 64

    #Fourth DeConv Layer
    with tf.variable_scope("hidden_layer4"):
        output4_shape = [batch_size, s, s, c_dim]
        W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, 
                                         strides=[1, 2, 2, 1], padding='VALID') + b_conv4
        H_conv4 = tf.nn.tanh(H_conv4,name="x_artificial")
    #Dimensions of H_conv4 = batch_size x 28 x 28 x 1

    return H_conv4




#########Build model
with tf.variable_scope('Input'):
    x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1],name="x_real") #Placeholder for input images to the discriminator
    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions],name="z") #Placeholder for input noise vectors to the generator
    phase = tf.placeholder(tf.bool, shape=[], name='phase')
    
    
with tf.variable_scope('Generator'):
    Gz = generator(z_placeholder, batch_size, z_dimensions)
    
with tf.variable_scope('Discriminator') as scope:    
    Dx = discriminator(x_placeholder) 
    scope.reuse_variables()
    Dg = discriminator(Gz) 

with tf.variable_scope('loss'):
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))
    
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
    d_loss = d_loss_real + d_loss_fake
    tf.summary.scalar('g_loss',g_loss)
    tf.summary.scalar('d_loss',d_loss)
    
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

with tf.variable_scope("training"): 
    trainerD = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
    trainerG = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

#######Training
#if not os.path.exists('out/'):
#    os.makedirs('out/')
    
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(tf.global_variables_initializer())

iterations = 3000
for i in range(iterations):
    z_batch = np.random.normal(-1, 1, size=[batch_size, z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)
    real_image_batch = np.reshape(real_image_batch[0],[batch_size,28,28,1])
    
    with tf.control_dependencies(update_ops):
        _,dLoss = sess.run([trainerD, d_loss],feed_dict={z_placeholder:z_batch,x_placeholder:real_image_batch,phase:1}) #Update the discriminator
        _,gLoss = sess.run([trainerG,g_loss],feed_dict={z_placeholder:z_batch,phase:1}) #Update the generator
    if i%100==0:
        print('Iter: {}'.format(i))
        print('D loss: {:.4}'.format(dLoss))
        print('G_loss: {:.4}'.format(gLoss))
        print()
        G_samples = sess.run(Gz,feed_dict={z_placeholder:z_batch,phase:1})
        tf.summary.image('Generated_samples',G_samples,max_outputs=16)
        merged=tf.summary.merge_all()
        rs = sess.run(merged,feed_dict={z_placeholder:z_batch,x_placeholder:real_image_batch,phase:1})
        writer.add_summary(rs,i)