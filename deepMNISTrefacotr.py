'''
It is necessary to rewrite all summary part into functional style.
check haskell api for more applicable solutions.
'''
import tensorflow as tf
import pandas as pd
import numpy as np

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from dataset import *

FLAGS = None

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def pool2d(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')
    ## what is the SAME means here ? ?

def trans_active(input,kernel_shape,bias_shape,kernel_op,active_op):
    kernel_init = tf.truncated_normal(kernel_shape,stddev=0.1) ## initial value ? 
    W = tf.get_variable('W',initializer=kernel_init)
    bias_init = tf.constant(0.1,shape=bias_shape)
    b = tf.get_variable('b',initializer = bias_init)

    trans = kernel_op(input,W)
    h = active_op(trans + b)

    with tf.name_scope('weights'):
        variable_summaries(W)
    with tf.name_scope('bias'):
        variable_summaries(b)
    with tf.name_scope('trans'):
        tf.summary.histogram('histogram',trans)
    with tf.name_scope('active'):
        tf.summary.histogram('histogram',h)
    return h

def variable_summaries(var,):
    with tf.name_scope('scalar_summaries'):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

'''
 def_GRAPH
 Input :
    Compose the computational graph, and the only input is 'x'

 Return:
    The composition of series of function, could be evaluated by a session or as an first order function feed to other functions, such as def_LOSS.
'''
def def_GRAPH(x):
    ## reshape the data from 1d to 2d
    with tf.name_scope('reshape'):
        x_2d_images = tf.reshape(x,[-1,28,28,1])
        tf.summary.image('input',x_2d_images,10)

    with tf.variable_scope('conv1'):
        h_conv1 = trans_active(x_2d_images,[5,5,1,32],[32],conv2d,tf.nn.relu)
        h_pool1 = pool2d(h_conv1)

    with tf.variable_scope('conv2'):
        h_conv2 = trans_active(h_pool1,[5,5,32,64],[64],conv2d,tf.nn.relu)
        h_pool2 = pool2d(h_conv2)

    with tf.variable_scope('fc1'):
        h_pool2_flatten = tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1 = trans_active(h_pool2_flatten,[7*7*64,1024],[1024],tf.matmul,tf.nn.relu)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)  ## placeholder must be passed or returned
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

        tf.summary.scalar('dropout_keep_probability',keep_prob)

    with tf.variable_scope('fc2'):
        y_conv = trans_active(h_fc1_drop,[1024,10],[10],tf.matmul,tf.nn.relu)

    return y_conv,keep_prob


def def_LOSS(y_,y):
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
        tf.summary.scalar('cross_entropy',tf.reduce_mean(cross_entropy))
    return cross_entropy

def main(_):

    ## data input part , this part better be a generator
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    ## define the Computational Graph
    mg = tf.Graph()

    with mg.as_default():
        x = tf.placeholder(dtype=tf.float32, shape = [None,784])
        y_ = tf.placeholder(dtype=tf.float32, shape = [None,10])

        y_Hypo, keep_prob = def_GRAPH(x)

        ## define optimizer
        optimizer = tf.train.AdamOptimizer(1e-4)

        ## get loss definition
        loss = def_LOSS(y_,y_Hypo)

        ## define training_handle
        training_handle = optimizer.minimize(loss)

        ## accuracy definition
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_Hypo,1),tf.argmax(y_,1))
            correct_prediction = tf.cast(correct_prediction,tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
            tf.summary.scalar('accuracy',accuracy)
        ## regiester all variable for initilization
        graph_varialbe_init= tf.global_variables_initializer()

        for v in tf.global_variables():
            print (v)
            print (v.name)

        merged_summary = tf.summary.merge_all()

    ## tensorboard
##    merged_summary = tf.summary.merge_all()
    training_summary= tf.summary.FileWriter(FLAGS.log_dir + '/train', mg)
    testing_summary = tf.summary.FileWriter(FLAGS.log_dir + '/test')

##    tensorboard_writer.add_graph(mg)
##    tensorboard_writer.close()

    ## start training
    sess = tf.Session(graph = mg)
    sess.run(graph_varialbe_init)

    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        ## need to specify Session instance explicitly
        training_handle.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5},session = sess)

        summary = sess.run(merged_summary,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
        training_summary.add_summary(summary)
        if i %100 == 0:
            training_accuracy = accuracy.eval(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1},session=sess)

            summary = merged_summary.eval(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1},session=sess)
            testing_summary.add_summary(summary)
            print ("step %d, training accuracy is: %g" % (i,training_accuracy))

    print ("")
    test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0},session=sess)
    print ("Training finished!")
    print ("test accuracy is %g" % test_accuracy)

    print (type(accuracy))
    print (type(training_handle))

    sess.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## data source config
    parser.add_argument(
      '--data_dir',
      type=str,
      default='./mnist',
      help='Directory for storing input data')

    ## log dir config
    parser.add_argument(
      '--log_dir',
      type=str,
      default = './event_log',
      help='Summaries log directory')


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
