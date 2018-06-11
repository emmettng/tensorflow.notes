import tensorflow as tf
import pandas as pd
import numpy as np

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def weightVariable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def pool2d(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')
    ## what is the SAME means here ? ?
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

    with tf.name_scope('conv1'):
        W_conv1 = weightVariable([5,5,1,32])
        b_conv1 = biasVariable([32])
        h_conv1 = tf.nn.relu(conv2d(x_2d_images,W_conv1)+b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = pool2d(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weightVariable([5,5,32,64])
        b_conv2 = biasVariable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = pool2d(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weightVariable([7*7*64,1024])
        b_fc1 = biasVariable([1024])

        h_pool2_flatten = tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten,W_fc1)+b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)  ## placeholder must be passed or returned
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weightVariable([1024,10])
        b_fc2 = biasVariable([10])
        y_conv = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

    return y_conv,keep_prob


def def_LOSS(y_,y):
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
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

        ## regiester all variable for initilization
        graph_varialbe_init= tf.global_variables_initializer()

    ## tensorboard
    tensorboard_writer= tf.summary.FileWriter("./event_log")
    tensorboard_writer.add_graph(mg)
    tensorboard_writer.close()

    ## start training
    sess = tf.Session(graph = mg)
    sess.run(graph_varialbe_init)

    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        ## need to specify Session instance explicitly
        training_handle.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5},session = sess)
        if i %100 == 0:
            training_accuracy = accuracy.eval(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1},session=sess)
            print ("step %d, training accuracy is: %g" % (i,training_accuracy))
    print ("")
    test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0},session=sess)
    print ("Training finished!")
    print ("test accuracy is %g" % test_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
