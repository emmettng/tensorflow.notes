# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def softMaxNet(x, y_, inDim,outDim):
    W1 = tf.get_variable("W1",[inDim,outDim],dtype = tf.float32)
    b1 = tf.get_variable("b1",[outDim], dtype = tf.float32)
    y = tf.nn.softmax(tf.matmul(x,W1) + b1)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    return y,cross_entropy

def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print ("")
    inDim = len(mnist.train.images[0])
    outDim = len(mnist.train.labels[0])

    mg = tf.Graph()

    with mg.as_default():
        x = tf.placeholder(tf.float32,shape=[None,784])
        y_ = tf.placeholder(tf.float32,shape= [None,10])
        y, loss = softMaxNet(x,y_,inDim,outDim)
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        train_softMax= optimizer.minimize(loss)
        init = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session(graph=mg)
    sess.run(init)
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_softMax, feed_dict={x: batch_xs, y_: batch_ys})

    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print (acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
