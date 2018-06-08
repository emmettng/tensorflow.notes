from math import sqrt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

######################## prepare the data ########################
x = tf.placeholder(tf.float32, shape= [None,3])

W1 = tf.Variable(tf.truncated_normal([3, 10], stddev=0.03), name='W1')
b1 = tf.Variable(tf.truncated_normal([10]), name='b1')

# hidden layer 1-to-output
W2 = tf.Variable(tf.truncated_normal([10, 3], stddev=0.03), name='W2')
b2 = tf.Variable(tf.truncated_normal([3]), name='b2')

######################## Activations, outputs ######################
# output hidden layer 1
lm = tf.matmul(x,W1)
print (lm.name)
la1 = tf.add(lm,b1,name='a1')
with tf.name_scope("ns1"):
    la = tf.add(lm,b1,name='a1')
print (la.name)

hidden_out = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))   #standard

# total output
y_ = tf.nn.relu(tf.add(tf.matmul(hidden_out, W2), b2))

print (W1.name)
