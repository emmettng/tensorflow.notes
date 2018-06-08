from math import sqrt

import numpy as np
import pandas as pd
import tensorflow as tf

def myNumpyFunc(nx):
    def myFunc (x):
        y = 0.2*x - 0.4*x**2 - x - 1
        return -y
    lx = list(nx)
    ly = list(map(myFunc, lx))
    ny = np.asarray(ly)
    return ny

def normalize(X):
    nx = list(X.reshape(-1))
    gap = np.max(nx) - np.min(nx)
    mx = np.asarray(list(map(lambda t: t/gap, nx)))
    rX = mx.reshape(N,1)
    return rX

def getXY(mi, ma, num, norm=False):
    nx = np.arange(mi,ma,(ma-mi)/num)
    ny = myNumpyFunc(nx)
    X = nx.reshape(num,1)
    Y = ny.reshape(num,1)
    if not norm:
        return (X,Y)
    X = normalize(X)
    Y = normalize(Y)
    return (X,Y)

def oneHiddenLayerNet(hiddenDimension, activeFunc,x,Y):
    W1 = tf.get_variable("W1",[1,hiddenDimension], dtype = tf.float32)
    b1 = tf.get_variable('b1',[hiddenDimension], dtype = tf.float32)
    hidden_1= activeFunc(tf.add(tf.matmul(x, W1), b1))   #standard
    W2 = tf.get_variable('W2',[hiddenDimension,1], dtype = tf.float32)
    b2 = tf.get_variable('b',[1], dtype = tf.float32)
    output = tf.add(tf.matmul(hidden_1,W2), b2)
    loss = tf.losses.mean_squared_error(labels=Y,predictions=output)
    print (x.name)
    return loss

def main():
    (X,Y) = getXY(0,10,50)
    testGraph = tf.Graph()
    testGraph2 = tf.Graph()

    with testGraph2.as_default():
        x = tf.placeholder(tf.float32, shape=[None,1])
        testGloss = oneHiddenLayerNet(10,tf.nn.leaky_relu,x,Y)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_relu = optimizer.minimize(testGloss)
        init = tf.global_variables_initializer()

    with testGraph.as_default():
        x = tf.placeholder(tf.float32, shape=[None,1])
        testGloss = oneHiddenLayerNet(10,tf.nn.leaky_relu,x,Y)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_relu = optimizer.minimize(testGloss)
        init = tf.global_variables_initializer()



    writer = tf.summary.FileWriter('./event_log')
    writer.add_graph(testGraph)
    writer.close()

    sess = tf.Session(graph=testGraph)
    sess.run(init)
    cnt = 0
    for i in range(50000):
        _,l_relu = sess.run((train_relu,testGloss),{x:X})
        cnt += 1
        if cnt % 5000 == 0:
            print ("in loop: " +str(i))
            print ("relu: "+ str(l_relu))
            print ("")

if __name__ == "__main__":
    main()
