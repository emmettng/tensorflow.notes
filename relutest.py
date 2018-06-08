from math import sqrt
import numpy as np
import pandas as pd
import tensorflow as tf

################## target function domain and sample size ####################
N = 100
MIN = 0
MAX = 10

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
## Both native data format and numpy are all avaliable
##nx = list(range(10))
##X = list(map(lambda x: [x],nx))
nx = np.arange(MIN,MAX,(MAX-MIN)/N)
ny = myNumpyFunc(nx)

print (nx)
print (ny)

X = nx.reshape(N,1)
Y = ny.reshape(N,1)
print (X)
print (Y)
print ("after normailze")

##X = normalize(X)
##Y = normalize(Y)
##print (X)
##print (Y)

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None,1])
D1 = 10
W1 = tf.get_variable('W1',[1,D1], dtype = tf.float32)
b1 = tf.get_variable('b1',[10], dtype = tf.float32)
hidden_1= tf.nn.leaky_relu(tf.add(tf.matmul(x, W1), b1))   #standard

W2 = tf.get_variable('W2',[10,1], dtype = tf.float32)
b2 = tf.get_variable('b',[1], dtype = tf.float32)
output = tf.add(tf.matmul(hidden_1,W2), b2)

loss_relu = tf.losses.mean_squared_error(labels=Y,predictions=output)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train_relu = optimizer.minimize(loss_relu)

init = tf.global_variables_initializer()
sess.run(init)

writer = tf.summary.FileWriter('./event_log')
writer.add_graph(tf.get_default_graph())
#print (sess.run((y_p_sigmod,y_p_relu),{x:X}))


tx = np.random.uniform(low=MIN,high=MAX,size=10)
print (tx)
ntx = tx.reshape(10,1)
nty = myNumpyFunc(ntx)
print (nty)
print (sess.run((output),{x:ntx}))
print ("inital error above")
print ("")

cnt = 0
for i in range(50000):
    _,l_relu = sess.run((train_relu,loss_relu),{x:X})
    cnt += 1
    if cnt % 5000 == 0:
        print ("in loop: " +str(i))
#        print ("sigmod: " + str(l_sigmod))
        print ("relu: "+ str(l_relu))
        print ("")

tx = np.random.uniform(low=MIN,high=MAX,size=10)
print (tx)
ntx = tx.reshape(10,1)
nty = myNumpyFunc(ntx)
print (nty)
#print (sess.run((y_p_sigmod),{x:ntx}))
print (sess.run((output),{x:ntx}))
##
## ## x = tf.placeholder(tf.float32, shape=[None,3])
## ## linear_model = tf.layers.Dense(units=1)
## ## y = linear_model(x)
##
## x = tf.constant ([[1],[2],[3],[4]], dtype = tf.float32)
## y_t = tf.constant ([[0],[-1],[-2],[-3]], dtype = tf.float32)
## linear_model = tf.layers.Dense(units=1)
## y_a = linear_model(x)
##
## loss = tf.losses.mean_squared_error(labels= y_t,predictions=y_a)
## optimizer = tf.train.GradientDescentOptimizer(0.1)
## train = optimizer.minimize(loss)
##
## init = tf.global_variables_initializer()
## sess.run(init)
##
## print (sess.run({'ya':y_a,"loss":loss}))
##
## for i in range(100):
##     _, loss_value = sess.run((train,loss))
##     print (loss_value)
##     print ("")
##
## print (sess.run(y_a))
## writer = tf.summary.FileWriter('.')
## writer.add_graph(tf.get_default_graph())
