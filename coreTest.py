import numpy as np
import pandas as pd
import tensorflow as tf

N = 100
MIN = 0
MAX = 10

def myNumpyFunc(nx):
    def myFunc (x):
        y = 0.2*x - 0.4*x**2 - x - 1
        return y
    lx = list(nx)
    ly = list(map(myFunc, lx))
    ny = np.asarray(ly)
    return ny

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

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None,1])

hidden_sigmod = tf.layers.Dense(units=10, activation=tf.nn.sigmoid)
hidden_relu = tf.layers.Dense(units=10,activation=tf.nn.relu)

linear_model = tf.layers.Dense(units=1)

optimizer = tf.train.GradientDescentOptimizer(0.1)

y_p_sigmod = linear_model(hidden_sigmod(x))
loss_sigmod = tf.losses.mean_squared_error(labels=Y,predictions=y_p_sigmod)
train_sigmod = optimizer.minimize(loss_sigmod)

y_p_relu =  linear_model(hidden_relu(x))
loss_relu= tf.losses.mean_squared_error(labels=Y,predictions=y_p_relu)
train_relu = optimizer.minimize(loss_relu)

init = tf.global_variables_initializer()
sess.run(init)

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
#print (sess.run((y_p_sigmod,y_p_relu),{x:X}))


tx = np.random.uniform(low=MIN,high=MAX,size=10)
print (tx)
ntx = tx.reshape(10,1)
nty = myNumpyFunc(ntx)
print (nty)
print (sess.run((y_p_sigmod),{x:ntx}))
print (sess.run((y_p_relu),{x:ntx}))
print ("inital error above")
print ("")

cnt = 0
for i in range(50000):
    _,l_sigmod= sess.run((train_sigmod,loss_sigmod),{x:X})
#    _,l_relu = sess.run((train_relu,loss_relu),{x:X})
    cnt += 1
    if cnt % 10000 == 0:
        print ("in loop: " +str(i))
        print ("sigmod: " + str(l_sigmod))
#        print ("relu: "+ str(l_relu))
        print ("")

tx = np.random.uniform(low=MIN,high=MAX,size=10)
print (tx)
ntx = tx.reshape(10,1)
nty = myNumpyFunc(ntx)
print (nty)
print (sess.run((y_p_sigmod),{x:ntx}))
#print (sess.run((y_p_relu),{x:ntx}))
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
