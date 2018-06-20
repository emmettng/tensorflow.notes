import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
FLAGS = None
SUMMARY = True
'''
Auxiliary Functions below:
'''
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def pool2d(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')

def variable_summaries(var,):
    with tf.name_scope('scalar_summaries'):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

'''
Necessary Functions below:
'''
def trans_active(input,kernel_shape,bias_shape,kernel_op,active_op):
    ## define ' kenel', 'bias', and corresponding 'initial value', 'shape'
    kernel_init = tf.truncated_normal(kernel_shape,mean=0, stddev=0.1) ## truncated_normal?
    W = tf.get_variable('W',initializer=kernel_init)
    bias_init = tf.constant(0.1,shape=bias_shape)
    b = tf.get_variable('b',initializer = bias_init)

    ## compose the transformation and active function together
    trans = kernel_op(input,W)
    h = active_op(trans + b)

    ## record summary
    if SUMMARY:
        with tf.name_scope('weights'):
            variable_summaries(W)
        with tf.name_scope('bias'):
            variable_summaries(b)
        with tf.name_scope('trans'):
            tf.summary.histogram('histogram',trans)
        with tf.name_scope('active'):
            tf.summary.histogram('histogram',h)
    return h



'''
 def_GRAPH
 Input :
    Compose the computational graph, and the only input is 'x'

 Return:
    The composition of series of function, could be evaluated by a session or as an first order function feed to other functions, such as def_LOSS.
'''
def def_GRAPH(x):
    ## reshape the data from 1d to 2d
    activeDict =  dict()
    with tf.name_scope('reshape'):
        x_2d_images = tf.reshape(x,[-1,28,28,1])
        ## Below is a test of summary
    with tf.name_scope('imgs'):
        if SUMMARY:
            tf.summary.image('input',x_2d_images,10)

    x_noise = tf.add(x,tf.truncated_normal(shape=[50,784],stddev=0.5))

    with tf.variable_scope('hidden1'):
        h1= trans_active(x_2d_images,[5,5,1,32],[32],conv2d,tf.nn.leaky_relu)
        h1_pool = pool2d(h1)
        activeDict['hidden1'] = h1


##    with tf.variable_scope('hidden2'):
##        h2= trans_active(h1,[2000,2000],[2000],tf.matmul,tf.nn.leaky_relu)

##        h_pool1 = pool2d(h_conv1)

##    with tf.variable_scope('conv2'):
##        h_conv2 = trans_active(h_pool1,[5,5,32,64],[64],conv2d,tf.nn.relu)
##        h_pool2 = pool2d(h_conv2)
##
##    with tf.variable_scope('fc1'):
##        h_pool2_flatten = tf.reshape(h_pool2,[-1,7*7*64])
##        h_fc1 = trans_active(h_pool2_flatten,[7*7*64,1024],[1024],tf.matmul,tf.nn.relu)
##
##    with tf.name_scope('dropout'):
##        keep_prob = tf.placeholder(tf.float32)  ## placeholder must be passed or returned
##        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
##
##        if SUMMARY:
##            tf.summary.scalar('dropout_keep_probability',keep_prob)
##
    with tf.variable_scope('out_put'):
        h1_pool_flatten = tf.reshape(h1_pool,[-1,14*14*32])
        y_Hypo = trans_active(h1_pool_flatten,[14*14*32,784],[784],tf.matmul,tf.nn.leaky_relu)

    with tf.name_scope('reshape'):
        y_2d_images = tf.reshape(y_Hypo,[-1,28,28,1])
        ## Below is a test of summary
    with tf.name_scope('imgs'):
        if SUMMARY:
            tf.summary.image('output',y_2d_images,10)

    return y_Hypo


def def_LOSS(y_,y_Hypo):
    with tf.name_scope('loss'):
        cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(y_,y_Hypo),2.0))
##        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)

        if SUMMARY:
            tf.summary.scalar('auto_loss',tf.reduce_mean(cost))
    return cost

def def_ACCURACY(y_Hypo, y_):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_Hypo,1),tf.argmax(y_,1))
        correct_prediction = tf.cast(correct_prediction,tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        if SUMMARY:
            tf.summary.scalar('accuracy',accuracy)
    return accuracy

def def_Dis_Session():
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
##    this order cannot be changed !!!! be aware ! what a crap api design !!!
##  Define the Session
    global_step = tf.contrib.framework.get_or_create_global_step()

    sess = tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir=FLAGS.log_dir,
                                           hooks=hooks)

    return (server,sess,global_step)

def def_Local_Session():
    sess = tf.Session()
    return sess

def def_ComposeGraph(global_step):
    if FLAGS.distribute == 'distribute':
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
        device_setter = tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_id,
            cluster=cluster)
    else:
        device_setter = "/cpu:0"

    with tf.device(device_setter):
        y_ = tf.placeholder(dtype=tf.float32, shape = [None,784])
        ## compose Graph
        y_Hypo = def_GRAPH(y_)
        ## get loss definition
        loss = def_LOSS(y_,y_Hypo)
        ## define optimizer
        optimizer = tf.train.AdamOptimizer(1e-4)

        ## define training_handle
        if FLAGS.distribute == 'distribute':
            training_handle = optimizer.minimize(loss,global_step = global_step)
        else:
            training_handle = optimizer.minimize(loss)
        ## accuracy definition
##           accuracy = def_ACCURACY(y_Hypo=y_Hypo, y_=y_)
        ## regiester all variable for initilization in this graph
        graph_varialbe_init= tf.global_variables_initializer()

        ## merge all summary tensor in this graph.
        merged_summary = tf.summary.merge_all()

    return ([],y_,y_Hypo,loss,training_handle,graph_varialbe_init,merged_summary)

'''
main:   compose all necessary functions together!
- data input
- grap define
    - placeholders for true <X,Y>
    - compose graph : X -> ( Y_hypo, others)
    - choice optimizer
    - compose cost : Y -> Y_hypo -> loss
    - define training_handle
    - compose accuracy for assessment
    - define variable initializer
- define sumamry (optional)
- define session
- compose training loop
'''
def train_boday():
    ## data input part , this part better be a generator
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    ## define the Computational Graph
    mg = tf.Graph()

    ## define session
    with mg.as_default():
        if FLAGS.distribute == 'distribute':
            server, sess,global_step = def_Dis_Session()
        else:
            sess = def_Local_Session()

    if FLAGS.job_name == 'ps':
        server.join()
    else:
        with mg.as_default():
            (
                clist,
                y_,
                y_Hypo,
                loss,
                training_handle,
                graph_varialbe_init,
                merged_summary
                ) = def_ComposeGraph(global_step)

                ## The following loop must be in the mg.as_default scope so globa_variables will contains all variables.
            for v in tf.global_variables():
                print (v)
                print (v.name)

            ## define summary log directory.
        training_summary= tf.summary.FileWriter(FLAGS.log_dir + '/train')
 ##                   testing_summary = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        training_summary.add_graph(mg)


        sess.run(graph_varialbe_init)

        ## compose training loop
        training_ROUND = 1000000
        cnt = training_ROUND

        with sess.as_default():
            while not sess.should_stop():
                batch_xs, batch_ys = mnist.train.next_batch(50)
                mon_sess.run(train_op,feed_dict={y_:batch_xs})
                cnt+=1
                if cnt % 10 ==0:
                    print ("the loss is")
                    tloss = sess.run(loss,feed_dict={y_:batch_xs})
                    summary = sess.run(merged_summary,feed_dict={y_:batch_xs})
                    training_summary.add_summary(summary,cnt)
                    print ("loss %d , in training loop %g" % (tloss,cnt))
##        for i in range(training_ROUND):
##            batch_xs, batch_ys = mnist.train.next_batch(50)
##            ## need to specify Session instance explicitly
##            training_handle.run(feed_dict={y_:batch_xs},session = sess)
##
##            if i % 100 == 0 and SUMMARY:
##                cost = loss.eval(feed_dict={y_:batch_xs},session = sess)
##                summary = sess.run(merged_summary,feed_dict={y_:batch_xs})
##                training_summary.add_summary(summary,i)
##                print ("step %d, training cost is: %g" % (i,cost))

    ## release all resources
    sess.close()
    training_summary.close()

def main(_):
    train_boday()
##    server, session = def_Session(FLAGS.distribute)
##    if FLAGS.job_name == 'ps':
##        server.join()
##    else:
##        train_boday(session)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--distribute",
      type=str,
      default="local",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  parser.add_argument(
      '--task_id',
      type=int,
      default=0,
      help='Directory for storing input data')
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default = './event_log',
      help='Summaries log directory')

  parser.add_argument(
      '--data_dir',
      type=str,
      default='./mnist',
      help='Directory for storing input data')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
