#coding:utf-8
import tensorflow as tf 
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets('./data', one_hot=True)

# Training Parameters
learning_rate = 0.001
training_steps = 800
#batch_size = 128
display_step = 20

num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # time steps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

tf.reset_default_graph()

# tf Draph input
X = tf.placeholder(tf.float32,[None,timesteps,num_input],name="input_x")
Y = tf.placeholder(tf.float32,[None,num_classes],name="input_y")
batch_size = tf.placeholder(tf.int32,[],name="batch_size")

# Define weights
weights = {
    # (28,128)
    'in': tf.Variable(tf.random_normal([num_input,num_hidden])),

    #(128,10)
    'out': tf.Variable(tf.random_normal([num_hidden,num_classes]))
}

biases = {
    #(128,)
    'in':tf.Variable(tf.constant(0.1,shape = [num_hidden,])),

    #(10,)
    'out':tf.Variable(tf.constant(0.1,shape = [num_classes,]))
}

# Define RNN

def RNN(X, batch_size, weights, biases):
    #### hidden layer for input to  cell #####

    #改变输入的shape X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X,[-1,num_input])
    X_in = tf.matmul(X,weights['in']+biases['in'])
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in,[-1,timesteps,num_hidden])


    ######################cell##########################
    cell = rnn.BasicLSTMCell(num_hidden)
    #一个lstm cell 有两部分 c_state, h_state
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    #### hindden layer for output as final results ####
    # # way 1
    # outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    # result = tf.matmul(outputs[-1], weights['out']) + biases['out']    #调用最后一个 outputs
    #or
    # way 2
    result = tf.matmul(final_state[1], weights['out']) + biases['out'] #直接调用final_state 中的 h_state (final_state[1]) 来进行运算
    return result  #shape = (128, 10)


# Define loss and optimizer
pred = RNN(X, batch_size, weights, biases)
tf.add_to_collection('pred', pred)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
tf.add_to_collection('loss_op', loss_op)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op =  optimizer.minimize(loss_op)


# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.add_to_collection('accuracy', accuracy)

#start
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1, training_steps+1):
        
        batch_x, batch_y = mnist.train.next_batch(128)        
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((128, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, batch_size: 128})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y, batch_size: 128})
            print("Step " + str(step) + ", 小批量损失 = " + \
                  "{:.4f}".format(loss) + ", 训练准确度 = " + \
                  "{:.3f}".format(acc))
    print("优化完成")
    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("测试精度为 :", sess.run(accuracy, feed_dict={X: test_data, Y: test_label, batch_size: 128}))
    
    saver = tf.train.Saver()
    model_path = "./model/my_model"
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
    

    

X3 = mnist.train.images[4]
img3 = X3.reshape([28, 28])
plt.imshow(img3, cmap='gray')
plt.show()

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./model/my_model.meta')
    new_saver.restore(sess, './model/my_model')
    graph = tf.get_default_graph()    
    loss_op = tf.get_collection('loss_op')[0]
    accuracy = tf.get_collection('accuracy')[0]
    pred = tf.get_collection('pred')[0]
    X = graph.get_operation_by_name("input_x").outputs[0]
    batch_size = graph.get_operation_by_name("batch_size").outputs[0]
    
    size1 = 1
    
    batch_x, batch_y = mnist.train.next_batch(size1)
    batch_x = batch_x.reshape((size1, timesteps, num_input))
    
    predict = sess.run([pred],feed_dict={X:batch_x,batch_size: size1})
    print("prediction:",predict)
    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y,batch_size: size1})
    print("Step " + str(step) + ", 小批量损失 = " + \
          "{:.4f}".format(loss) + ", 训练准确度 = " + \
          "{:.3f}".format(acc))
    maxlabel = np.argmax(predict)
    print("maxlabel:",maxlabel)