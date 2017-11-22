import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import reconstructionPlotter
import sys

# import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# parameters
num_steps = 10000
batch_size = 128
print_every = 100 # print loss every % batches
plot_every = 1000 # plot reconstruction every % batches
num_reconstr_digits = 10 # how many reconstructed digits to display at the end
save_logs = False # whether to save logs for use with tensorboard
save_model = False 
restore_model = False
save_path = "./model.ckp"

# network parameters
beta_max = 2/3
eta = 20
input_nodes = 784 # each image in the dataset has a size of 28*28px
encode1_nodes = 392 
encode2_nodes = 10
decode1_nodes = encode1_nodes
output_nodes = input_nodes
timesteps = 2

# variables
x = tf.placeholder("float", [None, input_nodes])

def create_weights(dims): 
    return tf.Variable(tf.truncated_normal(dims, stddev=0.0001))

def create_bias(dim): 
    return tf.Variable(tf.constant(0.1, shape=[dim]))

def comp_feedback(fb_inp):
    return 1/(1-tf.minimum(beta_max/eta * fb_inp, beta_max))

# network 
with tf.name_scope('encoder1_t1') as scope:
    w1 = create_weights([input_nodes, encode1_nodes])
    b1 = create_bias(encode1_nodes)
    encoder1 = tf.nn.relu(tf.add(tf.matmul(x, w1, name="mul1"), b1))

with tf.name_scope('encoder2_t1') as scope:
    w2 = create_weights([encode1_nodes, encode2_nodes])
    b2 = create_bias(encode2_nodes)
    encoder2 = tf.nn.relu(tf.add(tf.matmul(encoder1, w2, name="mul2"), b2))

with tf.name_scope('decoder1_t1') as scope:
    w3 = create_weights([encode2_nodes, decode1_nodes])
    b3 = create_bias(decode1_nodes)
    decoder1 = tf.nn.relu(tf.add(tf.matmul(encoder2, w3, name="mul3"), b3))

with tf.name_scope('decoder2_t1') as scope:
    w4 = create_weights([decode1_nodes, output_nodes])
    b4 = create_bias(output_nodes)
    decoder2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder1, w4, name="mul4"), b4))

# second pass
with tf.name_scope('encoder1_t2') as scope:
    w5 = tf.Variable(tf.truncated_normal([decode1_nodes+output_nodes, encode1_nodes], stddev=0.0001), name="feedback-weights")
    encoder1_2 = tf.add(tf.matmul((x), w1, name="mul5"), b1)
    fb = tf.matmul(tf.concat([decoder1, decoder2], 1), w5)
    encoder1_2 = comp_feedback(fb) * tf.nn.relu(encoder1_2)

with tf.name_scope('encoder2_t2') as scope:
    encoder2_2 = tf.nn.relu(tf.add(tf.matmul((encoder1_2), w2, name="mul6"), b2))

with tf.name_scope('decoder1_t2') as scope:
    decoder1_2 = tf.nn.relu(tf.add(tf.matmul(encoder2_2, w3), b3))

with tf.name_scope('decoder2_t2') as scope:
    decoder2_2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder1_2, w4), b4))

 
decoded = decoder2_2

# define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.log_loss(labels=x, predictions=decoded))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# initialize variables
init = tf.global_variables_initializer()

# training
sess = tf.Session()
sess.run(init)

if restore_model: saver.restore(sess, save_path)

for step in range(1, num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    if save_logs: writer = tf.summary.FileWriter('logs', sess.graph)
    sess.run(train_op, feed_dict={x: batch_xs})
    if save_logs: writer.close()

    if step % print_every == 0:
        loss = sess.run(loss_op, feed_dict={x: batch_xs})
        print("Step:", step, "| Minibatch loss:", loss)
        with sess.as_default():
            test_mnist = {x: mnist.validation.images}
            print("Loss on validation set:", loss_op.eval(test_mnist))


    if step % plot_every == 0: 
        plt.figure()
        original_images = mnist.test.images

        with sess.as_default():            
            plotter = reconstructionPlotter.ReconstructionPlotter(num_reconstr_digits, "mnist_autoencoder_")
            test_mnist = {x: mnist.test.images}
            second_pass_reconstruction = decoded.eval(test_mnist)
            first_pass_reconstruction = decoder2.eval(test_mnist)
            plotter.plotTwoPassWithDiff(original_images, first_pass_reconstruction, second_pass_reconstruction, str(step))

if save_model: save_path = saver.save(sess, save_path)
       
