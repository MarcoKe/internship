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
beta_max = 0.95
eta = 5
input_nodes = 784 # each image in the dataset has a size of 28*28px
encode1_nodes = 392 
encode2_nodes = 10
decode1_nodes = encode1_nodes
output_nodes = input_nodes
timesteps = 2
layer_dims = [input_nodes, encode1_nodes, encode2_nodes, decode1_nodes, output_nodes]
fb_connections = [[2, 3], [], [], []] # i.e. [[3],[],[],[]] means that the first layer receives feedback from layer 3. counting starts from the first hidden layer

# variables
x = tf.placeholder("float", [None, input_nodes])

def create_weights(dims): 
    return tf.Variable(tf.truncated_normal(dims, stddev=0.0001))

def create_bias(dim): 
    return tf.Variable(tf.constant(0.1, shape=[dim]))

def calc_fb_dims(layer): 
    return [sum([layer_dims[conn+1] for conn in fb_connections[layer]]), layer_dims[layer+1]]

def comp_feedback(fb_inp):
    return 1/(1-tf.minimum(beta_max/eta * fb_inp, beta_max))

architecture = [{'ffw': create_weights([layer_dims[l], layer_dims[l+1]]), 'b': create_bias(layer_dims[l+1]), 'fbw': create_weights(calc_fb_dims(l))} for l in range(len(layer_dims)-1)]

activations = [0.0 for i in range(len(fb_connections))]
prev_activations = activations

for t in range(timesteps): 
    prev_activations = activations

    with tf.name_scope('encoder1_t'+str(t)) as scope:
        encoder1_ff = tf.add(tf.matmul(x, architecture[0]['ffw'], name="ffmul"), architecture[0]['b'])
        encoder1_fb = 1 
        if t != 0 and architecture[0]['fbw'].shape[0] != 0: 
            encoder1_fb = tf.matmul(tf.concat([prev_activations[conn] for conn in fb_connections[0]], 1), architecture[0]['fbw'], name="fbmul")

        activations[0] = tf.multiply(comp_feedback(encoder1_fb), tf.nn.relu(encoder1_ff), name="appfbmul")

    with tf.name_scope('encoder2_t'+str(t)) as scope:
        encoder2_ff = tf.add(tf.matmul(activations[0], architecture[1]['ffw']), architecture[1]['b'])
        activations[1] = tf.nn.relu(encoder2_ff)

    with tf.name_scope('decoder1_t'+str(t)) as scope:
        decoder1_ff = tf.add(tf.matmul(activations[1], architecture[2]['ffw']), architecture[2]['b'])
        activations[2] = tf.nn.relu(decoder1_ff)

    with tf.name_scope('decoder2_t'+str(t)) as scope:
        decoder2_ff = tf.add(tf.matmul(activations[2], architecture[3]['ffw']), architecture[3]['b'])
        activations[3] = tf.nn.sigmoid(decoder2_ff)

 
decoded = activations[3]

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
            first_pass_reconstruction = prev_activations[3].eval(test_mnist)
            plotter.plotTwoPassWithDiff(original_images, first_pass_reconstruction, second_pass_reconstruction, str(step))

if save_model: save_path = saver.save(sess, save_path)
       
