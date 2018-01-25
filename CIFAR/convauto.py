import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# import data
from tensorflow.contrib.keras.python.keras.datasets import cifar10
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# to one-hot 
targets = np.array([y_train]).reshape(-1)
y_train = np.eye(10)[targets]
targets = np.array([y_test]).reshape(-1)
y_test = np.eye(10)[targets]

# normalisation
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_train = min_max_scaler.fit_transform(x_train.reshape([x_train.shape[0], 32*32*3])).reshape([x_train.shape[0], 32, 32, 3])
x_test = min_max_scaler.fit_transform(x_test.reshape([x_test.shape[0], 32*32*3])).reshape([x_test.shape[0], 32, 32, 3])

# parameters
save_path = "./cifar_auto.ckp"
num_steps = 10000
batch_size = 128
print_every = 100 # print loss every 100 epochs
num_reconstr_digits = 10 # how many reconstructed digits to display at the end
batchnorm = True

# network parameters
beta_max = 0.9
eta = 5
num_filters = 16

# variables
x = tf.placeholder("float", [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool, [])

def next_batch(size, images, labels):
    indices = np.arange(0 , len(images))
    np.random.shuffle(indices)
    indices = indices[:size]
    shuffled_images = [images[ i] for i in indices]
    shuffled_labels = [labels[ i] for i in indices]

    return np.asarray(shuffled_images), np.asarray(shuffled_labels)

def eval_testset():
    losses = []
    
    for i in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
        x_test_batch = x_test[i-1000:i]
        y_test_batch = y_test[i-1000:i]
        test_cifar = {x: x_test_batch, is_training: False}
        
        losses.append(loss_op.eval(test_cifar))

    test_loss = np.mean(losses)    
    print("Loss on test set:", test_loss)

def comp_feedback(fb_inp):
    return 1/(1-tf.minimum(beta_max/eta * fb_inp, beta_max))



with tf.variable_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 3, num_filters], stddev=0.0001))    
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[num_filters]))
    pre_activation = tf.nn.bias_add(conv, biases)
    if batchnorm: pre_activation = tf.layers.batch_normalization(pre_activation, training=is_training)

    conv1 = tf.nn.relu(pre_activation, name=scope.name)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

with tf.variable_scope('conv2') as scope:
    kernel2 = tf.Variable(tf.truncated_normal([5, 5, num_filters, num_filters], stddev=0.0001))    
    conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
    biases2 = tf.Variable(tf.constant(0.0, shape=[num_filters]))
    pre_activation2 = tf.nn.bias_add(conv2, biases2)
    if batchnorm: pre_activation2 = tf.layers.batch_normalization(pre_activation2, training=is_training)

    conv2 = tf.nn.relu(pre_activation2, name=scope.name)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

with tf.variable_scope('deconv1') as scope:
    kernel3 = tf.Variable(tf.truncated_normal([5, 5, num_filters, num_filters], stddev=0.0001))
    biases3 = tf.Variable(tf.constant(0.0, shape=[num_filters]))
    deconv1 = tf.nn.conv2d_transpose(pool2, kernel3, [tf.shape(x)[0], 16, 16, num_filters], [1, 2, 2, 1], padding='SAME')
    pre_activation3 = tf.nn.bias_add(deconv1, biases3)
    if batchnorm: pre_activation3 = tf.layers.batch_normalization(pre_activation3, training=is_training)

    deconv1 = tf.nn.relu(pre_activation3)

with tf.variable_scope('deconv2') as scope:
    kernel4 = tf.Variable(tf.truncated_normal([5, 5, 3, num_filters], stddev=0.0001))
    biases4 = tf.Variable(tf.constant(0.0, shape=[3]))
    out = tf.nn.conv2d_transpose(deconv1, kernel4, [tf.shape(x)[0], 32, 32, 3], [1, 2, 2, 1], padding='SAME')
    pre_activation4 = tf.nn.bias_add(out, biases4)
    out = tf.nn.tanh(pre_activation4)

####### 2nd pass
with tf.variable_scope('conv1_t2') as scope:    
    conv_t2 = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    pre_activation_t2 = tf.nn.bias_add(conv_t2, biases)
    if batchnorm: pre_activation_t2 = tf.layers.batch_normalization(pre_activation_t2, training=is_training)

    fb_i = tf.concat([tf.contrib.layers.flatten(pool2), tf.contrib.layers.flatten(deconv1), tf.contrib.layers.flatten(out)], 1)
    fb_w = tf.Variable(tf.truncated_normal([(8*8*num_filters)+(16*16*num_filters)+(32*32*3), num_filters], stddev=0.0001))
    fb = tf.reshape(tf.matmul(fb_i, fb_w, name="fbmul"), [tf.shape(x)[0], 1, 1, num_filters])
    
    conv_t2 = comp_feedback(fb) * tf.nn.relu(pre_activation_t2)


pool1_t2 = tf.layers.max_pooling2d(inputs=conv_t2, pool_size=[2, 2], strides=2)

with tf.variable_scope('conv2_t2') as scope:    
    conv2_t2 = tf.nn.conv2d(pool1_t2, kernel2, [1, 1, 1, 1], padding='SAME')
    pre_activation2_t2 = tf.nn.bias_add(conv2_t2, biases2)
    if batchnorm: pre_activation2_t2 = tf.layers.batch_normalization(pre_activation2_t2, training=is_training)

    fb_i = tf.concat([tf.contrib.layers.flatten(deconv1), tf.contrib.layers.flatten(out)], 1)
    fb_w = tf.Variable(tf.truncated_normal([(16*16*num_filters)+(32*32*3), num_filters], stddev=0.0001))
    fb = tf.reshape(tf.matmul(fb_i, fb_w, name="fbmul"), [tf.shape(x)[0], 1, 1, num_filters])

    conv2_t2 = comp_feedback(fb) * tf.nn.relu(pre_activation2_t2, name=scope.name)


pool2_t2 = tf.layers.max_pooling2d(inputs=conv2_t2, pool_size=[2, 2], strides=2)

with tf.variable_scope('deconv1_t2') as scope:    
    deconv1_t2 = tf.nn.conv2d_transpose(pool2_t2, kernel3, [tf.shape(x)[0], 16, 16, num_filters], [1, 2, 2, 1], padding='SAME')
    pre_activation3_t2 = tf.nn.bias_add(deconv1_t2, biases3)
    if batchnorm: pre_activation3_t2 = tf.layers.batch_normalization(pre_activation3_t2, training=is_training)

    fb_i = tf.concat([tf.contrib.layers.flatten(out)], 1)
    fb_w = tf.Variable(tf.truncated_normal([32*32*3, num_filters], stddev=0.0001))
    fb = tf.reshape(tf.matmul(fb_i, fb_w, name="fbmul"), [tf.shape(x)[0], 1, 1, num_filters])

    deconv1_t2 = comp_feedback(fb) * tf.nn.relu(pre_activation3_t2)


with tf.variable_scope('deconv2_t2') as scope:    
    out_t2 = tf.nn.conv2d_transpose(deconv1_t2, kernel4, [tf.shape(x)[0], 32, 32, 3], [1, 2, 2, 1], padding='SAME')
    pre_activation4_t2 = tf.nn.bias_add(out_t2, biases4)
    out_t2 = tf.nn.tanh(pre_activation4_t2)


############################################
decoded = out_t2

# define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.mean_squared_error(labels=x, predictions=decoded))

optimizer = tf.train.AdamOptimizer()

# update_ops is necessary for batchnorm to work properly 
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss_op)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# initialize variables
init = tf.global_variables_initializer()

# training
sess = tf.Session()
sess.run(init)

for step in range(1, num_steps):
    batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
    
    sess.run(train_op, feed_dict={x: batch_xs, is_training: True})

    if step % print_every == 0:
        loss = sess.run(loss_op, feed_dict={x: batch_xs, is_training: True})

        print("Step:", step, "| loss:", loss)

        with sess.as_default():
            eval_testset()

# save trained model 
save_path = saver.save(sess, save_path)
print("Parameters saved as:" + save_path)

# plot reconstructions 
with sess.as_default():
    xtest_batch, ytest_batch = next_batch(batch_size, x_train, y_train)
    test_cifar = {x: xtest_batch, is_training: False}
    reconstr = decoded.eval(test_cifar)

    for i in range(12):
        # original
        ax = plt.subplot(2, 12, i+1)
        plt.imshow((xtest_batch[i]+1)/2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # reconstructed
        ax = plt.subplot(2, 12, i+1+12)
        plt.imshow((reconstr[i]+1)/2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('cifar_reconstructions.png')


