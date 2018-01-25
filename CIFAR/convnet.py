import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.keras.python.keras.backend import random_binomial
import sys

# import data
from tensorflow.contrib.keras.python.keras.datasets import cifar10
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# to one-hot 
targets = np.array([y_train]).reshape(-1)
y_train = np.eye(10)[targets]
targets = np.array([y_test]).reshape(-1)
y_test = np.eye(10)[targets]


# parameters
save_path = "./convnet.ckp"
save_model = False
restore_model = False 
num_steps = 100000
batch_size = 128
print_every = 100 # print loss every 100 epochs
num_reconstr_digits = 10 # how many reconstructed digits to display at the end

# network parameters
beta_max = 0.9
eta = 5
##beta_max = float(sys.argv[1])
##eta = float(sys.argv[2])

num_filters1 = 64  # num filters in conv layer 1 
num_filters2 = 64  # num filters in conv layer 2 
num_units = 200    # num units in fully connected layer 
num_classes = 10 

# variables
x = tf.placeholder("float", [None, 32,32,3])
y_ = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool, [])
dropout_rate = tf.placeholder(tf.float32, [])

# 0 mean, unit variance normalisation 
def normalize(X_train,X_test):    
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

x_train, x_test = normalize(x_train, x_test)

# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

batches = datagen.flow(x_train, y_train, batch_size=batch_size)

def create_conv_weights(shape, name):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def comp_feedback(fb_inp):
    return 1/(1-tf.minimum(beta_max/eta * fb_inp, beta_max))

def create_dropout_mask(layer, rate): 
    return random_binomial(layer.get_shape()[1:].as_list(), p=1.0-rate)

def custom_dropout(layer, mask, rate): 
    return mask * (1.0/(1.0-rate)) * layer

def eval_testset():
    losses = []
    accs = []    
    
    for i in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
        x_test_batch = x_test[i-1000:i]
        y_test_batch = y_test[i-1000:i]
        test_cifar = {x: x_test_batch, y_: y_test_batch, dropout_rate: 0.0, is_training: False}
        
        losses.append(loss_op.eval(test_cifar))
        accs.append(accuracy.eval(test_cifar))        

    test_loss = np.mean(losses)
    test_acc = np.mean(accs)
    print("Loss on test set:", test_loss, "|acc:", test_acc)


#############################
# network 
with tf.variable_scope('conv1') as scope:
    kernel = create_conv_weights([5, 5, 3, num_filters1], 'kernel')    
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[num_filters1]))
    pre_activation = tf.nn.bias_add(conv, biases)    
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    conv1 = tf.layers.batch_normalization(conv1, training=is_training)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# custom dropout (tensorflow dropout does not allow to synchronize dropout between passes)
mask_pool1 = create_dropout_mask(pool1, dropout_rate)
pool1 = custom_dropout(pool1, mask_pool1, dropout_rate)

with tf.variable_scope('conv2') as scope:
    kernel2 = create_conv_weights([5, 5, num_filters1, num_filters2], 'kernel2')    
    conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
    biases2 = tf.Variable(tf.constant(0.0, shape=[num_filters2]))
    pre_activation2 = tf.nn.bias_add(conv2, biases2)    
    conv2 = tf.nn.relu(pre_activation2, name=scope.name)
    conv2 = tf.layers.batch_normalization(conv2, training=is_training)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
mask_pool2 = create_dropout_mask(pool2, dropout_rate)
pool2 = custom_dropout(pool2, mask_pool2, dropout_rate)

with tf.variable_scope('fully_connected') as scope:
    flattened = tf.contrib.layers.flatten(pool2)
    w1 = tf.Variable(tf.truncated_normal([flattened.get_shape().as_list()[1], num_units], stddev=0.0001))
    b1 = tf.Variable(tf.constant(0.1, shape=[num_units]))
    fc = tf.nn.relu(tf.add(tf.matmul(flattened, w1), b1))    
    mask_fc = create_dropout_mask(fc, dropout_rate)
    fc = custom_dropout(fc, mask_fc, dropout_rate)

with tf.variable_scope('softmax_linear') as scope: 
    w2 = tf.Variable(tf.truncated_normal([num_units, num_classes], stddev=0.0001))
    b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    fc2 = tf.add(tf.matmul(fc, w2), b2)


#######################################
with tf.variable_scope('conv1_t2') as scope:    
    conv_t2 = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    pre_activation_t2 = tf.nn.bias_add(conv_t2, biases)

    fb_i = tf.concat([tf.reshape(tf.reduce_mean(pool2, [1,2,3]), [tf.shape(x)[0],1]), fc, fc2], 1)
    fb_w = tf.Variable(tf.truncated_normal([fb_i.get_shape().as_list()[1], num_filters1], stddev=0.0001), name="feedback-weights")    
    fb = tf.reshape(tf.matmul(fb_i, fb_w, name="fbmul"), [tf.shape(x)[0], 1, 1, num_filters1])

    conv1_t2 = comp_feedback(fb) * tf.nn.relu(pre_activation_t2)
    conv1_t2 = tf.layers.batch_normalization(conv1_t2, training=is_training)

pool1_t2 = tf.layers.max_pooling2d(inputs=conv1_t2, pool_size=[2, 2], strides=2)
pool1_t2 = custom_dropout(pool1_t2, mask_pool1, dropout_rate)

with tf.variable_scope('conv2_t2') as scope:    
    conv2_t2 = tf.nn.conv2d(pool1_t2, kernel2, [1, 1, 1, 1], padding='SAME')
    pre_activation2_t2 = tf.nn.bias_add(conv2_t2, biases2)

    fb_i = tf.concat([fc, fc2], 1)
    fb_w = tf.Variable(tf.truncated_normal([fb_i.get_shape().as_list()[1], num_filters2], stddev=0.0001))
    fb = tf.reshape(tf.matmul(fb_i, fb_w), [tf.shape(x)[0], 1, 1, num_filters2])

    conv2_t2 = comp_feedback(fb) * tf.nn.relu(pre_activation2_t2, name=scope.name)
    conv2_t2 = tf.layers.batch_normalization(conv2_t2, training=is_training)

pool2_t2 = tf.layers.max_pooling2d(inputs=conv2_t2, pool_size=[2, 2], strides=2)
pool2_t2 = custom_dropout(pool2_t2, mask_pool2, dropout_rate)

with tf.variable_scope('fully_connected_t2'): 
    flattened_t2 = tf.contrib.layers.flatten(pool2_t2)
    pre_activation = tf.add(tf.matmul(flattened_t2, w1), b1)

    fb_i = fc2
    fb_w = tf.Variable(tf.truncated_normal([fb_i.get_shape().as_list()[1], num_units], stddev=0.0001))
    fb = tf.matmul(fb_i, fb_w)

    fc_t2 = comp_feedback(fb) * tf.nn.relu(pre_activation)
    fc_t2 = custom_dropout(fc_t2, mask_fc, dropout_rate) * fc_t2


with tf.variable_scope('softmax_linear_t2'):
    fc2_t2 = tf.add(tf.matmul(fc_t2, w2), b2)


############################################
output = fc2_t2
pred_classes = tf.argmax(input=output, axis=1)
pred_probs = tf.nn.softmax(output)

# define loss and optimizer
loss_op = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=output)
optimizer = tf.train.AdamOptimizer(epsilon=0.1)

# need update_ops for batchnorm to work properly 
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# initialize variables
init = tf.global_variables_initializer()

# training
sess = tf.Session()
sess.run(init)

if restore_model: saver.restore(sess, save_path)
    
for step in range(1, num_steps):
    batch_xs, batch_ys = batches.next()
    sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys, dropout_rate: 0.5, is_training: True})

    if step % print_every == 0:
        loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_xs, y_: batch_ys, dropout_rate: 0.5, is_training: True})

        print("Step:", step, "| loss:", loss, "| accuracy:", acc)

        with sess.as_default():
            eval_testset()

if save_model: 
    save_path = saver.save(sess, save_path)
    print("Parameters saved as:" + save_path)
