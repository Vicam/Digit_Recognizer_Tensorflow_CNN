# === IMPORT ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

# === SETTINGS ===
LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 2500            
DROPOUT = 0.5
BATCH_SIZE = 50
VALIDATION_SIZE = 2000
IMAGE_TO_DISPLAY = 5

# === DATA PREPARATION ===
data = pd.read_csv('./train.csv')
print('data({0[0]},{0[1]})'.format(data.shape))
#print (data.head())

images = data.iloc[:,1:].values #separate labels and pixels data
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0) #normalize data to range [0:1]
print('images({0[0]},{0[1]})'.format(images.shape))

image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8) #squared image
print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))

def display(img):
    one_image = img.reshape(image_width,image_height)    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)    
display(images[IMAGE_TO_DISPLAY])

labels_flat = data[[0]].values.ravel()
print('labels_flat({0})'.format(len(labels_flat)))

labels_count = np.unique(labels_flat).shape[0]
print('labels_count => {0}'.format(labels_count))

# convert class labels from scalars to one-hot vectors
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)
print('labels({0[0]},{0[1]})'.format(labels.shape))

# split data into training and validation set
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]
train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]
print('train_images({0[0]},{0[1]})'.format(train_images.shape))
print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))

# === TENSORFLOW CNN ===

# initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W): # convolution
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x): #pooling
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
x = tf.placeholder('float', shape=[None, image_size]) #images
y_ = tf.placeholder('float', shape=[None, labels_count]) # labels

# First convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
image = tf.reshape(x, [-1,image_width , image_height,1])
#print (image.get_shape()) # (40000,28,28,1)
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
#print (h_conv1.get_shape()) # (40000, 28, 28, 32)
h_pool1 = max_pool_2x2(h_conv1)
#print (h_pool1.get_shape()) # (40000, 14, 14, 32)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#print (h_conv2.get_shape()) # (40000, 14,14, 64)
h_pool2 = max_pool_2x2(h_conv2)
#print (h_pool2.get_shape()) # (40000, 7, 7, 64)

# Densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # (40000, 7, 7, 64) => (40000, 3136)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#print (h_fc1.get_shape()) # (40000, 1024)

# Dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#print (y.get_shape()) # (40000, 10)

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# optimisation function
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
# prediction function
predict = tf.argmax(y,1)

# === TRAIN ===

#stochastic training
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

def next_batch(batch_size):    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size  
    if index_in_epoch > num_examples:
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]
    
# start TensorFlow session
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

# accuracy visualization
train_accuracies = []
validation_accuracies = []
x_range = []
display_step = 1
for i in range(TRAINING_ITERATIONS):
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                  y_: batch_ys, 
                                                  keep_prob: 1.0})       
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE], 
                                                            y_: validation_labels[0:BATCH_SIZE], 
                                                            keep_prob: 1.0})                                  
            print('validation_accuracy => %.2f for step %d'%(validation_accuracy, i))
            validation_accuracies.append(validation_accuracy)     
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        if i%(display_step*10) == 0 and i:
            display_step *= 10
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
    
# final accuracy on validation set  
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images, 
                                                   y_: validation_labels, 
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)
    plt.plot(x_range, validation_accuracies,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.1, ymin = 0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()
    
# === PREDICTION ===

test_images = pd.read_csv('./test.csv').values
test_images = test_images.astype(np.float)
test_images = np.multiply(test_images, 1.0 / 255.0)
print('test_images({0[0]},{0[1]})'.format(test_images.shape))

# with batches
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], 
                                                                                keep_prob: 1.0})
print('predicted_lables({0})'.format(len(predicted_lables)))
display(test_images[IMAGE_TO_DISPLAY])
print ('predicted_lables[{0}] => {1}'.format(IMAGE_TO_DISPLAY,predicted_lables[IMAGE_TO_DISPLAY]))

# save results
np.savetxt('submission.csv', 
           np.c_[range(1,len(test_images)+1),predicted_lables], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')
