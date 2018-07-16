import numpy as np
import tensorflow as tf
import scipy.io as sio 
from PIL import Image

data_dict   = sio.loadmat('X_train.mat')
train_data  = data_dict['A']
train_data  = train_data.astype(np.float32)

labels_dict = sio.loadmat('Train_Labels.mat')
train_labels = labels_dict['Labels']
train_labels = train_labels.astype(np.float32)

#Preprocess train_data into 5 pixel images centered around each pixel (sliding window)
#or make a loop with a sliding window? I don't know which one is better so I'm 
#just going to try both. Eh. Actually the first one sounds like its going to 
#unnecessarily take up a lot of memory.

batch_size   = 1;
patch_size   = 2;
depth_size   = 1;
image_size   = 5;
num_channels = 3;
num_hidden   = 5;

graph = tf.Graph()

with graph.as_default():
    tf_train_data = tf.placeholder(
            tf.float32, shape = (batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(1,1))

    #Variables
    layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth_size], stddev = 0.1))
    layer1_biases = tf.Variable(tf.zeros([depth_size]))
     
    layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth_size, depth_size], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth_size]))
    
    layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth_size, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    #Model
    def model(data):
        
        #Layer 1
        conv   = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        pool   = tf.nn.max_pooling(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')
        
        #Fully connected Layer
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer2_weights) + layer2_biases)
        
        return tf.matmul(hidden, layer3_weights)
        
    # Training computation.
    logits = model(tf_train_data)
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    #valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    #test_prediction = tf.nn.softmax(model(tf_test_dataset))  
    
    
    num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  
  for step in range(num_steps):
      
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    
    batch_data = train_data[offset:(offset + batch_size), :, :, :]
    
    batch_labels = train_labels[offset:(offset + batch_size), :]
    
    feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels}
    
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    
    
'''    
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
'''    
     
    
    