"""
Using Keras will be more straightforward than trying to code Tensorflow, therefore
Keras is recommended for prototyping
"""

import numpy as np
import tensorflow as tf
from cnn_functions import create_input
from cnn_functions import shuffle_input

#Control Panel:
startYear = 1985
stopYear  = 2012
imageSize = 11
numForecast = 5
numChannels = numForecast+1
resolution = 100
batchSize = 50

tf.logging.set_verbosity(tf.logging.INFO)

'''RUNNING INTO WARM START ISSUES'''

def cnn_model_fn(features, labels, mode):
    
    # Input Layer
    inputLayer = tf.reshape(features["x"], [-1, imageSize, imageSize, numChannels])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs = inputLayer,
            filters = 32,
            kernel_size = [4,4],
            padding = "valid", #check
            activation = tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size = [2,2],
                                    strides = 2)
    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 64,
            kernel_size = [3,3],
            padding = "same", #check
            activation = tf.nn.relu)
    
    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2,2],
                                    strides = 2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [batchSize, -1])
    dense = tf.layers.dense(inputs = pool2_flat,
                            units = 1024,
                            activation = tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs=dense,
                                rate = 0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN) 
    # Regression Layer
    regPredictions = tf.layers.dense(inputs=dropout, units=1)
    
    
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=regPredictions)
    
    # Calculate Loss
    loss = tf.losses.mean_squared_error(labels=labels, predictions=regPredictions)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        trainOp = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=trainOp)      

def main(unused_argv):
    
    for year in range(startYear, stopYear+1):
        for month in range(0, 11):
    
            data, groundTruth, size = create_input(month, year, numForecast, imageSize, resolution)
            data, labels = shuffle_input(data, groundTruth)
        
            data = np.reshape(data, (size, numChannels, imageSize, imageSize))
            labels = np.reshape(labels, (size, 1))
            data = np.float32(data)
        
            if((year != startYear) and (month != 0)):
                ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from="./cnn")
                classifier = tf.estimator.Estimator(
                        model_fn=cnn_model_fn, model_dir="./cnn",
                        warm_start_from=ws)
            else:
                classifier = tf.estimator.Estimator(
                        model_fn=cnn_model_fn, model_dir="./cnn")               
            
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": data},
                    y=labels,
                    batch_size=batchSize,
                    num_epochs=None,
                    shuffle=True)
            classifier.train(
                    input_fn = train_input_fn,
                    steps=size//batchSize)


if __name__ == "__main__":   
   tf.app.run()


  
    