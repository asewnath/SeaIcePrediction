

import numpy as np
import tensorflow as tf
from cnn_functions import create_input
from cnn_functions import shuffle_input

#Control Panel:
startYear = 1985
stopYear  = 2012
#month = 6
#year  = 1995
imageSize = 11
numForecast = 5
numChannels = numForecast+1


tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    
    # Input Layer
    inputLayer = tf.reshape(features["x"], [-1, imageSize, imageSize, numChannels])
    #may not need this considering there's no dictionary
    #still need to make it a tf variable somehow
    
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs = inputLayer,
            filters = 32,
            kernel_size = [5,5],
            padding = "same", #check
            activation = tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size = [2,2],
                                    strides = 2)
    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 64,
            kernel_size = [5,5],
            padding = "same", #check
            activation = tf.nn.relu)
    
    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2,2],
                                    strides = 2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, imageSize//4 * imageSize//4 * 64])
    dense = tf.layers.dense(inputs = pool2_flat,
                            units = 1024,
                            activation = tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,
                                rate = 0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # Regression Layer
    regPredictions = tf.layers.dense(inputs=dropout, units=1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=regPredictions)
    
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=regPredictions)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        trainOp = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=trainOp)
    
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "mse": tf.metrics.mean_square_error(
                    labels=labels,
                    predictions = regPredictions)}
            
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)          


def main(unused_argv):
    #define train_data
    
    for year in range(startYear, stopYear+1):
        for month in range(0, 11):
    
            data, groundTruth = create_input(month, year, numForecast, imageSize)
            data, labels = shuffle_input(data, groundTruth)
        
            #make resolution a hyperparameter
            data = np.reshape(data, (3249, numChannels, imageSize, imageSize)) #don't hardcode this number.
            labels = np.reshape(labels, (3249, 1))
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
                    batch_size=100,
                    num_epochs=None,
                    shuffle=True)
            classifier.train(
                    input_fn = train_input_fn,
                    steps=20000)


            '''
            # Evaluate model
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x":eval_data},
                    y=eval_labels,
                    num_epochs=1,
                    shuffle.false)
            eval_results = classifier.evaluate(input_fn=eval_input_fn)
            '''
    
if __name__ == "__main__":
    tf.app.run()

  
    