# *eyeroll* here we go... tensorflow time

#LTSM for 3 month prediction ensemble
#Hold out 2015, 2016, 2017


from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
#import tensorflow as tf
#from data_collection import lstm_preprocess



#This code is being ripped off of http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
#def lstm_model(time_steps, rnn_layers, dense_layers=None):
    
def lstm_model(num_units, rnn_layers, dense_layers=None, learning_rate=0.1, optimizer='Adagrad'):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param num_units: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         ^That is so stupid and ambiguous, you don't completely know what's going on.
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """

    def lstm_cells(layers): #this uses rnn_layer, which, as said above, is a list of int
                
        
        if isinstance(layers[0], dict): #Check if this is a dict, which it will not be.
            
            #If the input to the function is a dictionary, then return a model based on whether keep_prob is true or not
            #Dropout wrapper: operator adding dropout to inputs and outputs of the given *cell*
            
            #I have no idea what the hell 'keep_prob' is supposed to mean
            #Right now they are initializing a LSTM cell with a number of units? Apparently the size of the cell but that's so ambiguous 
            return [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(layer['num_units'], state_is_tuple=True),layer['keep_prob']) 
                if layer.get('keep_prob') else tf.contrib.rnn.BasicLSTMCell(layer['num_units'],state_is_tuple=True) for layer in layers]
            
            
        #In this case,    
        return [tf.contrib.rnn.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = tf.unstack(X, axis=1, num=num_units)
        output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = dnn_layers(output[-1], dense_layers)
        prediction, loss = tflearn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model












#feat needs to be processed to be the type of input vectors 
#and ground truth that is needed
'''
feat, gtruth = lstm_preprocess()

time_steps = 3
batch_size = 200
num_features = 18
'''

#input_vect = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])

'''
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

hidden_state = tf.zeros([batch_size, lstm.state_size])
current_state = tf.zeros([batch_size, lstm.state_size])
state = hidden_state, current_state
probabilities = []
loss = 0.0


for input_vector in data:
    output, state = lstm(input_vector, state)
'''
'''
data = tf.placeholder(tf.float32, [batch_size, time_steps])
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

initial_state = state = tf.zeros([batch_size, lstm.state_size])
    
for i in range(time_steps):
    output, state = lstm(data[:,i], state)
    
final_state = state


np_state = initial_state.eval()
total_loss = 0.0
for input_vect in data:
    np_state, current_loss = session.run([final_state, loss],
        feed_dict={initial_state:np_state, input_vect: data})    
    total_loss = total_loss + current_loss
'''    