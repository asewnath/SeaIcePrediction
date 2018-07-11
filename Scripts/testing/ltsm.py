# *eyeroll* here we go... tensorflow time

#LTSM for 3 month prediction ensemble
#Hold out 2015, 2016, 2017

import tensorflow as tf
from data_collection import data_collection
from data_collection import lstm_preprocess

#feat, __ = data_collection()

#feat needs to be processed to be the type of input vectors 
#and ground truth that is needed

feat, gtruth = lstm_preprocess()