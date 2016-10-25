#-------------------------------------------------------------------------------
# Name:        temp_dql
# Purpose:
#
# Author:      thomasbl
#
# Created:     25/10/2016
# Copyright:   (c) thomasbl 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from dql_util import *
from config_file import *

#deep q networks and beyond
#dqn + experience replay + separate target network and
#prediction network + double dqn + dueling dqn

def experiment():
    Qnetwork = NeuralNetwork_TF(nb_actions=NB_ACTIONS, input_config=INPUT_CONFIG,\
                                layers_types=LAYERS_TYPES, layers_shapes=LAYERS_SHAPES,\
                                layers_activations=LAYERS_ACTIVATIONS,\
                                layers_strides=LAYERS_STRIDES, layers_padding=LAYERS_PADDING,\
                                weights_stddev=WEIGHTS_STDDEV, weights_init=WEIGHTS_INIT,\
                                bias_init=BIAS_INIT, bias_init_value=BIAS_INIT_VALUE)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        a = np.ones((1,80,80,4),dtype=np.float32)
        print(a.dtype)
        print(Qnetwork.input_layer.dtype)
        sess.run(Qnetwork.output_layer, feed_dict={Qnetwork.input_layer:a})
        print("input: ", Qnetwork.input_layer.eval().shape)
        print("input: ", Qnetwork.output_layer.eval().shape)


logging.basicConfig(filename='dqnLog.log', level=logging.DEBUG)
experiment()