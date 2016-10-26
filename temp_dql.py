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
                                bias_init=BIAS_INIT, bias_init_value=BIAS_INIT_VALUE,\
                                learning_rate=LEARNING_RATE)
    #debug for network
    with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            a = np.ones((84,84,4),dtype=np.float32)
            res = sess.run(Qnetwork.output_layer, feed_dict={Qnetwork.network_input:[a]})
            print("conv1", Qnetwork.all_layers[1].get_shape())
            print("conv2", Qnetwork.all_layers[2].get_shape())
            print("conv3", Qnetwork.all_layers[3].get_shape())
            print("fully1", Qnetwork.all_layers[4].get_shape())
            print("output_net: ", res.shape)
            print("output_net_a: ", res[0].shape)
            print("output: ", res)
            print("ouput_a: ", res[0])
    #debug for training method
    Qnetwork.create_training_method()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        state_batch = np.random.rand(2*84*84*4).reshape((2,84,84,4))
        action_batch = np.random.rand(2*2).reshape((2,2))
        y_batch = np.random.rand(2)
        Qnetwork.train_step.run(feed_dict={Qnetwork.y:y_batch, Qnetwork.actions_input:action_batch,\
                                                Qnetwork.network_input:state_batch})


logging.basicConfig(filename='dqnLog.log', level=logging.DEBUG)
experiment()