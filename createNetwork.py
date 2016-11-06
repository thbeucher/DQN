#-------------------------------------------------------------------------------
# Name:        createNetwork
# Purpose:
#
# Author:      tbeucher
#
# Created:     04/11/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from config_file import *
from dql_util import NeuralNetwork_TF

def create_network():
    mainDQN = NeuralNetwork_TF(nb_actions=NB_ACTIONS,\
                                input_config=INPUT_CONFIG,\
                                layers_types=LAYERS_TYPES,\
                                layers_shapes=LAYERS_SHAPES,\
                                layers_activations=LAYERS_ACTIVATIONS,\
                                layers_strides=LAYERS_STRIDES,\
                                layers_padding=LAYERS_PADDING,\
                                weights_stddev=WEIGHTS_STDDEV,\
                                weights_init=WEIGHTS_INIT,\
                                bias_init=BIAS_INIT,\
                                bias_init_value=BIAS_INIT_VALUE,\
                                learning_rate=LEARNING_RATE,\
                                dueling_dqn=DUELING_DQN,\
                                batch_size=BATCH_SIZE,\
                                update_freq=UPDATE_FREQ,\
                                gamma=GAMMA,\
                                start_epsilon=START_EPSILON,\
                                end_epsilon=END_EPSILON,\
                                annealing_steps_epsilon=ANNEALING_STEPS_EPSILON)

    targetDQN = NeuralNetwork_TF(nb_actions=NB_ACTIONS,\
                                 input_config=INPUT_CONFIG,\
                                 layers_types=LAYERS_TYPES,\
                                 layers_shapes=LAYERS_SHAPES,\
                                 layers_activations=LAYERS_ACTIVATIONS,\
                                 layers_strides=LAYERS_STRIDES,\
                                 layers_padding=LAYERS_PADDING,\
                                 weights_stddev=WEIGHTS_STDDEV,\
                                 weights_init=WEIGHTS_INIT,\
                                 bias_init=BIAS_INIT,\
                                 bias_init_value=BIAS_INIT_VALUE,\
                                 learning_rate=LEARNING_RATE,\
                                 dueling_dqn=DUELING_DQN,\
                                 batch_size=BATCH_SIZE,\
                                 update_freq=UPDATE_FREQ,\
                                 gamma=GAMMA,\
                                 start_epsilon=START_EPSILON,\
                                 end_epsilon=END_EPSILON,\
                                 annealing_steps_epsilon=ANNEALING_STEPS_EPSILON)

    return mainDQN, targetDQN