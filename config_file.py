#-------------------------------------------------------------------------------
# Name:        config_file
# Purpose:     list all configuration variables
#
# Author:      tbeucher
#
# Created:     25/10/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

DUELING_DQN = 'OFF' #could be ON or OFF
NB_ACTIONS = 2 #number of possible action in the environment
INPUT_CONFIG = ["float", [None, 84, 84, 4]] #type and shape for the input layer
LAYERS_TYPES = ['conv', 'conv', 'conv', 'fullyC', 'out_fullyC']
#layers_shapes must contain the shape of the last conv output in last shape if dueling_dqn = 'ON'
#LAYERS_SHAPES = [[8,8,4,32], [4,4,32,64], [3,3,64,64], [3136,512], 512]
LAYERS_SHAPES = [[8,8,4,32], [4,4,32,64], [3,3,64,64], [3136,512], [512, NB_ACTIONS]]
#layers_activations must contain the activation type of advantageLayer and valueLayer if dueling_dqn = 'ON'
#LAYERS_ACTIVATIONS = ['relu', 'relu', 'relu', 'relu']
LAYERS_ACTIVATIONS = ['relu', 'relu', 'relu', 'relu', 'none']
LAYERS_STRIDES = [[1,4,4,1], [1,2,2,1], [1,1,1,1], 'none', 'none']
LAYERS_PADDING = 'VALID' #type of convolution, could be VALID or SAME
WEIGHTS_STDDEV = 0.01
WEIGHTS_INIT = 'truncated_normal'
BIAS_INIT = 'constant'
BIAS_INIT_VALUE = 0.01
LEARNING_RATE = 0.000001
IMAGE_WIDTH_RESIZED = 84 #image width after resizing
IMAGE_HEIGHT_RESIZED = 84 #image height after resizing