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
BATCH_SIZE = 32 #How many experiences to use for each training step
UPDATE_FREQ = 4 #How often to perform a training step
GAMMA = .95 #Discount factor on the target Q-values
START_EPSILON = 0.1 #Starting chance of random action
END_EPSILON = 0.0001 #Final chance of random action
ANNEALING_STEPS_EPSILON = 3000000 #How many steps of training to reduce START_EPSILON to END_EPSILON
NB_EPISODES = 10000 #How many steps of random actions before training begins
TAU = 0.001 #Rate to update target network toward primary network
LOAD_MODEL = False #Whether to load a saved model
SAVING_PATH = "./dqn" #The path to save our model to
REPLAY_MEMORY_SIZE = 20000 #Number of previous transitions to remember
UPDATE_NETWORK_STEP_TIME = 1 #Time step to which update the target network with the primary network
#if 0, it update at every step using TAU, if you use a value for UPDATE_NETWORK_TIME you should
#set TAU = 1 to entirely copy the primary network at each update step
DOUBLE_DQN = 'OFF'
NB_STEPS_SAVING_NETWORK = 10000
FRAME_PER_ACTION = 1
GLOBAL_TIMESTEP = 0