#-------------------------------------------------------------------------------
# Name:        dql_util
# Purpose:
#
# Author:      thomasbl
#
# Created:     25/10/2016
# Copyright:   (c) thomasbl 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import tensorflow as tf
from collections import deque
import random
import cv2
import numpy as np
import logging

class Experience_replay:
    '''
    Experience replay mechanism
    '''
    def __init__(self, size):
        self.buffer = deque()
        self.buffer_size = size

    def add(self, experience):
        '''
        Adds an experience to the buffer
        '''
        #experience = tuple = (s, a, r, s1, terminal)
        self.buffer.append(experience)
        if len(self.buffer) > sekf.size:
            self.buffer.popleft()

    def sample(self, batch_size):
        '''
        Samples x experiences from the buffer
        '''
        return random.sample(self.buffer, batch_size)

def preprocess(frame):
    '''
    Preprocess raw image to 80*80 gray image
    '''
    frame = cv2.cvtColor(cv2.resize(frame, (80,80)), cv2.COLOR_BGR2GRAY)
    retVal, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(frame, (80,80,1))

def updateTargetGraph(tfVars, tau):
    '''
    Creates the graph to update the parameters of the target network with those of
    the primary network

    tfVars = tensorflow variables - primary network must be created before target network
    tau = Rate to update target network toward primary network - float
    if tau = 1, copy the primary network otherwise
    it slowly copy the values of the primary network

    eq : Wt <- tau*W + (1-tau)*Wt
    where Wt = weights of the target network and W = weights of the primary network

    '''
    total_vars = len(tfVars)
    mid_size = int(total_vars/2)
    op_holder = []
    for idx, var in enumerate(tfVars[0:mid_size]):
        op_holder.append(tfVars[idx+mid_size].assign((var.value()*tau) + ((1-tau)*\
                            tfVars[idx+mid_size].value())))
    return op_holder

def updateTarget(op_holder, sess):
    '''
    Updates the target network with values from the primary network

    op_holder - list of tensorflow operation to execute
    sess - tensorflow session
    '''
    for op in op_holder:
        sess.run(op)

class NeuralNetwork_TF:
    def __init__(self,**args):
        '''
        args - python dictionary
            .network
        '''
        self.nb_actions = args['nb_actions']
        self.input_config = args['input_config']
        self.layers_types = args['layers_types']
        self.layers_shapes = args['layers_shapes']
        self.layers_activations = args['layers_activations']
        self.layers_strides = args['layers_strides']
        self.layers_padding = args['layers_padding']
        self.weights_stddev = args['weights_stddev']
        self.weights_init = args['weights_init']
        self.bias_init = args['bias_init']
        self.bias_init_value = args['bias_init_value']
        self.create_network()

    def create_network(self):
        self.input_layer = tf.placeholder(self.input_config[0], shape=self.input_config[1])
        all_layers = []
        all_layers.append(self.input_layer)
        logging.info("class:NeuralNetwork_TF - fn:create_network - Network creation")
        for idx, layer in enumerate(self.layers_types):
            w = self.weight_variable(self.layers_shapes[idx], self.weights_init, self.weights_stddev)
            b = self.bias_variable([self.layers_shapes[idx][-1]], self.bias_init, init_value=self.bias_init_value)
            if layer == 'conv':
                if self.layers_activations[idx] == 'relu':
                    all_layers.append(tf.nn.relu(tf.nn.conv2d(all_layers[idx], w,\
                     strides=self.layers_strides[idx], padding=self.layers_padding) + b))
                else:
                    raise ValueError("Only 'relu' for conv layer is currently available")
            elif layer == 'fullyC':
                if self.layers_activations[idx] == 'relu':
                    prev_layer = tf.reshape(all_layers[idx], [-1, self.layers_shapes[idx][0]])
                    all_layers.append(tf.nn.relu(tf.matmul(prev_layer, w) + b))
                else:
                    raise ValueError("Only 'relu' for fullyC layer is currently available")
            elif layer == 'out_fullyC':
                if self.layers_activations[idx] == 'none':
                    prev_layer = tf.reshape(all_layers[idx], [-1, self.layers_shapes[idx][0]])
                    self.output_layer = tf.matmul(prev_layer, w) + b
                else:
                    raise ValueError("Only 'none' for out_fullyC layer is currently available")
            else:
                raise ValueError("Only 'conv','fullyC','out_fullyC' are currently available")
        logging.info("class:NeuralNetwork_TF - fn:create_network - Network created")

    def weight_variable(self, shape, init_type, stddev):
        if init_type == 'truncated_normal':
            initial = tf.truncated_normal(shape, stddev=stddev)
        else:
            raise ValueError("Only 'truncated_normal' init type is currently available")
        return tf.Variable(initial)

    def bias_variable(self, shape, init_type, init_value = 0):
        if init_type == 'constant':
            initial = tf.constant(init_value, shape=shape)
        else:
            raise ValueError("Only 'constant' init type is currently available")
        return tf.Variable(initial)
