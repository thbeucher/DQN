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
import os

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
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()

    def sample(self, batch_size):
        '''
        Samples x experiences from the buffer
        '''
        return random.sample(self.buffer, batch_size)

def preprocess(frame, image_width_resized, image_height_resized):
    '''
    Preprocess raw image to 84*84 gray image
    '''
    frame = cv2.cvtColor(cv2.resize(frame, (image_width_resized,image_height_resized)), cv2.COLOR_BGR2GRAY)
    retVal, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(frame, (image_width_resized,image_height_resized,1))

def create_updateTargetGraph(tfVars, tau):
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


def trainDQN(doubleDQN, gamma, mainDQN, targetDQN, replay_memory, batch_size, i):
    '''
    Performs double dqn update or simple update
    1 - get action from the mainDQN network
    2 - get Q-values from the targetDQN network
    '''
    #sample training batch from D
    minibatch = replay_memory.sample(batch_size)
    ss = [d[0] for d in minibatch]
    aa = [d[1] for d in minibatch]
    rr = [d[2] for d in minibatch]
    ss1 = [d[3] for d in minibatch]
    tt = [d[4] for d in minibatch]
    #calculate y=target for each minibatch
        #if s1 is terminal ie t=True then target = r
        #otherwise y=target = r + gamma * max Q-target
    #terminal_or_not_multiplier allows to do the if otherwise in comment above
    terminal_or_not_multiplier = -(np.asarray(tt) - 1)
    Q_values = targetDQN.output_layer.eval(feed_dict={targetDQN.network_input:ss1})
    if doubleDQN == 'ON':
        logging.info("trainDQN - double DQN update performed")
        Q_a = mainDQN.output_layer.eval(feed_dict={mainDQN.network_input:ss1})
        action = np.argmax(Q_a, 1)
        Q_batch = Q_values[range(batch_size), action]
    else:
        logging.info("trainDQN - simple update performed")
        Q_batch = np.max(Q_values, 1)
    targetQ = rr + (gamma * Q_batch * terminal_or_not_multiplier)
    #save cost
    #if i%1000 == 0:
        #cost = mainDQN.cost.eval(feed_dict={mainDQN.y:targetQ, mainDQN.actions:aa, mainDQN.network_input:ss})
        #with open("saved-cost.txt", "a") as f:
            #f.write("timeStep = " + str(i) + " - cost = " + str(cost) + "\n")
    #train network with state_batch, action_batch and y
    mainDQN.train_step.run(feed_dict={mainDQN.y:targetQ, mainDQN.actions:aa, mainDQN.network_input:ss})

class NeuralNetwork_TF:
    def __init__(self,**args):
        '''
        args - python dictionary
            .network
        '''
        self.dueling_dqn = args['dueling_dqn']
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
        self.learning_rate = args['learning_rate']
        self.update_freq = args['update_freq']
        self.gamma = args['gamma']
        self.start_epsilon = args['start_epsilon']
        self.end_epsilon = args['end_epsilon']
        self.annealing_steps_epsilon = args['annealing_steps_epsilon']

        self.create_network()
        self.create_training_method()

        #Set decrease step
        self.epsilon = self.start_epsilon
        self.decrease_step_epsilon = (self.start_epsilon - self.end_epsilon)/self.annealing_steps_epsilon

    def create_network(self):
        self.network_input = tf.placeholder(self.input_config[0], shape=self.input_config[1])
        self.all_layers = []
        self.all_layers.append(self.network_input)
        logging.info("class:NeuralNetwork_TF - fn:create_network - Network creation")
        for idx, layer in enumerate(self.layers_types):
            if (self.dueling_dqn == 'ON' and layer == 'conv') or self.dueling_dqn == 'OFF':
                w = self.weight_variable(self.layers_shapes[idx], self.weights_init, self.weights_stddev)
                b = self.bias_variable([self.layers_shapes[idx][-1]], self.bias_init, init_value=self.bias_init_value)
                if layer == 'conv':
                    if self.layers_activations[idx] == 'relu':
                        logging.info("Layer " + str(idx) + " , prev_layer shape: " + str(self.all_layers[idx].get_shape()))
                        self.all_layers.append(tf.nn.relu(tf.nn.conv2d(self.all_layers[idx], w,\
                         strides=self.layers_strides[idx], padding=self.layers_padding) + b))
                    else:
                        raise ValueError("Only 'relu' for conv layer is currently available")
                elif layer == 'fullyC':
                    if self.layers_activations[idx] == 'relu':
                        logging.info("Layer " + str(idx) + " , prev_layer shape: " + str(self.all_layers[idx].get_shape()))
                        prev_layer = tf.reshape(self.all_layers[idx], [-1, self.layers_shapes[idx][0]])
                        self.all_layers.append(tf.nn.relu(tf.matmul(prev_layer, w) + b))
                    else:
                        raise ValueError("Only 'relu' for fullyC layer is currently available")
                elif layer == 'out_fullyC':
                    if self.layers_activations[idx] == 'none':
                        logging.info("Layer " + str(idx) + " , prev_layer shape: " + str(self.all_layers[idx].get_shape()))
                        prev_layer = tf.reshape(self.all_layers[idx], [-1, self.layers_shapes[idx][0]])
                        self.output_layer = tf.matmul(prev_layer, w) + b
                    else:
                        raise ValueError("Only 'none' for out_fullyC layer is currently available")
                else:
                    raise ValueError("Only 'conv','fullyC','out_fullyC' are currently available")
            elif self.dueling_dqn == 'ON' and layer != 'conv':
                break
            else:
                raise TypeError("Potential error during network construction")
        if self.dueling_dqn == 'ON':
            logging.info("Dueling DQN - last conv shape: " + str(self.layers_shapes[-1]))
            prev_layer = tf.contrib.layers.flatten(self.all_layers[-1])
            logging.info("Dueling DQN - prev_layer shape: " + str(prev_layer.get_shape()))
            w_advantageHiddenLayer = self.weight_variable(self.layers_shapes[-2], self.weights_init, self.weights_stddev)
            b_advantageHiddenLayer = self.bias_variable([self.layers_shapes[-2][-1]], self.bias_init, init_value=self.bias_init_value)
            logging.info("Dueling DQN - w_advantageHiddenLayer shape: " + str(w_advantageHiddenLayer.get_shape()))
            logging.info("Dueling DQN - b_advantageHiddenLayer shape: " + str(b_advantageHiddenLayer.get_shape()))
            w_valueHiddenLayer = self.weight_variable(self.layers_shapes[-2], self.weights_init, self.weights_stddev)
            b_valueHiddenLayer = self.bias_variable([self.layers_shapes[-2][-1]], self.bias_init, init_value=self.bias_init_value)
            logging.info("Dueling DQN - w_valueHiddenLayer shape: " + str(w_valueHiddenLayer.get_shape()))
            logging.info("Dueling DQN - b_valueHiddenLayer shape: " + str(b_valueHiddenLayer.get_shape()))
            w_advantageLayer = self.weight_variable([self.layers_shapes[-1], self.nb_actions], self.weights_init, self.weights_stddev)
            b_advantageLayer = self.bias_variable([self.nb_actions], self.bias_init, init_value=self.bias_init_value)
            logging.info("Dueling DQN - w_advantageLayer shape: " + str(w_advantageLayer.get_shape()))
            logging.info("Dueling DQN - b_advantageLayer shape: " + str(b_advantageLayer.get_shape()))
            w_valueLayer = self.weight_variable([self.layers_shapes[-1], 1], self.weights_init, self.weights_stddev)
            b_valueLayer = self.bias_variable([1], self.bias_init, init_value=self.bias_init_value)
            logging.info("Dueling DQN - w_valueLayer shape: " + str(w_valueLayer.get_shape()))
            logging.info("Dueling DQN - b_valueLayer shape: " + str(b_valueLayer.get_shape()))
            if self.layers_activations[-1] == 'relu':
                advantageHiddenLayer = tf.nn.relu(tf.matmul(prev_layer, w_advantageHiddenLayer) + b_advantageHiddenLayer)
                valueHiddenLayer = tf.nn.relu(tf.matmul(prev_layer, w_valueHiddenLayer) + b_valueHiddenLayer)
                logging.info("Dueling DQN - advantageHiddenLayer shape: " + str(advantageHiddenLayer.get_shape()))
                logging.info("Dueling DQN - valueHiddenLayer shape: " + str(valueHiddenLayer.get_shape()))
                advantageLayer = tf.matmul(advantageHiddenLayer, w_advantageLayer) + b_advantageLayer
                valueLayer = tf.matmul(valueHiddenLayer, w_valueLayer) + b_valueLayer
                logging.info("Dueling DQN - advantageLayer shape: " + str(advantageLayer.get_shape()))
                logging.info("Dueling DQN - valueLayer shape: " + str(valueLayer.get_shape()))
            else:
                raise ValueError("Only 'relu' for actionLayer and valueLayer is currently available")
            #combine advantageLayer and valueLayer to obtain final Q-values
            #Q = V + (A - mean(A))
            self.output_layer = valueLayer + tf.sub(advantageLayer, tf.reduce_mean(advantageLayer, reduction_indices=1, keep_dims=True))
        logging.info("class:NeuralNetwork_TF - fn:create_network - Network created")

    def create_training_method(self):
        #when we feed actions_input, it must be a one_hot vector if it could be
        #otherwise maybe change self.actions_input by:
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_input = tf.one_hot(self.actions, self.nb_actions, dtype=tf.float32)
        #self.actions_input = tf.placeholder("float", [None, self.nb_actions])
        self.y = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.mul(self.output_layer, self.actions_input), reduction_indices=1)
        logging.info("create_training_method - Q_action shape: " + str(Q_action.get_shape()))
        self.cost = tf.reduce_mean(tf.square(self.y - Q_action))
        logging.info("create_training_method - cost shape: " + str(self.cost.get_shape()))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def get_action(self, states, sess, i):
        '''
        Chooses an action follow e-greedy policy

        states - python list or numpy array
        sess - tensorflow session

        return action - int
        '''
        logging.info("get_action - epsilon = " + str(self.epsilon))
        if np.random.rand(1) <= self.epsilon:
            action = np.random.randint(0, self.nb_actions)
            logging.info("get_action - timeStep " + str(i) + " - random action choosed: " + str(action))
        else:
            Q_values = sess.run(self.output_layer, feed_dict={self.network_input:[states]})
            action = np.argmax(Q_values, 1)[0]
            logging.info("get_action - timeStep " + str(i) + " - best action choosed: " + str(action))
        return action

    def weight_variable(self, shape, init_type, stddev):
        if init_type == 'truncated_normal':
            initial = tf.truncated_normal(shape, stddev=stddev)
        elif init_type == 'random_normal':
            initial = tf.random_normal(shape, stddev=stddev)
        else:
            raise ValueError("Only 'truncated_normal', 'random_normal' init type is currently available")
        return tf.Variable(initial)

    def bias_variable(self, shape, init_type, init_value = 0):
        if init_type == 'constant':
            initial = tf.constant(init_value, shape=shape)
        else:
            raise ValueError("Only 'constant' init type is currently available")
        return tf.Variable(initial)

