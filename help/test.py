#-------------------------------------------------------------------------------
# Name:        test
# Purpose:
#
# Author:      tbeucher
#
# Created:     04/11/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath('..'))
from dql_util import *
from createNetwork import *

def network_runningtest(Qnetwork, a):
    if a == 0:
        #debug for network
        with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                a = np.ones((5,84,84,4),dtype=np.float32)
                res = sess.run(Qnetwork.output_layer, feed_dict={Qnetwork.network_input:a})
                print("conv1 shape: " + str(Qnetwork.all_layers[1].get_shape()))
                print("conv2 shape: " + str(Qnetwork.all_layers[2].get_shape()))
                print("conv3 shape: " + str(Qnetwork.all_layers[3].get_shape()))
                print("fully1 shape: " + str(Qnetwork.all_layers[4].get_shape()))
                print("output_net: ", res.shape)
                print("output_net_a: ", res[0].shape)
                print("output: ", res)
                print("ouput_a: ", res[0])
                res[2,1] = 0.5
                res[3,0] = 0.5
                print(res)
                argmax = tf.argmax(res, 1)
                print("argmax: ", argmax.eval())
    else:
        #debug for dueling dqn
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            a = np.ones((84,84,4),dtype=np.float32)
            res = sess.run(Qnetwork.output_layer, feed_dict={Qnetwork.network_input:[a]})
            print("conv1 shape: " + str(Qnetwork.all_layers[1].get_shape()))
            print("conv2 shape: " + str(Qnetwork.all_layers[2].get_shape()))
            print("conv3 shape: " + str(Qnetwork.all_layers[3].get_shape()))
            print("output_net: ", res.shape)
            print("output_net_a: ", res[0].shape)
            print("output: ", res)
            print("ouput_a: ", res[0])

def trainingMethod_runningtest(Qnetwork):
    #debug for training method
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        state_batch = np.random.rand(2*84*84*4).reshape((2,84,84,4))
        action_batch = np.random.rand(2*2).reshape((2,2))
        y_batch = np.random.rand(2)
        Qnetwork.train_step.run(feed_dict={Qnetwork.y:y_batch, Qnetwork.actions_input:action_batch,\
                                                Qnetwork.network_input:state_batch})

def experiment():
    Qnetwork = NeuralNetwork_TF(nb_actions=NB_ACTIONS, input_config=INPUT_CONFIG,\
                                layers_types=LAYERS_TYPES, layers_shapes=LAYERS_SHAPES,\
                                layers_activations=LAYERS_ACTIVATIONS,\
                                layers_strides=LAYERS_STRIDES, layers_padding=LAYERS_PADDING,\
                                weights_stddev=WEIGHTS_STDDEV, weights_init=WEIGHTS_INIT,\
                                bias_init=BIAS_INIT, bias_init_value=BIAS_INIT_VALUE,\
                                learning_rate=LEARNING_RATE, dueling_dqn=DUELING_DQN,\
                                batch_size=BATCH_SIZE, update_freq=UPDATE_FREQ,\
                                gamma=GAMMA, start_epsilon=START_EPSILON,\
                                end_epsilon=END_EPSILON, annealing_steps_epsilon=ANNEALING_STEPS_EPSILON)
    network_runningtest(Qnetwork, 0)


def test_updateGraph():
    nn1, nn2 = create_network()
    tfTAU = tf.Variable(1., name='TAU') #add of one more variable to see if the trainable graph is changed
    trainables = tf.trainable_variables()
    target_ops = create_updateTargetGraph(trainables, 1.)
    input_test = np.ones((1, 84, 84, 4))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print("nn1 ouput: ", nn1.output_layer.eval(feed_dict={nn1.network_input:input_test}))
        print("nn2 ouput: ", nn2.output_layer.eval(feed_dict={nn2.network_input:input_test}))
        print("update nn2 with nn1")
        updateTarget(target_ops, sess)
        print("nn1 ouput: ", nn1.output_layer.eval(feed_dict={nn1.network_input:input_test}))
        print("nn2 ouput: ", nn2.output_layer.eval(feed_dict={nn2.network_input:input_test}))

#test_updateGraph()

def test_loadingGraph():
    nn1, nn2 = create_network()
    input_test = np.ones((1,84,84,4))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        print("nn1 ouput: ", nn1.output_layer.eval(feed_dict={nn1.network_input:input_test}))
        print("save model")
        saver.save(sess, "net-test.cptk")

def test_loadingGraph2():
    nn1, nn2 = create_network()
    input_test = np.ones((1,84,84,4))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        print("nn1 ouput: ", nn1.output_layer.eval(feed_dict={nn1.network_input:input_test}))
        print("load model")
        check_point = tf.train.get_checkpoint_state(os.getcwd())
        saver.restore(sess, check_point.model_checkpoint_path)
        print("nn1 reloaded ouput: ", nn1.output_layer.eval(feed_dict={nn1.network_input:input_test}))

#test_loadingGraph()
#test_loadingGraph2()



