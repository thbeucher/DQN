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
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from dql_util import *
from config_file import *


#deep q networks and beyond
#dqn + experience replay + separate target network and
#prediction network + double dqn + dueling dqn

def run_experiment():
    #Make a path for our model to be saved in
    if not os.path.exists(SAVING_PATH):
        os.makedirs(SAVING_PATH)
    name = "/model"
    if DUELING_DQN == 'ON':
        name += "-DuelingDQN"
    if DOUBLE_DQN == 'ON':
        name += "-DoubleDQN"
    name += "-"
    #initialize Q and Q-Target
    mainDQN = NeuralNetwork_TF(nb_actions=NB_ACTIONS, input_config=INPUT_CONFIG,\
                                layers_types=LAYERS_TYPES, layers_shapes=LAYERS_SHAPES,\
                                layers_activations=LAYERS_ACTIVATIONS,\
                                layers_strides=LAYERS_STRIDES, layers_padding=LAYERS_PADDING,\
                                weights_stddev=WEIGHTS_STDDEV, weights_init=WEIGHTS_INIT,\
                                bias_init=BIAS_INIT, bias_init_value=BIAS_INIT_VALUE,\
                                learning_rate=LEARNING_RATE, dueling_dqn=DUELING_DQN,\
                                batch_size=BATCH_SIZE, update_freq=UPDATE_FREQ,\
                                gamma=GAMMA, start_epsilon=START_EPSILON,\
                                end_epsilon=END_EPSILON, annealing_steps_epsilon=ANNEALING_STEPS_EPSILON,\
                                nb_episodes=NB_EPISODES)
    targetDQN = NeuralNetwork_TF(nb_actions=NB_ACTIONS, input_config=INPUT_CONFIG,\
                                layers_types=LAYERS_TYPES, layers_shapes=LAYERS_SHAPES,\
                                layers_activations=LAYERS_ACTIVATIONS,\
                                layers_strides=LAYERS_STRIDES, layers_padding=LAYERS_PADDING,\
                                weights_stddev=WEIGHTS_STDDEV, weights_init=WEIGHTS_INIT,\
                                bias_init=BIAS_INIT, bias_init_value=BIAS_INIT_VALUE,\
                                learning_rate=LEARNING_RATE, dueling_dqn=DUELING_DQN,\
                                batch_size=BATCH_SIZE, update_freq=UPDATE_FREQ,\
                                gamma=GAMMA, start_epsilon=START_EPSILON,\
                                end_epsilon=END_EPSILON, annealing_steps_epsilon=ANNEALING_STEPS_EPSILON,\
                                nb_episodes=NB_EPISODES)
    #init the replay memory D
    D = Experience_replay(REPLAY_MEMORY_SIZE)
    #create graph to copy main network into target network
    tfTAU = tf.Variable(1., name='TAU') # 1 is passed in order to have mainDQN = targetDQN at first time
    trainables = tf.trainable_variables()
    target_ops = create_updateTargetGraph(trainables, tfTAU)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        #load the model
        if LOAD_MODEL == True:
            check_point = tf.train.get_checkpoint_state(SAVING_PATH)
            saver.restore(sess, check_point.model_checkpoint_path)
        sess.run(tf.initialize_all_variables())
        #set the target network to be equal to the primary network
        logging.info("run_experiment - Init mainDQN and targetDQN to be equal")
        updateTarget(target_ops, sess)
        opT = tfTAU.assign(TAU)
        sess.run(opT)
        #observe initial state s
        logging.info("run_experiment - Initialize game state")
        flappyBird = game.GameState()
        action = 1
        s, reward, terminal = flappyBird.frame_step(np.array([action,0]))
        s = cv2.cvtColor(cv2.resize(s, (IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED)), cv2.COLOR_BGR2GRAY)
        ret, s = cv2.threshold(s,1,255,cv2.THRESH_BINARY)
        state = np.stack((s, s, s, s), axis=2)
        #feed the replay memory D with random experience
        logging.info("run_experiment - Feed the replay memory with random experience")
        for i in range(BATCH_SIZE*100):
            action_index = mainDQN.get_action(state, sess, i)
            print("Feeding of D - action = " + str(action_index))
            action = np.eye(NB_ACTIONS)[action_index]
            s, r, t = flappyBird.frame_step(action)
            s = preprocess(s, IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED)
            s1 = np.append(state[:,:,1:], s, axis=2)
            D.add((state, action_index, r, s1, t))
            state = s1
        logging.info("run_experiment - Number of experience stored: " + str(len(D.buffer)))
        logging.info("run_experiment - example of experience stored: ")
        logging.info("run_experiment - state shape: " + str(D.buffer[0][0].shape))
        logging.info("run_experiment - action: " + str(D.buffer[0][1]))
        logging.info("run_experiment - reward: " + str(D.buffer[0][2]))
        logging.info("run_experiment - state1 shape: " + str(D.buffer[0][3].shape))
        logging.info("run_experiment - terminal: " + str(D.buffer[0][4]))
        #repeat:
        i = 0
        while 1:
            #get action a
            if i%FRAME_PER_ACTION == 0:
                a = mainDQN.get_action(state, sess, i)
            else:
                #do nothing
                a = 0
            action = np.eye(NB_ACTIONS)[a]
            #carry out a and observe reward r and new state s1
            s, r, t = flappyBird.frame_step(action)
            s = preprocess(s, IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED)
            s1 = np.append(state[:,:,1:], s, axis=2)
            #store experience <s,a,r,s1,t> in D
            D.add((state, a, r, s1, t))
            #update every x steps
            if t%UPDATE_FREQ == 0:
                logging.info("run_experiment - run trainDQN")
                trainDQN(DOUBLE_DQN, GAMMA, mainDQN, targetDQN, D, BATCH_SIZE)
            #set s = s1
            state = s1
            #copy network to target network
            if i%UPDATE_NETWORK_STEP_TIME == 0:
                logging.info("run_experiment - mainDQN copied into target network")
                updateTarget(target_ops, sess)
            #decrease epsilon
            if mainDQN.epsilon > END_EPSILON:
                mainDQN.epsilon -= mainDQN.decrease_step_epsilon
                logging.info("run_experiment - epsilon = " + str(mainDQN.epsilon))
            #save networks every x steps
            i += 1
            if i % NB_STEPS_SAVING_NETWORK == 0:
                saver.save(sess, SAVING_PATH + name + str(i) + '.cptk')
                logging.info("run_experiment - network step " + str(i) + "saved")
            logging.info("timestep = " + str(i) + " - action = " + str(a) + " - reward = " + str(r))
            print("timestep = " + str(i) + " - action = " + str(a) + " - reward = " + str(r))

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
                                end_epsilon=END_EPSILON, annealing_steps_epsilon=ANNEALING_STEPS_EPSILON,\
                                nb_episodes=NB_EPISODES)
    network_runningtest(Qnetwork, 0)

logging.basicConfig(filename='dqnLog.log', level=logging.DEBUG)
run_experiment()


