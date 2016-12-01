#-------------------------------------------------------------------------------
# Name:        experiment
# Purpose:     deep q network and beyond
#              dqn + experience replay + separate target network & prediction
#              network + double dqn + dueling dqn
#
# Author:      thomasbl
#
# Created:     25/10/2016
# Copyright:   (c) thomasbl 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from dql_util import *
from createNetwork import *
from Utils import logging_Dbuffer, RTplot
from PER import PER

import sys
sys.path.append("game/")
import wrapped_flappy_bird as game


def getGame():
    '''
    Initialize game envrionment and return the play function

    return play_function
    '''
    #flappy bird
    flappyBird = game.GameState()
    return flappyBird.frame_step

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
    #Initialize obj to plot cumulative reward during training
    rtplot = RTplot("Time step", "Cumulative rewards")
    #get play_function of the game
    play_function = getGame()
    #initialize Q and Q-Target
    mainDQN, targetDQN = create_network()
    #init the replay memory D
    if PER_ON:
        D = PER(size=REPLAY_MEMORY_SIZE, alpha=PER_ALPHA, beta_zero=PER_BETA_ZERO, batch_size=BATCH_SIZE,\
                nb_segments=NB_SEGMENTS, annealing_beta_steps=ANNEALING_BETA_STEPS)
    else:
        D = Experience_replay(REPLAY_MEMORY_SIZE)
    #create graph to copy main network into target network
    tfTAU = tf.Variable(1., name='TAU') # 1 is passed in order to have mainDQN = targetDQN at first time
    trainables = tf.trainable_variables()
    target_ops = create_updateTargetGraph(trainables, tfTAU)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())

        #load the model
        D_loaded = False
        if LOAD_MODEL:
            D_loaded = loadModelData(SAVING_PATH, sess, saver, D, PER_ON)

        #set the target network to be equal to the primary network
        logging.info("run_experiment - Init mainDQN and targetDQN to be equal")
        updateTarget(target_ops, sess)
        opT = tfTAU.assign(TAU)
        sess.run(opT)

        #observe initial state s
        logging.info("run_experiment - Initialize game state")
        action = 1
        s, reward, terminal = play_function(np.array([action,0]))
        s = cv2.cvtColor(cv2.resize(s, (IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED)), cv2.COLOR_BGR2GRAY)
        ret, s = cv2.threshold(s,1,255,cv2.THRESH_BINARY)
        state = np.stack((s, s, s, s), axis=2)

        #feed the replay memory D with random experience
        if LOAD_MODEL == False or D_loaded == False:
            logging.info("run_experiment - Feed the replay memory with random experience")
            for i in range(REPLAY_MEMORY_SIZE):
                action_index = mainDQN.get_action(state, sess, i)
                print("Feeding of D - step " + str(i) + " - action = " + str(action_index) + " - epsilon = " + str(mainDQN.epsilon))
                action = np.eye(NB_ACTIONS)[action_index]
                s, r, t = play_function(action)
                s = preprocess(s, IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED)
                s1 = np.append(state[:,:,1:], s, axis=2)
                #D.add((state, action_index, r, s1, t))
                D.store((state, action_index, r, s1, t))
                state = s1
            #logging_Dbuffer(D) # doesn't work for PER
            #D.save()
        #cumulative rewards
        cr = 0
        #repeat:
        i = GLOBAL_TIMESTEP
        while 1:
            #get action a
            if i%FRAME_PER_ACTION == 0:
                a = mainDQN.get_action(state, sess, i)
            else:
                #do nothing
                a = 0
            action = np.eye(NB_ACTIONS)[a]

            #carry out a and observe reward r and new state s1
            s, r, t = play_function(action)
            s = preprocess(s, IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED)
            s1 = np.append(state[:,:,1:], s, axis=2)

            #store experience <s,a,r,s1,t> in D
            #D.add((state, a, r, s1, t))
            D.store((state, a, r, s1, t))

            #cumul reward or save it if it's the end of a game
            if r == -1:
                with open("cumulative-rewards.txt", "a") as f:
                    f.write("timeStep = " + str(i) + " - cumulative rewards = " + str(cr) + "\n")
                cr = 0
            else:
                cr += r

            #update every x steps
            if t%UPDATE_FREQ == 0:
                logging.info("run_experiment - run trainDQN")
                trainDQN(DOUBLE_DQN, GAMMA, mainDQN, targetDQN, D, BATCH_SIZE, i, PER_ON)
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
                #save the replay memory D
                if PER_ON:
                    #D.save()
                    donothing = 1
                else:
                    np.save("replayMemory", D.buffer)
            logging.info("timestep = " + str(i) + " - action = " + str(a) + " - reward = " + str(r))
            print("timestep = " + str(i) + " - action = " + str(a) + " - reward = " + str(r) + " - epsilon = " + str(mainDQN.epsilon))

            #plot cumulative reward
            if i % PLOT_TIMESTEP == 0:
                rtplot.plotCRFromFile()


logging.basicConfig(filename='dqnLog.log', level=logging.CRITICAL)
run_experiment()
