#-------------------------------------------------------------------------------
# Name:        evalNetwork
# Purpose:     functions used to eval network
#
# Author:      tbeucher
#
# Created:     07/11/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from createNetwork import *
from dql_util import *

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

def getCR(nb_games, path_saved_network):
    '''
    Plays nb_games games and return a list of the cumulative rewards
    obtains during each games
    '''
    mainDQN, targetDQN = create_network()
    mainDQN.epsilon = 0

    with tf.Session() as sess:
        print("Load networks")
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        check_point = tf.train.get_checkpoint_state(path_saved_network)
        saver.restore(sess, check_point.model_checkpoint_path)

        print("Game initialization")
        play_function = getGame()
        action = 1
        s, reward, terminal = play_function(np.array([action,0]))
        s = cv2.cvtColor(cv2.resize(s, (IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED)), cv2.COLOR_BGR2GRAY)
        ret, s = cv2.threshold(s,1,255,cv2.THRESH_BINARY)
        state = np.stack((s, s, s, s), axis=2)

        all_CR = []
        print("Start evaluation")
        for i in range(nb_games):
            print("Start game " + str(i))
            cR = 0
            t = False
            while t != True:
                a = mainDQN.get_action(state, sess, i)
                action = np.eye(NB_ACTIONS)[a]

                s, r, t = play_function(action)
                s = preprocess(s, IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED)
                s1 = np.append(state[:,:,1:], s, axis=2)

                cR += r
                state = s1
            all_CR.append(cR)
        print("End of evaluation")
    return all_CR




