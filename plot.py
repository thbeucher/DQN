#-------------------------------------------------------------------------------
# Name:        plot
# Purpose:
#
# Author:      tbeucher
#
# Created:     04/11/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from evalNetwork import getCR
from Utils import readCR

def plotMeanCR():
    x, y = readCR()
    #if 72154 get 7, if 132451 get 13
    divX = int(x[-1]/(10**(len(str(x[-1]))-2)))
    rX = int(len(x)/divX)
    newX = [x[i:i+rX] for i in range(0, len(x), rX)]
    newY = [y[i:i+rX] for i in range(0, len(y), rX)]
    x1 = [np.mean(xx) for xx in newX]
    y1 = [np.mean(yy) for yy in newY]
    plt.plot(x1, y1)
    plt.ylabel('mean cumulative rewards')
    plt.xlabel("time step")
    plt.show()

def plot_cumulativeRewards():
    x, y = readCR()
    plt.plot(x, y)
    plt.ylabel('cumulative rewards')
    plt.xlabel("time step")
    plt.show()

def eval_network_plot_result(nb_games, path_saved_network):
    y = getCR(nb_games, path_saved_network)
    x = list(range(len(y)))
    plt.plot(x, y)
    plt.ylabel('cumulative rewards')
    plt.xlabel("games")
    plt.show()


#plot_cumulativeRewards()
plotMeanCR()
#eval_network_plot_result(50, "/media/thomas/deep//git/DQN/dqn")
