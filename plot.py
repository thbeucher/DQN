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

def readData():
    raw_data = []
    with open("cumulative-rewards.txt", "r") as f:
        tmp = f.readlines()
        for el in tmp:
            el = el.rstrip("\n")
            raw_data.append(el)
    #get x and y points
    x, y = [], []
    for el in raw_data:
        tmp = el.split("=")
        y.append(float(tmp[2]))
        tmp2 = tmp[1].split("-")
        x.append(int(tmp2[0]))
    return x, y

def plotMeanCR():
    x, y = readData()
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
    x, y = readData()
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
#plotMeanCR()
eval_network_plot_result(50, "/media/thomas/deep//git/DQN/dqn")
