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

def plot_cumulativeRewards():
    x, y = readData()
    plt.plot(x, y)
    plt.ylabel('cumulative rewards')
    plt.xlabel("time step")
    plt.show()


plot_cumulativeRewards()