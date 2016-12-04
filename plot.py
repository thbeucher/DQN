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
from Utils import readCR, getMeanCR

def plotMeanCR():
	x, y = readCR("cumulative-rewards.txt")
	x1, y1 = getMeanCR(x, y)
	plt.plot(x1, y1)
	plt.ylabel('mean cumulative rewards')
	plt.xlabel("time step")
	plt.show()

def plot_cumulativeRewards():
	x, y = readCR("cumulative-rewards.txt")
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

	#compute mean and save it
	meanEval = np.mean(y)
	name = "meanCR-" + str(nb_games) + "games"
	with open(name, "w") as f:
		f.write(str(meanEval))
		
def plotAllMCR():
	x, y = readCR("resStandard/cumulative-rewards.txt")
	xMean, yMean = getMeanCR(x, y)
	xDUELING, yDUELING = readCR("resDUELING/cumulative-rewards.txt")
	xMeanDUELING, yMeanDUELING = getMeanCR(xDUELING, yDUELING)
	xDDQN, yDDQN = readCR("resDDQN/cumulative-rewards.txt")
	xDDQN_DUELING, yDDQN_DUELING = readCR("resDDQN_DUELING/cumulative-rewards.txt")
	xMeanDDQN, yMeanDDQN = getMeanCR(xDDQN, yDDQN)
	xMeanDDQN_DUELING, yMeanDDQN_DUELING = getMeanCR(xDDQN_DUELING, yDDQN_DUELING)
	dqn, = plt.plot(xMean, yMean, label='dqn')
	dqnDueling, = plt.plot(xMeanDUELING, yMeanDUELING, label='dqn_dueling')
	ddqn, = plt.plot(xMeanDDQN, yMeanDDQN, label='ddqn')
	ddqn_dueling, = plt.plot(xMeanDDQN_DUELING, yMeanDDQN_DUELING, label='ddqn_dueling')
	plt.legend(handles=[dqn, dqnDueling, ddqn, ddqn_dueling])
	plt.ylabel('mean cumulative rewards')
	plt.xlabel("time step")
	plt.show()


#plot_cumulativeRewards()
plotMeanCR()
#eval_network_plot_result(50, "/media/thomas/deep//git/DQN/dqn")
#plotAllMCR()
