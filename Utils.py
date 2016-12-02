#-------------------------------------------------------------------------------
# Name:        Utils
# Purpose:
#
# Author:      tbeucher
#
# Created:     21/11/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import pyqtgraph as pg #pyqtgraph install = pip install pyside then pip install pyqtgraph
import logging
import numpy as np

def logging_Dbuffer(D):
    logging.info("run_experiment - Number of experience stored: " + str(len(D.buffer)))
    logging.info("run_experiment - example of experience stored: ")
    logging.info("run_experiment - state shape: " + str(D.buffer[0][0].shape))
    logging.info("run_experiment - action: " + str(D.buffer[0][1]))
    logging.info("run_experiment - reward: " + str(D.buffer[0][2]))
    logging.info("run_experiment - state1 shape: " + str(D.buffer[0][3].shape))
    logging.info("run_experiment - terminal: " + str(D.buffer[0][4]))

def readCR():
    '''
    '''
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

def getMeanCR(x, y):
    '''
    '''
    #if 72154 get 7, if 132451 get 13
    divX = int(x[-1]/(10**(len(str(x[-1]))-2)))
    rX = int(len(x)/divX)
    newX = [x[i:i+rX] for i in range(0, len(x), rX)]
    newY = [y[i:i+rX] for i in range(0, len(y), rX)]
    x1 = [np.mean(xx) for xx in newX]
    y1 = [np.mean(yy) for yy in newY]
    return x1, y1

class RTplot:
    '''
    This class alow to plot in real time

    pyQtGraph allow to plot in real time not by update the figure
    but by quickly clear and plot again
    '''
    def __init__(self, xName, yName):
        '''
        xName - string - abssice name
        yName - string - ordinate name
        '''
        self.pw = pg.plot()
        self.pw.setLabels(left=(yName))
        self.pw.setLabels(bottom=(xName))
        self.x = []
        self.y = []

    def updatePlot(self, newX, newY):
        '''
        Updates the figure

        newX - number - new abssice point
        newY - number - new ordinate point
        '''
        self.x.append(newX)
        self.y.append(newY)
        self.pw.plot(self.x, self.y, clear=True)
        pg.QtGui.QApplication.processEvents()

    def plotCRFromFile(self):
        '''
        Plots the mean cumulative rewards from file
        '''
        #xTmp, yTmp = readCR()
        #x, y = getMeanCR(xTmp, yTmp)
        x, y = readCR()
        self.pw.plot(x, y, clear=True)
        pg.QtGui.QApplication.processEvents()


def dict_to_array(d, dtypeC):
    '''
    Converts a dictionary to a numpy array

    d - dict - key and values must be numbers
    dtypeC - string - f8 or i8 etc
    '''
    dtypes = dict(names=['id', 'data'], formats=[dtypeC, dtypeC]) # f8 = 64-bits floating point
    try:
        return np.fromiter(d.iteritems(), dtype=dtypes, count=len(d))
    except:
        return np.fromiter(d.items(), dtype=dtypes, count=len(d))


class SumTree:
    '''
    '''

    def __init__(self, size):
        self.size = size

        self.tree = np.zeros(2*self.size-1)
        self.data = np.zeros(self.size, dtype=object)





