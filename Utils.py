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

def logging_Dbuffer(D):
    logging.info("run_experiment - Number of experience stored: " + str(len(D.buffer)))
    logging.info("run_experiment - example of experience stored: ")
    logging.info("run_experiment - state shape: " + str(D.buffer[0][0].shape))
    logging.info("run_experiment - action: " + str(D.buffer[0][1]))
    logging.info("run_experiment - reward: " + str(D.buffer[0][2]))
    logging.info("run_experiment - state1 shape: " + str(D.buffer[0][3].shape))
    logging.info("run_experiment - terminal: " + str(D.buffer[0][4]))

def readCR():
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

class RTplot:
    '''
    This class alow to plot in real time
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
        pyqtgraph allow to plot in real time not by update the graph
        but by quickly clear and plot again

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
        x, y = readCR()
        self.pw.plot(x, y, clear=True)
        pg.QtGui.QApplication.processEvents()