#-------------------------------------------------------------------------------
# Name:        Food
# Purpose:
#
# Author:      tbeucher
#
# Created:     10/11/2015
# Copyright:   (c) tbeucher 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
from math import sqrt as msqrt

class Food:

    def __init__(self, gameWindow, nbFood, sizeFood, colorFood = "green"):
        '''
        Initializes the class object

        Input:
            gameWindow - window class object
            nbFood - int - number of food object wanted
            sizeFood - int - size of the food
            colorFood - string - color of the food

        '''
        #set the tkinter object
        self.f = gameWindow
        #the number of food in the windows
        self.nbFood = nbFood
        #size of the food
        self.size = sizeFood
        #color of the food
        self.color = colorFood
        self.createFood()

    def createFood(self):
        coords = np.random.random_integers(0, self.f.width, [self.nbFood,2])
        self.foodList = [self.f.drawcircleColor(el[0], el[1], self.size, self.color) for el in coords]

    def getFoodList(self):
        '''
        Return the list of food object

        Output:
            foodList - python list - list of food

        '''
        return self.foodList

    def deleteFood(self, foodToDelete):
        '''
        Delete the given food object

        Input:
            foodToDelete - int - food to delete

        '''
        self.f.canv.delete(foodToDelete)
        self.foodList.remove(foodToDelete)

    def setFoodList(self, foodList):
        self.foodList = foodList

    def getNbFood(self):
        return len(self.foodList)

    def checkIfFoodEated(self, predator):
        '''
        Cheks if the predator eat food

        Input:
            predator - int -

        '''
        coordPredator = self.f.canv.coords(predator)
        for el in self.foodList:
            coordFood = self.f.canv.coords(el)
            if(coordFood[0]>=coordPredator[0] and coordFood[1]>=coordPredator[1] and coordFood[2]<=coordPredator[2] and coordFood[3]<=coordPredator[3]):
                self.deleteFood(el)
                #creates new food
                newCoords = np.random.random_integers(0, self.f.width, [1, 2])
                self.foodList.append(self.f.drawcircleColor(newCoords[0,0], newCoords[0,1], self.size, self.color))


    def lookAtNearestFood(self, individu):
        '''
        finds the nearest food and return the vector pointing at it

        Input:
            individu - int - id of the object to link it to the window

        Output:
            lookAtNearestFood - python list - coordinates x and y of the nearest food

        '''
        coordPredator = self.f.canv.coords(individu)
        coordsFood = [self.f.canv.coords(el) for el in self.foodList]
        dist = [msqrt((el[0] - coordPredator[0])**2 + (el[1] - coordPredator[1])**2) for el in coordsFood]
        ion = dist.index(min(dist))
        xCenterFood = coordsFood[ion][2] - (coordsFood[ion][2] - coordsFood[ion][0])/2
        yCenterFood = coordsFood[ion][3] - (coordsFood[ion][3] - coordsFood[ion][1])/2
        xCenterPred = coordPredator[2] - (coordPredator[2] - coordPredator[0])/2
        yCenterPred = coordPredator[3] - (coordPredator[3] - coordPredator[1])/2
        lookAtNearestFood = [xCenterFood - xCenterPred, yCenterFood - yCenterPred]
        return lookAtNearestFood
