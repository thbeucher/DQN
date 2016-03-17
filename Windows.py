#-------------------------------------------------------------------------------
# Name:        Windows
# Purpose:
#
# Author:      tbeucher
#
# Created:     15/03/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import tkinter as tk


class Window:

    def __init__(self, width = 500, height = 500):
        '''
        Creates a window

        '''
        self.win = tk.Tk()
        self.canv = tk.Canvas(self.win, width = width, height = height)
        self.canv.pack()

    def closeWindow(self):
        '''
        Closes the window

        '''
        self.win.destroy()

    def showAndRefreshScreen(self):
        '''
        Refreshs the screen

        '''
        self.canv.update()

    def drawcircleColor(self, x, y, rad, color = "black"):
        '''
        Draws a circle

        Inputs:
            x: int, absciss of the center
            y: int, ordinate of the center
            rad: int, size of the circle ie the diameter
            color: string, color of the circle

        Ouput:
            circle: int, the number corresponding to the object created for the canvas

        '''
        circle = self.canv.create_oval(x-rad,y-rad,x+rad,y+rad,width=0,fill=color)
        return circle

    def checkOut(self, obj):
        '''
        Checks if the object is out of the windows and put it back

        Input: obj, the object to check

        Output: t, tuple, coordinates of the object after checking

        '''
        coords = self.canv.coords(obj)
        t = ()
        if coords[0] > self.width:
            t = (0, coords[1], abs(coords[2]-coords[0]), coords[3])
            self.canv.coords(obj, t)
        if coords[1] > self.height:
            t = (coords[0], 0, coords[2], abs(coords[1]-coords[3]))
            self.canv.coords(obj, t)
        if coords[2] < 0:
            t = (self.width-abs(coords[2]-coords[0]), coords[1], self.width, coords[3])
            self.canv.coords(obj, t)
        if coords[3] < 0:
            t = (coords[0], self.height-abs(coords[1]-coords[3]), coords[2], self.height)
            self.canv.coords(obj, t)
        return t

    def moveWithCoord(self, obj, newCoords):
        '''
        Moves an object at the given coordinates

        Input:
            obj - int - object to move
            newCoords - tuple - the new coordinates of the object

        Output:
            coord - tuple - coordinates of the new position
            newPosCenter - python list - coordinates of the object center

        '''
        self.canv.coords(obj, newCoords)
        coord = self.checkOut(obj)
        newPosCenter = [(t[2]+t[0])/2, (t[3]+t[1])/2]
        return coord, newPosCenter
