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

    def refreshScreen(self):
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
