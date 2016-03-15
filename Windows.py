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
