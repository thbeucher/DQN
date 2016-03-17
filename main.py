#-------------------------------------------------------------------------------
# Name:        main
# Purpose:
#
# Author:      tbeucher
#
# Created:     15/03/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from Windows import Window
import time

def main():
    mainWin = Window()
    cir = mainWin.drawcircleColor(50, 50, 10)
    mainWin.showAndRefreshScreen()
    time.sleep(2)
    mainWin.moveWithCoord(cir, (100, 100))
    mainWin.showAndRefreshScreen()
    time.sleep(2)
    mainWin.closeWindow()

if __name__ == '__main__':
    main()
