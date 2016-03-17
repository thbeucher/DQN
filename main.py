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

import time
import os
import sys

tmp = os.getcwd()
path = tmp[:tmp.rfind("\\")] + "/LIB/"
sys.path.append(path)

from Windows import Window
from Food import Food

def main():
    mainWin = Window()
    cir = mainWin.drawcircleColor(50, 50, 10)
    f = Food(mainWin, 5, 10)
    print(f.lookAtNearestFood(cir))
    mainWin.showAndRefreshScreen()
    mainWin.win.mainloop()
    #mainWin.closeWindow()

if __name__ == '__main__':
    main()
