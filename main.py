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
