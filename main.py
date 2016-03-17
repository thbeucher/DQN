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
    print("la")
    mainWin.showAndRefreshScreen()
    time.sleep(5)
    print("now")
    mainWin.closeWindow()

if __name__ == '__main__':
    main()
