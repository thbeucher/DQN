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
from Utils import readParams
from ANN import MLP

def main():
    params = readParams(os.getcwd() + "/parameters.txt")
    print(params['nonbhl'])
    rn = MLP(2, 1, 1, params['nonbhl'], 0)
    print(rn.ANN)
    print(rn.update([0, 1]))

if __name__ == '__main__':
    main()
