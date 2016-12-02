#-------------------------------------------------------------------------------
# Name:        test_perf
# Purpose:
#
# Author:      tbeucher
#
# Created:     21/11/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#--------------------------------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath('..'))
import time
from PER_NP import PER_NP
from PER import PER
import numpy as np
##sys.path.append("/media/thomas/deep/git/prioritized-experience-replay-master/")
##import rank_based

def evalPerfSample():
    '''
    '''
    per = PER(size=1000, alpha=0.7, beta_zero=0.5, batch_size=4, nb_segments=4, annealing_beta_steps=15)
##    per2 = PER_NP(size=1000, alpha=0.7, beta_zero=0.5, batch_size=4, nb_segments=4, annealing_beta_steps=15)
    per3 = PER(size=1000, alpha=0.7, beta_zero=0.5, batch_size=4, nb_segments=4, annealing_beta_steps=15)
##    per4 = rank_based.Experience({'size': 1000,'learn_start': 10,'partition_num': 4,'total_step': 100,'batch_size': 4})
    for i in range(1000):
        per.add((np.ones((84,84)), 1, 1, np.ones((84,84)), True))
##        per2.add((np.ones((84,84)), 1, 1, np.ones((84,84)), True))
        per3.add2((np.ones((84,84)), 1, 1, np.ones((84,84)), True))
##        per4.store((np.ones((84,84)), 1, 1, np.ones((84,84)), True))

    t = time.time()
    for i in range(500):
        a,b,c = per.sample(4)
    print(c)
    tf1 = time.time() - t
    print("Execution time: ", tf1)

##    t = time.time()
##    for i in range(500):
##        a,b,c = per2.sample(4)
##    print(c)
##    tf2 = time.time() - t
##    print("Execution time: ", tf2)

    t = time.time()
    for i in range(500):
        a,b,c = per3.sample2(4)
    print(c)
    tf3 = time.time() - t
    print("Execution time: ", tf3)

##    t = time.time()
##    for i in range(500):
##        a,b,c = per4.sample(50)
##    print(c)
##    tf4 = time.time() - t
##    print("Execution time: ", tf4)

##    print("rapport: ", tf1/tf2, tf1/tf3, tf3/tf2)
    # with my setup I have
    # tf1 = 30.0676s
    # tf2 = 0.7692s
    # tf3 = 3.4518s
    # r1 = 39 - r2 = 8.7 - r3 = 4.5
    # PER_NP is clearly more efficient than PER

evalPerfSample()


def testnpvslist():
    '''
    Here numpy solution is more efficient
    '''
    a = list(range(10000))
    a2 = np.ones(10000)

    t = time.time()
    for i in range(10000):
        b = [a[i] for i in range(len(a))]
    print("time: ", time.time() - t)


    t = time.time()
    for i in range(10000):
        b = a2[range(len(a))]
    print("time: ", time.time() - t)
