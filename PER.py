#-------------------------------------------------------------------------------
# Name:        PER
# Purpose:
#
# Author:      tbeucher
#
# Created:     21/11/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from pqdict import pqdict
import math
import numpy as np
import random

'''
P(i) = Pi**alpha / sum_k Pk**alpha
Wi = ( 1 / N * P(i) )**beta
-rank-based: Pi = 1 / rank(i)
-proportional: Pi = abs(delta_i) + epsilon
-We sample a segment and then sample uniformly among the transitions within it
-Choose k=batch_size and sample one transition from each segment
-Change Q-learning update by using Wi*delta_i instead of delta_i
-For stability reason, normalize weights by 1 / max_i Wi
-Given that prioritized replay picks high-error transitions more often, the typical gradient
magnitudes are larger, so reduced the step-size (learning rate) by a factor 4 compared
to the standard setup (DQN, DDQN setup)
-alpha = 0.7 beta_zero = 0.5 for rank-based
-alpha = 0.6 beta_zero = 0.4 for the proportional variant

-for rank-based variant, the cumulative density function can be approximate with a piecewise
linear function with k segments of equal probability
'''

def testPQ():
    pq = pqdict({'a':(3, (1,2,3,4)), 'b':(5, (1,2,3,4)), 'c':(8, (1,2,3,4))}, reverse=True, key=lambda x:x[0])
    print(pq)
    print(pq.top(), pq.topitem())
    pq.additem('d', (7, (1,2,3,4)))
    print(pq)
    pq.additem('e', (pq.topitem()[1][0], (4,3,2,1)))
    print(pq)
    pq.updateitem('e', (4, pq.get('e')[1]))
    print(pq)


class PER:

    def __init__(self, **args):
        '''
        '''
        self.size = args['size'] # size of the replay memory
        self.alpha = args['alpha'] # determine how much prioritization is used, alpha = 0 = uniform case
        self.beta_zero = args['beta_zero'] # beta is annealed from beta_zero to 1
        self.batch_size = args['batch_size']
        self.k = args['nb_segments'] # k, number of segments

        self.tsp = self.build_tsp()

    def build_tsp(self):
        '''
        P(i) = Pi**alpha / sumk(Pk**alpha)
        rank-based prioritization: Pi = 1 / rank(i)
            -> P(i) = rank(i)**(-alpha) / sumk(rank(k)**(-alpha))
        '''
        #probability density function
        pdf = list(map(lambda x: x**(-self.alpha), range(1,self.size+1)))
        pdf_sum = math.fsum(pdf)
        #transition sampling probability
        tsp = list(map(lambda x: x/pdf_sum, pdf))
        #each segment has probability of 1/batch_size
        #cumulative density function
        cdf = np.cumsum(tsp)
        #start and end for all segments
        segments_idx = {1:0, self.batch_size+1:self.size}
        step = 1./self.batch_size
        i = 1
        for s in range(2,self.batch_size+1):
            while cdf[i] < step:
                i += 1
            segments_idx[s] = i
            step += 1./self.batch_size
        return {'pdf':pdf, 'segments_idx':segments_idx}

    def sample(self):
        '''
        '''
        a=1


a = PER(size=10, alpha=0.7, beta_zero=0.5, batch_size=4, nb_segments=4)
print(a.tsp)




