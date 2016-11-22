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
        self.size = args['size']
        self.alpha = args['alpha'] # determine how much prioritization is used, alpha = 0 = uniform case
        self.beta_zero = args['beta_zero'] # beta is annealed from beta_zero to 1
        self.batch_size = args['batch_size']
        self.k = args['nb_segments'] # k, number of segments

        self.distribution = self.build_distribution()

    def build_distribution(self):
        '''
        P(i) = Pi**alpha / sumk(Pk**alpha)
        rank-based prioritization: Pi = 1 / rank(i)
            -> P(i) = rank(i)**(-alpha) / sumk(rank(k)**(-alpha))
        '''
        res = {}
        partition_num = 1
        partition_size = self.size // self.k
        for n in range(partition_size, self.size+1, partition_size):
            print("n: ", n)
            distribution = {}
            proba_i_list = list(map(lambda x: x**(-self.alpha), range(1, n+1)))
            proba_sum = math.fsum(proba_i_list)
            distri = list(map(lambda x: x / proba_sum, proba_i_list))
            cum_distri = np.cumsum(distri)
            print("proba_i_list: ", proba_i_list)
            print("proba_sum: ", proba_sum)
            print("distri: ", distri)
            print("cumdistri: ", cum_distri)

            strata_ends = {1: 0, self.batch_size + 1: n}
            step = 1 / self.batch_size
            index = 1
            for s in range(2, self.batch_size + 1):
                while cum_distri[index] < step:
                    index += 1
                strata_ends[s] = index
                step += 1 / self.batch_size
            print("strata_ends: ", strata_ends)
            distribution['pdf'] = distri
            distribution['strata_ends'] = strata_ends
            res[partition_num] = distribution
            partition_num += 1
        print(res)
        return res

    def sample(self):
        '''
        '''
        dist_index = math.floor(10 / self.size * self.k)
        partition_max = dist_index * self.k
        distribution = self.distribution[dist_index]
        rank_list = []
        # sample from k segments
        for n in range(1, self.batch_size + 1):
            print("from: ", distribution['strata_ends'][n] + 1)
            print("to: ", distribution['strata_ends'][n+1])
            index = random.randint(distribution['strata_ends'][n] + 1,
                                   distribution['strata_ends'][n + 1])
            rank_list.append(index)
        print("rank_list: ", rank_list)


a = PER(size=10, alpha=0.7, beta_zero=0.5, batch_size=3, nb_segments=3)
a.sample()
a.sample()
a.sample()




