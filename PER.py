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

from pqdict import pqdict, nsmallest, nlargest
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

class PER:

    def __init__(self, **args):
        '''
        Prioritized experience replay

        Currently, the rank-based variant is implemented but not the proportional variant
        '''
        self.size = args['size'] # size of the replay memory
        self.alpha = args['alpha'] # determine how much prioritization is used, alpha = 0 = uniform case
        beta_zero = args['beta_zero'] # beta is annealed from beta_zero to 1
        self.beta = beta_zero
        self.batch_size = args['batch_size']
        self.k = args['nb_segments'] # k, number of segments
        self.annealing_beta_steps = args['beta_grad'] # number of step to anneale beta from beta_zero to 1

        self.decrease_step_beta = (1 - beta_zero) / self.annealing_beta_steps

        #priority queue, reverse = True = max heap
        #transition must be saved as (priority, (s, a, r, s1, terminal))
        self.pq = pqdict({}, reverse=True)
        self.buffer = {} # experience store

        self.tsp = self.build_tsp()

    def add(self, experience):
        '''
        Add a new element in the experience memory with the highest priority

        experience - tuple or list - (s, a, r, s1, terminal)
        '''
        max_priority = self.pq.topitem()[1] if len(self.pq) > 0 else 1 # get the highest priority
        #get a free key to insert into the priority queue
        if len(self.pq) >= self.size:
            key_to_insert = nsmallest(1, self.pq)[0] # get key of item to replace
            self.pq.updateitem(key_to_insert, max_priority) # replace item with highest priority
        else:
            key_to_insert = len(self.pq) # get a free key
            self.pq.additem(key_to_insert, max_priority) # store experience index and priority in the heap
        self.buffer[key_to_insert] = experience # store the experience


    def update(self, priorities, experience_ids):
        '''
        Updates the priority of the given experience

        priorities - list - list of priorities ie delta
        experience_ids - list - list of ids of experience to update
        '''
        for key, priority in zip(experience_ids, priorities):
            self.pq.updateitem(key, priority)

    def build_tsp(self):
        '''
        Preprocess the probability sampling, need the replay memory to be full
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

    def sample(self, batch_size):
        '''
        Samples x transitions from the replay memory (x = batch_size)

        return:
            -experiences - list
            -w - numpy array - IS weights (IS = importance sampling)
        '''
        # sample one element on each segments
        seg_idx = self.tsp['segments_idx']
        sample_idx = [np.random.randint(seg_idx[i], seg_idx[i+1]) for i in range(1, self.batch_size+1)]
        # annealing beta
        tmp = self.beta + self.decrease_step_beta
        self.beta =  tmp if tmp < 1 else 1
        # compute IS weights - Wi = ( 1 / N * P(i) )**beta
        p_i = [self.tsp['pdf'][i] for i in sample_idx]
        w = np.power(p_i*self.k, -self.beta)
        w_max = w.max()
        w = w / w_max # normalize w
        # get experience
        all_exp_ids = nlargest(len(self.pq), self.pq)
        experience_ids = [all_exp_ids[i] for i in sample_idx]
        experiences = [self.buffer[i] for i in experience_ids]
        print("all_exp_ids: ", all_exp_ids)
        print("experience_ids: ", experience_ids)
        print("sample_idx: ", sample_idx)
        return experiences, w



a = PER(size=10, alpha=0.7, beta_zero=0.5, batch_size=4, nb_segments=4, beta_grad=15)
print("tsp: ", a.tsp)

## sample test ##
for i in range(10):
    a.add((i, i+1))
a.update([3, 5, 6, 9, 4], [2, 4, 7, 3, 5])
print("pq: ", a.pq)
print("buffer: ", a.buffer)
e, w = a.sample(4)
print("e: ", e)
## end of sample test ##

## add test ##
##for i in range(10):
##    a.add((i, i+1))
##print(a.pq)
##print(a.buffer)
##a.add((10, 11))
##print(a.pq)
##print(a.buffer)
## end of add test ##

## update priority test ##
##for i in range(10):
##    a.add((i, i+1))
##print(a.pq)
##print(a.buffer)
##a.update([3, 5, 6], [2, 4, 7])
##print(a.pq)
##print(a.buffer)
##print(a.pq.topitem())
##a.add((10, 11))
##print(a.pq)
##print(a.buffer)
##print(a.pq.topitem())
##a.update([4], [7])
##print(a.pq)
##print(a.buffer)
## end of update priority test ##


def testPQ():
    pq = pqdict({'a':(3, (1,2,3,4)), 'b':(5, (1,2,3,4)), 'c':(8, (1,2,3,4))}, reverse=True, key=lambda x:x[0])
    print(pq)
    print(nsmallest(1, pq))
    #print(pq.top(), pq.topitem())
    #pq.additem('d', (7, (1,2,3,4)))
    #print(pq)
    #pq.additem('e', (pq.topitem()[1][0], (4,3,2,1)))
    #print(pq)
    #pq.updateitem('e', (4, pq.get('e')[1]))
    #print(pq)

#testPQ()

