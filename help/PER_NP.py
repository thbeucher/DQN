#-------------------------------------------------------------------------------
# Name:        PER_NP
# Purpose:
#
# Author:      tbeucher
#
# Created:     21/11/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from math import fsum
import numpy as np

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

class PER_NP:

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
		self.annealing_beta_steps = args['annealing_beta_steps'] # number of step to anneale beta from beta_zero to 1

		self.decrease_step_beta = (1 - beta_zero) / self.annealing_beta_steps

		dtypesPQ = dict(names=['priority', 'experience'], formats=['f8', tuple])
		self.pq = np.array([(1., (np.ones(1), 1, 1, np.ones(1), False)) for i in range(self.size)], dtype=dtypesPQ)
		self.compteur = 0 # used to fill pq

		self.tsp = self.build_tsp()

	def add(self, experience):
		'''
		Add a new element in the experience memory with the highest priority

		experience - tuple or list - (s, a, r, s1, terminal)
		'''
		max_priority = np.amax(self.pq['priority']) +0.001 # get the highest priority
		if self.compteur >= self.size:
			key_to_insert = np.argmin(self.pq['priority']) # get a free key
			self.pq[key_to_insert] = (max_priority, experience) # update priority queue
		else:
			self.pq[self.compteur] = (max_priority, experience)
			self.compteur += 1

	def update(self, priorities, experience_ids):
		'''
		Updates the priority of the given experience

		priorities - list - list of priorities ie delta
		experience_ids - list - list of ids of experience to update
		'''
		self.pq['priority'][experience_ids] = priorities

	def build_tsp(self):
		'''
		Preprocess the probability sampling, need the replay memory to be full
		P(i) = Pi**alpha / sumk(Pk**alpha)
		rank-based prioritization: Pi = 1 / rank(i)
			-> P(i) = rank(i)**(-alpha) / sumk(rank(k)**(-alpha))
		'''
		#probability density function
		pdf = list(map(lambda x: x**(-self.alpha), range(1,self.size+1)))
		pdf_sum = fsum(pdf)
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
		self.sample_idx = [np.random.randint(seg_idx[i], seg_idx[i+1]) for i in range(1, self.batch_size+1)]
		# annealing beta
		tmp = self.beta + self.decrease_step_beta
		self.beta =  tmp if tmp < 1 else 1
		# compute IS weights - Wi = ( 1 / N * P(i) )**beta
		p_i = [self.tsp['pdf'][i] for i in self.sample_idx]
		w = np.power(np.array(p_i)*self.k, -self.beta)
		w_max = w.max()
		w = w / w_max # normalize w
		# get experience
		self.pq[::-1].sort(order='priority')
		experiences = self.pq['experience'][self.sample_idx]
		return experiences, w, self.sample_idx


	def save(self):
		'''
		Saves the priority queue array pq
		'''
		np.save('myPER_NP', self.pq)

	def load(self):
		'''
		Loads the priority queue array pq
		'''
		self.pq = np.load('myPER_NP.npy')


def test_PER_NP():
	'''
	'''
	per = PER_NP(size=10, alpha=0.7, beta_zero=0.5, batch_size=4, nb_segments=4, annealing_beta_steps=15)
	for i in range(10):
		per.add((np.ones(2), i, i+1, np.ones(2), False))
	print(per.pq)
	per.update([2,5,4,9,3,1,7,10,8,6], [0,1,2,3,4,5,6,7,8,9])
	print(per.pq)
	e, w, ids = per.sample(4)
	print("ids: ", ids)
	print("experiences: ", e)
	per.add((np.ones(2), 10, 11, np.ones(2), False))
	print(per.pq)
	e, w, ids = per.sample(4)
	print("ids: ", ids)
	print("experiences: ", e)


#test_PER_NP()

