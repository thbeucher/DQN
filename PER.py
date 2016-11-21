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

pq = pqdict({'a':(3, (1,2,3,4)), 'b':(5, (1,2,3,4)), 'c':(8, (1,2,3,4))}, reverse=True, key=lambda x:x[0])
print(pq)
print(pq.top(), pq.topitem())
pq.additem('d', (7, (1,2,3,4)))
print(pq)
pq.additem('e', (pq.topitem()[1][0], (4,3,2,1)))
print(pq)
pq.updateitem('e', (4, pq.get('e')[1]))
print(pq)
