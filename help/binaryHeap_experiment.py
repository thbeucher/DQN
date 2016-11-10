#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      tbeucher
#
# Created:     10/11/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import heapq
import heapq_max

class MaxHeapObj(object):
  def __init__(self,val): self.val = val
  def __lt__(self,other): return self.val > other.val
  def __eq__(self,other): return self.val == other.val
  def __str__(self): return str(self.val)

class MaxPriorityQueue(object):

    def __init__(self, **args):
        '''
        Creates a new max-heap
        '''
        self.h = []
        self.size = args['size'] if 'size' in args else None
        self.size_fixed = True if self.size != None else False

    def push(self, priority, item):
        '''
        Push an item with priority into the heap
        if size is fixed, push new item and pop item with lowest priority
        '''
        if not self.size_fixed:
            heapq.heappush(self.h, MaxHeapObj((priority, item)))
        else:
            if len(self.h) < self.size:
                heapq.heappush(self.h, MaxHeapObj((priority, item)))
            else:
                _ = heapq.heappushpop(self.h, MaxHeapObj((priority, item)))

    def store(self):
        a=1

    def pop(self):
        '''
        Returns the item with highest priority
        It's a pop action so the element is delete from the heap
        '''
        return heapq.heappop(self.h).val

    def get_highest_priority_obj(self):
        '''
        Returns item with highest priority
        '''
        return heapq.nlargest(1, self.h, MaxHeapObj)[0].val

    def get_highest_priority(self):
        '''
        Returns value of the highest priority
        '''
        return heapq.nlargest(1, self.h, MaxHeapObj)[0].val[0]

    def get_all(self):
        '''
        Returns a list with all elements
        '''
        return [el.val for el in heapq.nlargest(len(self.h), self.h, MaxHeapObj)]


def runTest():
    print("test max heap")
    test_data = [(1., 0), (0.9, 1), (1.1, 2), (1.1, 3), (3.3, 4), (0., 5), (0.93, 6)]
    mpq = MaxPriorityQueue(size=7)
    for p, i in test_data:
        mpq.push(p, i)
    print(mpq.get_all())
    mpq.push(2.5, 22)
    print(mpq.get_all())

#runTest()


import heapq_max

class MaxPriorityQueue2(object):

    def __init__(self, **args):
        '''
        Creates a new max-heap
        '''
        self.h = []
        self.size = args['size'] if 'size' in args else None
        self.size_fixed = True if self.size != None else False

    def push(self, priority, item):
        '''
        Push an item with priority into the heap
        if size is fixed, push new item and pop item with lowest priority
        '''
        if not self.size_fixed:
            heapq_max.heappush_max(self.h, (priority, item))
        else:
            if len(self.h) < self.size:
                heapq_max.heappush_max(self.h, (priority, item))
            else:
                _ = heapq_max.heappushpop_max(self.h, (priority, item))

    def store(self):
        a=1

    def pop(self):
        '''
        Returns the item with highest priority
        It's a pop action so the element is delete from the heap
        '''
        return heapq_max.heappop_max(self.h)

    def get_highest_priority_obj(self):
        '''
        Returns item with highest priority
        '''
        return self.h[0]

    def get_highest_priority(self):
        '''
        Returns value of the highest priority
        '''
        return self.h[0][0]

    def get_all(self):
        '''
        Returns a list with all elements
        '''
        heap = heapq.nlargest(len(self.h), self.h, MaxHeapObj)
        heap.reverse()
        return heap

def run_test():
    test_data = [(1., 0), (0.9, 1), (1.1, 2), (1.1, 3), (3.3, 4), (0., 5), (0.93, 6)]
    mpq = MaxPriorityQueue2(size=7)
    for p, i in test_data:
        mpq.push(p, i)
    print(mpq.get_all())
    mpq.push(2.5, 22)
    print(mpq.get_all())

#run_test()