#-------------------------------------------------------------------------------
# Name:        theanoL
# Purpose:
#
# Author:      tbeucher
#
# Created:     14/10/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import theano.tensor as T
import theano as th


#adding two scalars
def addTwoScalars(a, b):
    #define symbols ie variables
    #dscalar = floating-point scalar
    x = T.dscalar('x')
    y = T.dscalar('y')
    #operation we want to perform
    z = x + y
    #create function taking x and y as inputs and giving z as output
    #first arg is list of variables that will be provided as inputs
    #second arg is a single or a list of variables, it is what we want to see as output
    f = th.function([x, y], z)
    print(f(a, b))

#addTwoScalars(2, 3)

#adding two matrices
def addTwoMatrices(a, b):
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x + y
    f = th.function([x, y], z)
    print(f(a, b))

#addTwoMatrices([[1,2], [3,4]], [[10,20], [30,40]])

#exemple
def ex1(a, b):
    x = T.vector('x')
    y = T.vector('y')
    z = x**2 + y**2 + 2*x*y
    f = th.function([x, y], z)
    print(f(a, b))

#ex1([1,2,3], [10,20,30])


#apply function to a matrix element-wise
#here the logistic function
def applyLogistic(a):
    x = T.dmatrix('x')
    out = 1/(1+T.exp(-x))
    logi = th.function([x], out)
    print(logi(a))

#applyLogistic([[0,1], [-1,-2]])

#compute more than one thing at the same time
def multipleThingAtTheSameTime(a, b):
    x, y = T.dmatrices('x', 'y')
    diff = x - y
    abs_diff = abs(diff)
    diff_squared = diff**2
    summ = x + y
    f = th.function([x,y], [diff, abs_diff, diff_squared, summ])
    print(f(a, b))

#multipleThingAtTheSameTime([[1,2], [3,4]], [[2,3], [4,5]])

#default value for an argument
def defaultValue(*arg):
    x, y, w = T.dscalars('x', 'y', 'w')
    z = x + y + w
    f = th.function([x, th.In(y, value=1), th.In(w, value=2, name='wName')], z)
    if len(arg) == 3:
        print(f(arg[0], wName = arg[1], y = arg[2]))
    elif len(arg) == 2:
        print(f(arg[0], arg[1]))
    else:
        print(f(arg[0]))

#defaultValue(10)
#defaultValue(10, 20)
#defaultValue(10, 20, 30)


#shared variables
def accumulator(a):
    #shared variable can be use by more than one function
    state = th.shared(0)
    inc = T.iscalar('inc')
    #update must be a list of pairs of the form (shared-variable, new expression)
    #or a dictionary whose keys are shared-variables and values are the new expressions
    acc = th.function([inc], state, updates=[(state, state+inc)])
    for el in a:
        print(str(el) + " --> " + str(acc(el)))
    #we can acces to the state value with state.get_value() and state.set_value()

#accumulator([1, 10, 200, 10])

#if we have defined a formula using a specific shared variable but we don't
#want to use its value --> we can use givens parameter
def replaceSharedVariable(a, b):
    state = th.shared(0)
    inc = T.iscalar('inc')
    #formula defined using state shared-variable
    fn = state * 2 + inc
    #the type of the argument used to replace the previous shared variable
    #, here state, must match
    foo = T.scalar(dtype=state.dtype)
    dec = th.function([inc, foo], fn, givens=[(state, foo)])
    print(dec(a, b))

#replaceSharedVariable(1,3)

#derivatives
#compute gradient
def gradient(a):
    x = T.dscalar('x')
    y = x**2
    z = 1/x
    gy = T.grad(y, x)
    gz = T.grad(z, x)
    print(th.pp(gy))
    print(th.pp(gz))
    f = th.function([x], gy)
    g = th.function([x], gz)
    print(f(a))
    print(g(a))

#gradient(2)


#conditions
#ifelse vs switch


#loop
