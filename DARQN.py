#-------------------------------------------------------------------------------
# Name:        DARQN
# Purpose:     Implementation of the Deep Attention Recurrent Q-Network
#              algorithm.
#
# Author:      tbeucher
#
# Created:     25/11/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

'''
s_t -> CNN -> D feature maps of dimension m*m -> Attention network transformes
these maps into a set of vectors v_t = {v_t^1, ..., v_t^L} £ R^D, L = m*m and
outputs their linear combination z_t £ R^D (z_t=context vector) -> LSTM network
takes z_t as input along with the previous hidden state h_t-1 and memory state c_t-1
and produces hidden state h_t that is used by:
    (1)a linear layer for evaluation Q-value of each action a_t that the agent
    can take being in state s_t
    (2)the attention network for generating a context vector at the next time step t+1

2 approaches to the Context vector calculation -> (1) soft attention - (2) hard attention

'''
