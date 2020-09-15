# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:50:44 2018

@author: amax
"""

import tensorflow as tf
from tensorpack.utils.argtools import graph_memoized


@graph_memoized
def get_fw_fa(bitsW, bitsA, bitsB, bitsBN_G, bitsBN_B, bitsBN_mean, bitsBN_var, bitsBN_x):
    
    @tf.custom_gradient
    def Q(x, bits):
        x=tf.cast(x,tf.float32)
        bits=tf.cast(bits,tf.float32)
        n=tf.pow(2.0,bits-1)
        y=tf.round(x*n)/n
        def grad(dy):
            return dy
        return y,grad
    
    def clip(x,bits):
        if bits>=32:
            delta=0.0
        else:
            x=tf.cast(x,tf.float32)
            bits=tf.cast(bits,tf.float32)
            delta=1./tf.pow(2.0,bits-1)
        MAX=+1-delta
        MIN=-1+delta
        x=tf.clip_by_value(x,MIN,MAX,name='saturate')
        return x
    
    def Shift(x):
        return 2 ** tf.round(tf.log(x) / tf.log(2.0))
    def S(bits):
        return 2.0 ** (bits - 1)
        
    @tf.custom_gradient
    def fw(x):
        def grad(dy):
            return dy            
        if bitsW >=32:
            return x,grad
        else:
            return clip(Q(x,bitsW),bitsW),grad
           
    @tf.custom_gradient
    def fa(x):
        def grad(dy):
            return dy
        if bitsA>=32:
            return x,grad
        else:
            return Q(x,bitsA),grad
    
    
    def fBits(x,bitsB=32):           
        if  bitsB >=32:
           return x
        else:
           return Q(x,bitsB)    
    
    @tf.custom_gradient
    def fbn_G(x):
        def grad(dy):
            return dy
        if bitsBN_G>=32:
            return x,grad
        else:
            return Q(x,bitsBN_G),grad
    
    @tf.custom_gradient
    def fbn_B(x):
        def grad(dy):
            return dy
        if bitsBN_B>=32:
            return x,grad
        else:
            return Q(x,bitsBN_B),grad
    
    @tf.custom_gradient
    def fbn_mean(x):
        def grad(dy):
            return dy
        if bitsBN_mean>=32:
            return x,grad
        else:
            return Q(x,bitsBN_mean),grad
            
    @tf.custom_gradient
    def fbn_var(x):
        def grad(dy):
            return dy
        if bitsBN_var>=32:
            return x,grad
        else:
            return Q(x,bitsBN_var),grad
    
    @tf.custom_gradient
    def fbn_x(x):
        def grad(dy):
            return dy
        if bitsBN_x>=32:
            return x,grad
        else:
            return Q(x,bitsBN_x),grad

    return fw,fa,fBits,fbn_G,fbn_B,fbn_mean,fbn_var,fbn_x


bitsW=8
bitsA=8
bitsB=8
bitsBN_G=8
bitsBN_B=8
bitsBN_mean=8
bitsBN_var=8
bitsBN_x=8


fw,fa,fBits,fbn_G,fbn_B,fbn_mean,fbn_var,fbn_x=\
  get_fw_fa(bitsW=bitsW, bitsA=bitsA,bitsB=bitsB,bitsBN_G=bitsBN_G,bitsBN_B=bitsBN_B,bitsBN_mean=bitsBN_mean,bitsBN_var=bitsBN_var,bitsBN_x=bitsBN_x)
        

         
        
    
         
