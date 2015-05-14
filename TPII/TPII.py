# -*- coding: utf-8 -*-
"""
Created on Thu May 14 08:57:44 2015

@author: Murilo Camargos

https://docs.python.org/2/reference/datamodel.html#object.__getitem__
http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html
https://github.com/murilocamargos/sinais-e-sistemas/blob/master/TPI
"""

import numpy as np
import matplotlib.pyplot as plt

class DSignal(object):
    def __init__(self, dom, img):
        self.sig1 = self.sig2 = None
        
        if type(img) in [list, tuple, np.ndarray]:
            self.img = np.array(img)
        else:
            raise ValueError('You must provide a list, tuple or numpy.array.')
            
        if type(dom) == int:
            self.dom = np.arange(dom, dom + len(self.img))
        elif type(dom) in [list, tuple, np.ndarray]:
            self.dom = np.array(dom)
        else:
            raise ValueError('You must provide a list, tuple or numpy.array.')
            
    def __neg__(self):
        return DSignal(-self.dom[::-1], self.img[::-1])
        
    def __getitem__(self, key):
        if key in self.dom:
            return self.img[list(self.dom).index(key)]
        return 0
        
    def conv(self, sig2):
        if type(sig2) != DSignal:
            raise ValueError('You must provide another DSignal.')
        
        sig1 = -self
        
        ni = sig2.dom[0] - sig1.dom[-1]
        nf = sig2.dom[-1] - sig1.dom[0]
        domConv = np.arange(ni, nf + 1)
        
        sig1.dom += domConv[0] - 1
        
        yn = []
        for n in domConv:
            sig1.dom += 1
            inter = set(sig1.dom) & set(sig2.dom)

            yn += [sum([sig1[i] * sig2[i] for i in inter])]
        
        convolved = DSignal(domConv, yn)
        convolved.sig1 = self
        convolved.sig2 = sig2
        
        return convolved
    
    def plot(self, title = 'Signal', padding = 1, conv = False):
        pad_min = lambda arr: min(arr) - padding
        pad_max = lambda arr: max(arr) + padding
        axis = lambda x,y: [pad_min(x), pad_max(x), pad_min(y), pad_max(y)]
        
        if conv == True and self.sig1 != None and self.sig2 != None:
            sig1 = self.sig1
            sig2 = self.sig2
        
            fig, ((ax1,ax2,ax3)) = plt.subplots(1, 3, sharex='col', sharey='row')
            
            ax1.stem(sig1.dom, sig1.img, linefmt='b')
            ax1.set_title('Signal 1')
            ax1.axis(axis(sig1.dom, sig1.img))
            
            ax2.stem(sig2.dom, sig2.dom)
            ax2.set_title('Signal 2')
            ax2.axis(axis(sig2.dom, sig2.img))
        
            ax3.stem(self.dom, self.img)
            ax3.set_title('Convolution')
            ax3.axis(axis(self.dom, self.img))
        
        else:
            plt.stem(self.dom, self.img, linefmt='b')
            plt.axis(axis(self.dom, self.img))
            plt.title(title)
            plt.show()
    
        

s1 = DSignal(-1, [1,2,1])
s2 = DSignal(0, [2,3,-2])
s1.conv(s2).plot()