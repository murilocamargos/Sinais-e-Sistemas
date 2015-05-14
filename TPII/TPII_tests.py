# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:39:18 2015

@author: Murilo Camargos

Este arquivo irá realizar testes unitários com a função de convolução implemen-
tada na classe `DSignal` do arquivo `TPII.py`. A comparação dos resultados será
feita utilizando a função `convolve` da biblioteca `numpy`.

"""

import unittest
import numpy as np
from TPII import DSignal

class TestDecomposition(unittest.TestCase):
    """ Classe de testes unitários """

    def assertArrEq(self, arr1, arr2):
        """
        Função que verifica se dois numpy.array são iguais
        """
        self.assertEqual(list(arr1), list(arr2))

    def test_1(self):
        """ Convoluçao de p[n] * q[n]
        p[n] = /  1,  x = -1               q[n] = /  2,  x = 0
               |  2,  x = 0                       |  3,  x = 1
               |  1,  x = 1                       | -2, x = 2
               \  0,  caso contrário              \  0,  caso contrário
        """
        sig1 = DSignal(-1, [  1,  2,  1])
        sig2 = DSignal( 0, [  2,  3, -2])
        sig3 = sig1 ** sig2
        self.assertArrEq(sig3.img, np.convolve([1, 2, 1], [2, 3, -2]))
    
    def test_2(self):
        """ Convoluçao de p[n] * q[n]
        p[n] = /  1,  x = 0                q[n] = /  1,  x = 0
               |  1,  x = 1                       |  2,  x = 1
               |  1,  x = 2                       \  0,  caso contrário
               \  0,  caso contrário              
        """
        sig1 = DSignal( 0, [  1,  1,  1])
        sig2 = DSignal( 0, [  1,  2])
        sig3 = sig1 ** sig2
        self.assertArrEq(sig3.img, np.convolve([1, 1, 1], [1, 2]))
        
    def test_3(self):
        """ Convoluçao de p[n] * q[n]
        p[n] = /  5,  x = -5               q[n] = /  4,  x = 4
               |  0,  x = -4                      |  2,  x = 5
               |  1,  x = -3                      \  0,  caso contrário
               \  0,  caso contrário              
        """
        sig1 = DSignal( 0, [  5,  0,  1])
        sig2 = DSignal( 0, [  4,  2])
        sig3 = sig1 ** sig2
        self.assertArrEq(sig3.img, np.convolve([5, 0, 1], [4, 2]))


if __name__ == '__main__':
    #executa os testes
    unittest.main()