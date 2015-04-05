# -*- coding: utf-8 -*-
import unittest
import numpy as np
from TPI import decompose

class TestDecomposition(unittest.TestCase):
    """ Classe de testes unitários """

    def assertArrEq(self, arr1, arr2):
        """
        Função que verifica se dois numpy.array são iguais
        """
        self.assertEqual(list(arr1), list(arr2))

    def test_sen_meio_de_n_pi(self):
        """ Teste y[n] = sin(n * pi / 2) """
        yin = np.sin(np.pi * np.arange(-5,6) / 2)
        n0  = -5
        dom, yout, even, odd = decompose(yin, n0)
        
        self.assertArrEq(even, np.zeros(11))
        self.assertArrEq(odd, yin)

    def test_cos_n_pi(self):
        """ Teste y[n] = cos(n * pi) """
        yin = np.cos(np.pi * np.arange(-2,3))
        n0  = -2
        dom, yout, even, odd = decompose(yin, n0)

        self.assertArrEq(even, yin)
        self.assertArrEq(odd, np.zeros(5))

    def test_cos_n_pi_2(self):
        """ Teste y[n] = cos(n * pi) com -1 <= n <= 3"""
        yin = np.cos(np.pi * np.arange(-1,4))
        n0  = -1
        dom, yout, even, odd = decompose(yin, n0)

        self.assertArrEq(dom, np.arange(-3,4))
        self.assertArrEq(yout, np.hstack(([0,0], np.cos(np.pi * np.arange(-1,4)))))
        self.assertArrEq(even, [-0.5, 0.5, -1, 1, -1, 0.5, -0.5])
        self.assertArrEq(odd, [0.5, -0.5, 0, 0, 0, 0.5, -0.5])

    def test_sin_n_pi_quadrado(self):
        """ Teste y[n] = [sin(n * pi / 2)]^2 """
        yin = np.sin(np.pi * np.arange(-3,4) / 2) ** 2
        n0  = -3
        dom, yout, even, odd = decompose(yin, n0)

        self.assertArrEq(even, yin)
        self.assertArrEq(odd, np.zeros(7))

    def test_sin_meio_n_pi_vezes_cos_n_pi(self):
        """ Teste y[n] = sin(n * pi / 2) * cos(n * pi) """
        yin = np.sin(np.pi * np.arange(-3,4) / 2) * np.cos(np.pi * np.arange(-3,4))
        n0  = -3
        dom, yout, even, odd = decompose(yin, n0)

        self.assertArrEq(even, np.zeros(7))
        self.assertArrEq(odd, yin)

    def test_e_elevado_cos_n_pi(self):
        """ Teste y[n] = e^cos(n * pi) """
        yin = np.exp(np.cos(np.pi * np.arange(-3,4)))
        n0  = -3
        dom, yout, even, odd = decompose(yin, n0)
        
        self.assertArrEq(even, yin)
        self.assertArrEq(odd, np.zeros(7))

if __name__ == '__main__':
    #executa testes
    unittest.main()