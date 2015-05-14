# -*- coding: utf-8 -*-
"""
Created on Thu May 14 08:57:44 2015

@author: Murilo Camargos

O segundo trabalho prático da disciplina Sinais e Sistemas consistiu na  imple-
mentação de um algoritmo em Python para realização da convolução entre dois si-
nais de tempo discreto (finitos).

"""

import numpy as np
import matplotlib.pyplot as plt

class DSignal(object):
    """
    Esta classe representa sinais de tempo discreto.
    """
    
    def __init__(self, dom, img):
        """
        O construtor recebe o sinal de entrada e define as  constantes iniciais
        de operação.
        
        Parametros
        +--------+-------------+----------------------------------------------+
        |  Nome  |     Tipo    | Descrição                                    |
        +--------+-------------+----------------------------------------------+
        |        | int      ou | É o domínio do sinal de entrada (variável in-|
        |        | list     ou | dependente). Caso seja list, tuple ou numpy. |
        |   dom  | tuple    ou | array, ele será o conjunto domínio. Caso seja|
        |        | numpy.array | inteiro, o conjunto será gerado a partir dele|
        |        |             | com tamanho igual ao da imagem.              |
        +--------+-------------+----------------------------------------------+
        |        | list     ou | São os valores que a variável dependente as- |
        |   img  | tuple    ou | sume no intervalo de domínio dado.           |
        |        | numpy.array |                                              |
        +--------+-------------+----------------------------------------------+
        
        """
        
        # Inicialmente, estes atributos são iguais ao elemento  Nulo do python.
        # Ele terá serventia quando o usuário fizer  uma  convolução  e  quiser
        # plotar os gráficos dos sinais e do resultado da convolução; para isso
        # teremos que armazenar os dois sinais em duas variáveis.
        self.sig1 = self.sig2 = None
        
        # Só aceita conjuntos de imagem formados por listas, tuplas  ou  arrays
        # do numpy.
        if type(img) in [list, tuple, np.ndarray]:
            self.img = np.array(img)
        else:
            raise ValueError('You must provide a list, tuple or numpy.array.')
        
        # Só aceita conjuntos de domínio formados por listas, tuplas  ou arrays
        # do numpy.
        if type(dom) == int:
            # Caso o usuário forneça apenas o instante inicial do sinal,  o do-
            # mínio será criado iniciando-se desse valor com a mesma cardinali-
            # dade do conjunto imagem.
            self.dom = np.arange(dom, dom + len(self.img))
        elif type(dom) in [list, tuple, np.ndarray]:
            # Não deixa que o usuário insira um conjunto de domínio com cardina-
            # lidade diferente ao conjunto imagem.
            if len(dom) != len(self.img):
                raise ValueError("The domain set must have the same size of the image set.")
            self.dom = np.array(dom)
        else:
            raise ValueError('You must provide a list, tuple or numpy.array.')
            
    def __neg__(self):
        """
        Esta função é chamada sempre que o usuário utilizar o operador de  sub-
        tração (-) antes de uma instância da classe DSignal.
        
        Retorna
        +--------+-------------+----------------------------------------------+
        |  Nome  |     Tipo    | Descrição                                    |
        +--------+-------------+----------------------------------------------+
        |        |             | Retorna uma nova instância da classe DSignal.|
        |        |   DSignal   | Ou seja, um novo sinal, que será o sinal ini-|
        |        |             | cial "rebatido", ou x[-n]                    |
        +--------+-------------+----------------------------------------------+
        
        """
        return DSignal(-self.dom[::-1], self.img[::-1])
        
    def __getitem__(self, key):
        """
        Esta função é chamada sempre que o usuário tentar acessar  um  elemento
        da instância da classe DSignal utilizando a notação de colchetes  (como
        se faz para acessar um elemento de uma lista, por exemplo). Neste caso,
        intuitivamente, o usuário estará avaliando o sinal (função)  num  valor
        passado por parâmetro: x[5], por exemplo.
        
        Parametros
        +--------+-------------+----------------------------------------------+
        |  Nome  |     Tipo    | Descrição                                    |
        +--------+-------------+----------------------------------------------+
        |   key  |     int     | É o valor da variável independente n, que se |
        |        |             | deseja saber o valor.                        |
        +--------+-------------+----------------------------------------------+
        
        Retorna
        +--------+-------------+----------------------------------------------+
        |  Nome  |     Tipo    | Descrição                                    |
        +--------+-------------+----------------------------------------------+
        |        |             | Retorna o valor da variável dependente quando|
        |        |   decimal   | a variável independente é igual ao parâmetro |
        |        |             | "key".                                       |
        +--------+-------------+----------------------------------------------+
        
        """
        
        # Caso o sinal não esteja definido  no  valor  de variável independente
        # recebido, retorna 0
        if key in self.dom:
            return self.img[list(self.dom).index(key)]
        return 0
        
    def __pow__(self, sig):
        """
        Esta função realiza a convolução entre dois sinais. Ela sobrecarrega o
        operador de potenciação do python para que quando o usuário utilizar a
        operação: DSignal ** DSignal, o resultado seja um novo DSignal que é
        exatamente a convolução entre os dois primeiros.
        
        Retorna
        +--------+-------------+----------------------------------------------+
        |  Nome  |     Tipo    | Descrição                                    |
        +--------+-------------+----------------------------------------------+
        |        |             | Retorna uma nova instância da classe DSignal.|
        |        |   DSignal   | Ou seja, um novo sinal, que será a convolução|
        |        |             | dos dois sinais envolvidos na operação.      |
        +--------+-------------+----------------------------------------------+
        
        """
        
        # Só realiza a operação se a potência for uma instância de DSignal.
        if type(sig) != DSignal:
            raise ValueError('You must provide another DSignal.')
        
        # O sinal sig2 é o segundo sinal da operação. Ele será rebatido, pois é
        # ele quem irá se movimentar.
        sig2 = -sig
        sig1 = self
        
        # Calcula  o  instante inicial e final em que o somatório de convolução
        # será diferente de zero.
        ni = sig1.dom[0] - sig2.dom[-1]
        nf = sig1.dom[-1] - sig2.dom[0]
        domConv = np.arange(ni, nf + 1)
        
        # Como sig2 irá se movimentar, deve-se posicioná-lo uma unidade atrás
        # do instante inicial da convolução.
        sig2.dom += domConv[0] - 1
        
        yn = []
        for n in domConv:
            # movimenta o domínio de sig2 de uma em uma unidade até chegar no
            # instante final da convolução
            sig2.dom += 1
            
            # Interseção entre os domínios dos dois sinais. É onde poderá haver
            # algum valor diferente de zero na multiplicação.
            inter = set(sig1.dom) & set(sig2.dom)

            yn += [sum([sig1[i] * sig2[i] for i in inter])]
        
        # Retorna o sinal resultante da convolução, mas salva no mesmo objeto
        # os sinais utilizados na operação, para que eles possam ser plotados
        # caso o usuário queira.
        convolved = DSignal(domConv, yn)
        convolved.sig1 = sig1
        convolved.sig2 = sig
        
        return convolved
    
    def plot(self, title = 'Signal', padding = 1, conv = False):
        """
        Esta função é responsável pela plotagem dos sinais.
        
        Parametros
        +---------+-------------+---------------------------------------------+
        |   Nome  |     Tipo    | Descrição                                   |
        +---------+-------------+---------------------------------------------+
        |  title  |    string   | É o título que será colocado no gráfico.    |
        +---------+-------------+---------------------------------------------+
        | padding |     int     | Espaçamento dentro do espaço de plotagem.   |
        +---------+-------------+---------------------------------------------+
        |   conv  |   boolean   | Identifica se o usuário quer plotar os três |
        |         |             | gráficos: sinal 1, sinal 2 e convolução.    |
        +---------+-------------+---------------------------------------------+
        
        """
    
        pad_min = lambda arr: min(arr) - padding
        pad_max = lambda arr: max(arr) + padding
        axis = lambda x,y: [pad_min(x), pad_max(x), pad_min(y), pad_max(y)]
        
        # Para imprimir os gráficos dos três sinais, conv deve ser True e o obj
        # deve possuir os sinais 1 e 2 salvos nos atributos.
        if conv == True and type(self.sig1) == DSignal and type(self.sig2) == DSignal:
            fig, ((ax1,ax2,ax3)) = plt.subplots(1, 3, sharex='col', sharey='row')
            
            ax1.stem(self.sig1.dom, self.sig1.img, linefmt='b')
            ax1.set_title('Signal 1')
            ax1.axis(axis(sig1.dom, sig1.img))
            
            ax2.stem(self.sig2.dom, self.sig2.img)
            ax2.set_title('Signal 2')
            ax2.axis(axis(sig2.dom, sig2.img))
        
            ax3.stem(self.dom, self.img)
            ax3.set_title('Convolution')
            ax3.axis(axis(self.dom, self.img))
        
        # Caso contrário, plota apenas o gráfico do objeto.
        else:
            plt.stem(self.dom, self.img, linefmt='b')
            plt.axis(axis(self.dom, self.img))
            plt.title(title)
            plt.show()