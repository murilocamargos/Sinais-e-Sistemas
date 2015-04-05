# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def graph(dom, orig, even, odd):
    """
    Esta função  é  responsável pela plotagem dos gráficos do sinal de entrada,
    parte par e ímpar e soma das partes do sinal.

    Parametros
    +--------+-------------+--------------------------------------------------+
    |  Nome  |     Tipo    | Descrição                                        |
    +--------+-------------+--------------------------------------------------+
    |        |             | É o domínio extendido do sinal de entrada.  Para |
    |  dom   | numpy.array | facilitar, o mesmo domínio foi utilizado para os |
    |        |             | quatro sinais.                                   |
    +--------+-------------+--------------------------------------------------+
    |  orig  | numpy.array | Sinal original extendido.                        |
    +--------+-------------+--------------------------------------------------+
    |  even  | numpy.array | Parte par do sinal.                              |
    +--------+-------------+--------------------------------------------------+
    |  odd   | numpy.array | Parte ímpar do sinal.                            |
    +--------+-------------+--------------------------------------------------+      

    """

    # Divide a janela de plotagem em quatro regiões.
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    ax1.stem(dom, orig, linefmt='b')
    ax1.set_title('Sinal de Entrada')
    
    ax2.stem(dom, even)
    ax2.set_title('Parte Par')

    ax3.stem(dom, odd)
    ax3.set_title('Parte Impar')

    ax4.stem(dom, (odd + even))
    ax4.set_title('Soma das Partes')

    # Corrige os eixos das plotagens para que elas tenham  uma margem em branco
    # que facilita a leitura dos gráficos.
    def new_axis(ax, mr = 0.5):
        x0, x1, y0, y1 = ax.axis()
        return (x0 - mr, x1 + mr, y0 - mr, y1 + mr)
    
    ax1.axis(new_axis(ax1))
    ax2.axis(new_axis(ax2))
    ax3.axis(new_axis(ax3))
    ax4.axis(new_axis(ax4))

    plt.subplots_adjust(hspace=0.5)

    plt.show()


def expand(dom_y, n0, yn):
    """
    Esta função expande o conjunto do contra-domínio fazendo com que os elemen-
    tos não mapeados do domínio extendido sejam mapeados para zero.

    Parametros
    +------------+-------------+----------------------------------------------+
    |    Nome    |     Tipo    | Descrição                                    |
    +------------+-------------+----------------------------------------------+
    |   dom_y    | numpy.array | O domínio extendido do sinal de entrada.     |
    +------------+-------------+----------------------------------------------+
    |     n0     | numpy.array | O instante inicial do sinal de entrada.      |
    +------------+-------------+----------------------------------------------+
    |     yn     | numpy.array | Sinal de entrada original.                   |
    +------------+-------------+----------------------------------------------+ 

    Retorna
    +------------+-------------+----------------------------------------------+
    |    Nome    |     Tipo    | Descrição                                    |
    +------------+-------------+----------------------------------------------+
    | y_expanded | numpy.array | Sinal de entrada expandido que mapeia todos  |
    |            |             | os elementos do domínio extendido.           |
    +------------+-------------+----------------------------------------------+                            

    """

    # Domínio original do sinal de entrada.
    dom = range(n0, n0 + len(yn))

    y_expanded = np.zeros(len(dom_y))

    # Para cada novo elemento  do contra-domínio extendido, temos que verificar
    # se este elemento existe no domínio original. Se existir, devemos mapeá-lo
    # à saída original, caso contrário, a saída permanecerá zero.
    for i, n in enumerate(dom_y):
        try:
            y_expanded[i] = yn[dom.index(n)]
        except ValueError:
            pass

    return y_expanded

def decompose(yn, n0):
    """
    Esta função realiza a decomposicao de um sinal y[n] qualquer em suas partes
    par e ímpar.

    Parametros
    +---------+-------------+-------------------------------------------------+
    |   Nome  |     Tipo    | Descrição                                       |
    +---------+-------------+-------------------------------------------------+
    |         |             | Conjunto que contém os valores da variável de-  |
    |    yn   | numpy.array | pendente. São os valores mapeados pelo sinal a  |
    |         |             | partir do instante inicial n0.                  |
    +---------+-------------+-------------------------------------------------+
    |    n0   |     int     | Valor do instante inicial do sinal.             |
    +---------+-------------+-------------------------------------------------+

    Retorna
    +---------+-------------+-------------------------------------------------+
    |   Nome  |     Tipo    | Descrição                                       |
    +---------+-------------+-------------------------------------------------+
    |  dom_y  | numpy.array | Novo domínio extendido. Auxiliará na plotagem   |
    |         |             | dos gráficos.                                   |
    +---------+-------------+-------------------------------------------------+
    | y_input | numpy.array | Sinal de entrada extendido para o mesmo tamanho |
    |         |             | de dom_y, mapeando os novos valores para zero.  |
    +---------+-------------+-------------------------------------------------+
    | y_even  | numpy.array | Parte par do sinal extendido.                   |
    +---------+-------------+-------------------------------------------------+
    |  y_odd  | numpy.array | Parte ímpar do sinal extendido.                 |
    +---------+-------------+-------------------------------------------------+

    """

    # Cálculo  do  raio  do intervalo fechado centrado em zero, que auxiliará a
    # criação do domínio extendido, como explicado na documentação.
    raio = max(abs(n0), abs(n0 + len(yn) - 1))

    # Para decompor o sinal, é necessário  que o valor de n esteja num conjunto
    # onde, para qualquer valor de inteiro  de n que esteja mapeado pelo sinal,
    # eu possa acessar o valor da variável dependente em -n.
    dom_y = np.arange(-raio, raio + 1)

    # Devemos extender o contra-domínio também, considerando  que nos instantes
    # em que a entrada não foi mapeada pelo sistema, a saída é zero.
    y_input = expand(dom_y, n0, yn)

    # Para realizar a decomposição foi necessário obter  o  sinal y[-n].  Neste
    # caso, bastou-se reverter o sinal de entrada.
    y_negative = y_input[::-1]

    y_even = 0.5 * (y_input + y_negative)
    y_odd  = 0.5 * (y_input - y_negative)

    return (dom_y, y_input, y_even, y_odd)