# 🧠 Perceptron Simples: Classificação de Portas Lógicas e Dataset Iris

Este projeto em Python implementa um **Perceptron de Camada Única** (o modelo mais básico de rede neural).

## 🎯 Funcionalidades

O código permite treinar o Perceptron para resolver quatro problemas distintos:

1.  **Porta Lógica OR**
2.  **Porta Lógica AND**
3.  **Porta Lógica XOR**
4.  **Classificação Binária no Dataset Iris** (Setosa vs. Não-Setosa)

### ⚠️ O Limite do Perceptron

O Perceptron Simples só consegue aprender problemas **linearmente separáveis** (como AND, OR e Iris Setosa vs. Outras).

* **AND** e **OR** **convergem** rapidamente.
* **XOR** (Exclusivo OR) é **linearmente inseparável** e, portanto, o algoritmo falhará em convergir em 100 épocas. 

## 🛠️ O Algoritmo de Treinamento

O Perceptron utiliza a **função de ativação degrau** (`sinal_degrau(V)`):

$$y = \begin{cases} 1 & \text{se } V \ge 0 \\ 0 & \text{se } V < 0 \end{cases}$$

Onde $V$ é o produto interno dos pesos $W$ e das entradas $X$ (incluindo o *bias*): $V = W \cdot X$.

A **Regra de Atualização dos Pesos** é:

$$W_{novo} = W_{antigo} + \eta \cdot (t - y) \cdot X$$

* $\eta$ (eta) é a **taxa de aprendizado** (`eta=0.1`).
* $t$ é a classe **desejada** (Target).
* $y$ é a classe **prevista** (Output).
* $X$ é o vetor de entrada.

## 🚀 Como Rodar o Projeto

### 1. Pré-requisitos

O projeto requer as seguintes bibliotecas:

```bash
pip install numpy scikit-learn matplotlib
```

### 2. Execução

Salve o código como um arquivo Python (ex: perceptron.py) e execute-o no seu terminal:

```bash
python perceptron.py
```
