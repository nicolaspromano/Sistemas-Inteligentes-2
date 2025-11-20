# 🤖 Denoising Autoencoder (dAE) para Transfer Learning no CIFAR-10

Este projeto implementa um **Denoising Autoencoder (dAE)** em Keras/TensorFlow para demonstrar a eficácia do **Pré-treinamento Não Supervisionado** na melhoria da performance de um classificador em cenários de **dados rotulados limitados**.

## 🧠 O que é este Código?

O código realiza três etapas principais:

1.  ### **Treinamento do Denoising Autoencoder (dAE)**
    * Um Autoencoder Convolucional é treinado para **remover ruído Gaussiano** das imagens do dataset **CIFAR-10**.
    * O objetivo é forçar o **Encoder** a aprender características robustas e essenciais da imagem, ignorando o ruído.
    * 

[Image of Denoising Autoencoder Architecture]


2.  ### **Transferência de Conhecimento**
    * As camadas do **Encoder** treinado são transferidas para um novo modelo de **Classificação**.
    * Essas camadas de extração de características são **congeladas** (`trainable=False`) para preservar o conhecimento adquirido.

3.  ### **Comparação de Desempenho**
    * O **Classificador Pré-treinado** (usando os pesos do dAE) é comparado a um **Classificador Do Zero** (pesos aleatórios).
    * Ambos os modelos são treinados em um **subconjunto muito pequeno** de dados rotulados (apenas **500 imagens**), destacando como o pré-treinamento não supervisionado compensa a falta de dados rotulados.

## 🚀 Como Rodar o Projeto

### 1. Pré-requisitos

Certifique-se de ter as bibliotecas necessárias instaladas no seu ambiente Python:

```bash
pip install numpy matplotlib tensorflow keras
```

### 2. Execução

Salve o código como um arquivo Python e execute-o via terminal:

```bash
python [nome_do_arquivo].py
