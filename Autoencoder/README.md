# 📉 Autoencoder Simples (MLP) para Reconstrução de Imagens MNIST

Este projeto implementa um **Autoencoder** básico usando uma **Multi-Layer Perceptron (MLP)**. O objetivo é treinar a rede para comprimir e, em seguida, reconstruir as imagens do dataset **MNIST**. O *encoder* aprende uma representação de baixa dimensão que captura as características essenciais dos dígitos.

## 🧠 Arquitetura do Modelo (MLP Autoencoder)

O Autoencoder (AE) possui uma arquitetura simétrica de três camadas:

| Camada | Função | Dimensões | Parâmetros Chave |
| :--- | :--- | :--- | :--- |
| **Input (E)** | Recebe a Imagem | 784 neurônios | $28 \times 28$ pixels (MNIST) |
| **Hidden (H)** | **Encoded Representation** (Latent Space) | `--hidden` neurônios | **Compressão de Características** |
| **Output (O)** | Reconstrução da Imagem | 784 neurônios | Tenta igualar a camada Input |

As principais etapas do treinamento são:

1.  **Forward Propagation:**
    * A imagem de entrada é mapeada para a camada oculta (`H`).
    * A camada oculta (`H`) é mapeada para a camada de saída (`O`), que é a imagem reconstruída ($\hat{X}$).
    * A função de ativação utilizada é a **Sigmoid** para ambas as camadas.

2.  **Backpropagation:**
    * O erro é calculado usando a função de perda **Mean Squared Error (MSE)** entre a entrada original ($X$) e a saída reconstruída ($\hat{X}$).
    * O erro é propagado de volta para ajustar os pesos ($\mathbf{W_h}$ e $\mathbf{W_o}$) usando a **Taxa de Aprendizado** (`--lr`).

## ⚙️ Como Rodar o Projeto

### 1. Pré-requisitos

Instale as bibliotecas Python necessárias:

```bash
pip install numpy matplotlib scikit-learn
```

### 2. Execução

O script utiliza o módulo argparse para receber parâmetros de linha de comando. Use o seguinte formato para executar o treinamento:

```bash
python MLP_autoencoder.py --hidden 128 --epochs 20 --lr 0.05 --log-every 1
```
