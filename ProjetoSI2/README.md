# 🧠 Reconhecimento de Emoções com CNN (FER2013)

Este repositório contém a implementação de uma **Rede Neural Convolucional (CNN)** desenvolvida para classificar 7 tipos de emoções humanas a partir de expressões faciais.

O projeto foi desenvolvido como parte da disciplina de **Sistemas Inteligentes 2** do curso de Engenharia de Computação da **UTFPR** (2025).

## 📋 Sobre o Projeto

O objetivo é treinar um modelo de Deep Learning capaz de identificar emoções em imagens de baixa resolução (48x48 pixels) em escala de cinza, utilizando técnicas de Visão Computacional.

### Dataset
Foi utilizado o dataset **FER2013** (Facial Expression Recognition), contendo:
* **7 Classes:** Raiva, Nojo, Medo, Feliz, Neutro, Triste, Surpresa.
* **Imagens:** 48x48 pixels (Grayscale).
* **Divisão:** Treino (~28k imagens) e Teste (~7k imagens).
Disponível em: https://www.kaggle.com/datasets/msambare/fer2013

Estrutura das pastas
```bash
/dataset
    ├── train/
    │   ├── angry/
    │   ├── happy/
    │   └── ...
    └── test/
        ├── angry/
        └── ...
```

## 🚀 Tecnologias Utilizadas

* **Python**
* **TensorFlow / Keras** (Construção e treinamento da CNN)
* **Scikit-learn** (Métricas de avaliação, Pesos de Classe e Matriz de Confusão)
* **Matplotlib** (Visualização de gráficos de acurácia/perda)
* **Argparse** (Execução flexível via linha de comando)

## 🛠️ Como Rodar

### 1. Pré-requisitos
Certifique-se de ter as bibliotecas instaladas. Você pode instalar via pip:

```bash
pip install tensorflow scikit-learn matplotlib numpy
```

### 2. Compilar

## Rodar com configurações padrão
```bash
python CNN_emocoes.py --data_dir "caminho/para/dataset"
```

## Rodar personalizando épocas, batch size e pasta de saída
```bash
python CNN_emocoes.py --data_dir "./dataset" --epochs 50 --batch_size 64 --out_dir "meus_resultados"
```

## Autor

Nicolas de Paulo Romano
