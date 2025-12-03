## 🩺 CNN para Detecção de Pneumonia em Radiografias Torácicas

Este projeto implementa e treina uma **Rede Neural Convolucional (CNN) clássica** usando **Keras/TensorFlow** para a classificação binária de imagens de raios-X de tórax, distinguindo entre **Pneumonia** e **Normal**.

O código é otimizado para lidar com o desbalanceamento de classes, inclui técnicas de *data augmentation* e gera um relatório completo de métricas (Acurácia, AUC, Precisão, Revocação, F1-Score) ao final.

---

### 🧠 Arquitetura da CNN

A rede utiliza uma arquitetura simples e profunda baseada em blocos convolucionais:

1.  **Blocos Convolucionais:** Cada bloco consiste em $\text{Conv2D} \rightarrow \text{BatchNormalization} \rightarrow \text{ReLU} \rightarrow \text{MaxPooling2D}$.
    * O uso de **Batch Normalization** estabiliza o treinamento e acelera a convergência.
2.  **Global Average Pooling:** Reduz o volume de dados da última camada convolucional para um único vetor, diminuindo drasticamente o número de parâmetros na parte densa e agindo como um regularizador.
3.  **Saída:** Camada densa final com ativação **Sigmoid** para classificação binária.



---

### 🗂️ Estrutura de Pastas (Dataset)

O script espera que as imagens do dataset **Chest X-Ray (Pneumonia)** (disponível no Kaggle) estejam organizadas na seguinte estrutura. Você deve fornecer o caminho para a pasta `<DATA_DIR>` no argumento `--data_dir`:

```bash
chest_xray/
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── val/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  └── test/
      ├── NORMAL/
      └── PNEUMONIA/
```

### 📈 Técnicas Chave de Treinamento

| Técnica | Objetivo | Implementação |
| :--- | :--- | :--- |
| **Data Augmentation** | Aumentar a generalização e evitar *overfitting*. | `ImageDataGenerator` aplica rotação, zoom e *shift*. |
| **Pesos de Classe** | Lidar com o desbalanceamento de classes (Pneumonia >> Normal). | `compute_class_weight` ajusta o peso das amostras minoritárias. |
| **Callbacks** | Otimizar e controlar o treinamento. | `EarlyStopping`, `ReduceLROnPlateau` e `ModelCheckpoint` (salva o melhor modelo). |

---

### 🚀 Como Executar

O projeto utiliza argumentos de linha de comando para configuração.

#### 1. Pré-requisitos

Instale as bibliotecas necessárias:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 2. Execução 

Execute o script fornecendo o caminho para a pasta raiz dos seus dados

```bash
python cnn_pneumonia.py --data_dir "caminho/para/chest_xray" --epochs 20 --batch_size 32 --img_size 224
```
