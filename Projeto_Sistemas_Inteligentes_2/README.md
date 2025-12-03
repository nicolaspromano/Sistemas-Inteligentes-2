# 🎭 Reconhecimento de Emoções em Imagens Utilizando Redes Neurais Convolucionais

Este projeto implementa um sistema de Visão Computacional baseado em **Deep Learning** capaz de identificar e classificar emoções humanas através da webcam em tempo real.

O sistema utiliza uma **Rede Neural Convolucional (CNN)** treinada no dataset **FER2013**. O fluxo de desenvolvimento foi dividido em duas etapas estratégicas: treinamento de alta performance na nuvem (Google Colab) e inferência leve em máquina local (VS Code).

Junto aos códigos, foram incluidos o artigo e a apresentação em slides solicitadas pelo professor Dr. Rafael Gomes Mantovani.

---

## 🧠 Arquitetura do Projeto

O projeto é dividido em dois módulos principais:

1.  **Treinamento (Cloud/Colab):** Onde a "mágica pesada" acontece. Utilizamos o Google Colab para aproveitar a aceleração de GPU para treinar a CNN, realizar Data Augmentation e gerar o modelo final.
2.  **Inferência (Local/Edge):** Um script Python leve que roda no computador. Ele captura o vídeo da webcam, carrega o modelo treinado e classifica as emoções ao vivo.

### A Rede Neural (CNN)
O modelo possui uma arquitetura sequencial otimizada para imagens $48 \times 48$ pixels (escala de cinza):
* **3 Blocos Convolucionais:** Extração de características (filtros 32, 64 e 128) com ativação ReLU, BatchNormalization e MaxPooling.
* **Regularização:** Camadas de Dropout (0.25 e 0.5) para evitar overfitting.
* **Classificação:** Camadas densas finais com saída Softmax para 7 classes de emoção.

---

## 📂 Estrutura de Arquivos

Certifique-se de que sua pasta local esteja organizada da seguinte forma para o script funcionar:

```text
/meu-projeto-emocoes
│
├── projeto_final_sistemas_inteligentes.ipynb  # Notebook de treinamento (Rodar no Colab)
├── webcam_emocoes.py                          # Script da aplicação (Rodar no VS Code)
├── modelo_emocoes_fer2013.h5                  # Arquivo do modelo gerado (Download do Colab)
└── README.md
```

---

## 🛠️ Tecnologias Utilizadas
* Linguagem: Python 3.x
* Deep Learning: TensorFlow / Keras
* Visão Computacional: OpenCV (cv2)
* Manipulação de Dados: NumPy, Pandas
* Visualização: Matplotlib, Seaborn

---

## 🚀 Como Executar

Passo 1: Treinamento do Modelo (Google Colab):

1.  Abra o arquivo projeto_final_sistemas_inteligentes.ipynb no Google Colab.
2.  Certifique-se de que o ambiente de execução esteja configurado para usar GPU (Melhora drasticamente a velocidade).
3.  Execute todas as células. O notebook irá:
    * Baixar o dataset FER2013.
    * Treinar a CNN por 40 épocas.
    * Salvar o melhor modelo como melhor_modelo_emocoes.h5.
4. Faça o download do arquivo .h5 gerado ao final.

Passo 2: Execução Local (VS Code / Terminal):

1. Pré-requisitos: Instale as bibliotecas necessárias no seu ambiente local:
   ```bash
   pip install tensorflow opencv-python numpy
   ```
2. Configuração:
   * Coloque o arquivo modelo_emocoes_fer2013.h5 (baixado do Colab) na mesma pasta do script webcam_emocoes.py.
3. Rodar: Abra o terminal na pasta do projeto e execute:
   ```bash
   python webcam_emocoes.py
   ```
4. Interação:
     * A webcam abrirá e detectará seu rosto automaticamente.
     * A emoção predita e a barra de confiança aparecerão sobre sua imagem.
      * Pressione a tecla 'q' para encerrar o programa.

---

## 👨‍💻 Autores
Nicolas de Paulo Romano:

Felipe Natan Zanqueta Macaúbas

Michael Pariz Pereira
